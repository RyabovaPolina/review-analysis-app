import sys
import json
import os
import gc
import pandas as pd
import numpy as np

from concurrent.futures import ThreadPoolExecutor

from collections import Counter, defaultdict

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from preprocessing import preprocess_corpus
from ml.aspect_extractor import AspectExtractor, aggregate_aspects_by_sentiment
from ml.business_insights_analyzer import BusinessInsightAnalyzer, format_analysis_for_display

from services.model_service import load_pipeline, tfidf_cache

from services.data_filter import filter_low_quality_reviews
from services.data_processing import process_csv_streaming

from services.s3_service import (
    download_csv,
    upload_csv
)

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")


# ═════════════════════════════════════════════════════════════════════
# БАТЧ-ОБРАБОТКА ТОНАЛЬНОСТИ
# ═════════════════════════════════════════════════════════════════════

BATCH_SIZE = 256

def predict_batch(texts, model, batch_size=BATCH_SIZE):
    sentiments = []
    probabilities = []

    total = len(texts)
    total_batches = (total + batch_size - 1) // batch_size

    for i in range(0, total, batch_size):

        batch_num = i // batch_size + 1

        log(
            f"🧠 Батч {batch_num}/{total_batches} "
            f"({min(i + batch_size, total)}/{total})"
        )

        batch = texts[i:i + batch_size]

        # predict
        batch_sentiments = model.predict(batch)

        # predict_proba
        batch_probabilities = model.predict_proba(batch)

        sentiments.extend(batch_sentiments)

        probabilities.extend(
            batch_probabilities.astype(np.float32)
        )

        # cleanup памяти
        del batch
        del batch_sentiments
        del batch_probabilities

        gc.collect()

    return (
        np.array(sentiments),
        np.array(probabilities, dtype=np.float32)
    )
# ═════════════════════════════════════════════════════════════════════
# ЛОГИРОВАНИЕ В STDERR (не влияет на JSON вывод)
# ═════════════════════════════════════════════════════════════════════

def log(message):

    print(message, file=sys.stderr)


# ═════════════════════════════════════════════════════════════════════
# ГЛАВНАЯ ФУНКЦИЯ АНАЛИЗА С АСПЕКТАМИ
# ═════════════════════════════════════════════════════════════════════

def analyze_reviews_with_aspects(
    s3_key,
    review_column
):
    
    log("📥 Загружаю данные из S3...")
    df = download_csv(s3_key)
    
    log("📝 Обрабатываю отзывы потоково...")

    # Потоковый препроцессинг только для df
    processed = preprocess_corpus(df[review_column].fillna("").astype(str).tolist())

    df["processed"] = processed
    gc.collect()

    # Анализ тональности
    best_model = load_pipeline()

    sentiments, probabilities = predict_batch(
        processed,
        best_model,
        batch_size=256
    )

    class_names = best_model.named_steps["clf"].classes_
    sentiments_str = [class_names[s] if isinstance(s, (int, np.integer)) else s for s in sentiments]

    df["sentiment"] = pd.Categorical(sentiments_str)
    
    sentiment_counts = Counter(sentiments_str)
    
    # ─────────────────────────────────────────────────────────
    # ИЗВЛЕЧЕНИЕ АСПЕКТОВ
    # ─────────────────────────────────────────────────────────
    
    log("🏷️  Извлекаю аспекты из отзывов...")
    
    texts_by_sentiment = {
        st: [processed[i] for i, s in enumerate(sentiments_str) if s == st]
        for st in ["positive", "negative", "neutral"]
    }
    
    # ─────────────────────────────────────────────────────────
    # АНАЛИЗ АСПЕКТОВ
    # ─────────────────────────────────────────────────────────
    
    log("⚡ Параллельный анализ аспектов и keywords...")


    def extract_aspects_task():

        log("🏷️ Извлекаю аспекты...")

        aspect_extractor = AspectExtractor()

        aggregated_aspects = aggregate_aspects_by_sentiment(
            texts_by_sentiment,
            aspect_extractor
        )

        log("💡 Генерирую рекомендации...")

        business_analyzer = BusinessInsightAnalyzer()

        full_analysis = business_analyzer.analyze_reviews_by_aspects(
            df,
            processed_texts=processed
        )

        return aggregated_aspects, full_analysis


    def extract_keywords_task():

        log("🔑 Извлекаю ключевые слова...")

        feature_union = best_model.named_steps["features"]
        word_vectorizer = feature_union.transformer_list[0][1]

        from ml.keyword_extractor import extract_keywords_v2

        keywords = {}
        for st in texts_by_sentiment:

            try:
                keywords[st] = extract_keywords_v2(
                texts_by_sentiment,
                    st,
                    word_vectorizer,
                    top_n=10,                        
                    strategy="combined",
                    cache=tfidf_cache,
                )

            except Exception as e:

                log(f"⚠️ Ошибка keywords {st}: {e}")
                keywords[st] = {
                    "unigrams": [],
                    "bigrams": []
                }
        return keywords


    with ThreadPoolExecutor(max_workers=2) as executor:

        future_aspects = executor.submit(
                extract_aspects_task
        )

        future_keywords = executor.submit(
                extract_keywords_task
        )

        aggregated_aspects, full_analysis = (
                future_aspects.result()
        )

        keywords = future_keywords.result()        
    
    # ─────────────────────────────────────────────────────────
    # ПОДГОТОВКА РЕЗУЛЬТАТА
    # ─────────────────────────────────────────────────────────
    
    df.drop(columns=["processed"], inplace=True, errors="ignore")
    
    result_key = upload_csv(df, s3_key)
    
    # ─────────────────────────────────────────────────────────
    # ФИНАЛЬНЫЙ РЕЗУЛЬТАТ
    # ─────────────────────────────────────────────────────────
    
    result = {
        # Базовая статистика
        "summary": full_analysis["summary"],
        
        # Анализ тональности
        "sentiment_analysis": {
            "positive_count": sentiment_counts.get("positive", 0),
            "negative_count": sentiment_counts.get("negative", 0),
            "neutral_count": sentiment_counts.get("neutral", 0),
            "total_reviews": len(df),
            "avg_sentiment_score": float(probabilities.max(axis=1).mean()),
        },
        
        # Анализ аспектов
        "aspect_analysis": {
            "positive": {
                aspect: {
                    "mentions": stats["mentions"],
                    "avg_sentiment": float(stats["avg_sentiment"]),
                    "coverage": float(stats["coverage"])
                }
                for aspect, stats in aggregated_aspects.get("positive", {}).items()
            },
            "negative": {
                aspect: {
                    "mentions": stats["mentions"],
                    "avg_sentiment": float(stats["avg_sentiment"]),
                    "coverage": float(stats["coverage"])
                }
                for aspect, stats in aggregated_aspects.get("negative", {}).items()
            },
            "neutral": {
                aspect: {
                    "mentions": stats["mentions"],
                    "avg_sentiment": float(stats["avg_sentiment"]),
                    "coverage": float(stats["coverage"])
                }
                for aspect, stats in aggregated_aspects.get("neutral", {}).items()
            }
        },
        
        # Проблемные места
        "top_problems": [
            {
                "aspect": p["aspect"],
                "issue_score": float(p["issue_score"]),
                "mentions": p["mentions"],
                "coverage_pct": float(p["coverage_pct"]),
                "severity": p["severity"],
                "avg_sentiment": float(p["avg_sentiment"])
            }
            for p in full_analysis["top_problems"]
        ],
        
        # Сильные стороны
        "top_strengths": [
            {
                "aspect": s["aspect"],
                "strength_score": float(s["strength_score"]),
                "mentions": s["mentions"],
                "coverage_pct": float(s["coverage_pct"]),
                "avg_sentiment": float(s["avg_sentiment"])
            }
            for s in full_analysis["top_strengths"]
        ],
        
        # Рекомендации
        "recommendations": full_analysis["recommendations"],
        
        # Ключевые слова (старый метод)
        "keywords": keywords,
        
        # Путь к сохранённому файлу
        "result_key": result_key,
    }

    log("\n" + "=" * 120)
    log(format_analysis_for_display(full_analysis))
    log("=" * 120)
    
    return result


# ═════════════════════════════════════════════════════════════════════
# ТОЧКА ВХОДА
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    s3_key = sys.argv[1]
    review_column = sys.argv[2]
    
    result = analyze_reviews_with_aspects(
        s3_key,
        review_column
    )
    
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        )
    )
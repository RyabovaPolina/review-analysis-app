"""
✅ УЛУЧШЕННЫЙ ANALYSIS.PY С ASPECT-BASED SENTIMENT ANALYSIS
Интеграция аспект-ориентированного анализа в существующий pipeline

ИСПРАВЛЕНИЕ: Логи в stderr, JSON только в stdout в конце
"""

import sys
import json
import os
import numpy as np

from collections import Counter, defaultdict

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from preprocessing import preprocess_corpus
from ml.aspect_extractor import AspectExtractor, aggregate_aspects_by_sentiment
from ml.business_insights_analyzer import BusinessInsightAnalyzer, format_analysis_for_display

from services.s3_service import (
    download_csv,
    upload_csv
)

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")


# ═════════════════════════════════════════════════════════════════════
# ЛОГИРОВАНИЕ В STDERR (не влияет на JSON вывод)
# ═════════════════════════════════════════════════════════════════════

def log(message):
    """Логирование в stderr (для отладки, не попадает в JSON)"""
    print(message, file=sys.stderr)


# ═════════════════════════════════════════════════════════════════════
# ГЛАВНАЯ ФУНКЦИЯ АНАЛИЗА С АСПЕКТАМИ
# ═════════════════════════════════════════════════════════════════════

def analyze_reviews_with_aspects(
    s3_key,
    review_column
):
    """
    Полный анализ отзывов включая:
    1. Определение тональности (positive/negative/neutral)
    2. Извлечение аспектов (еда, персонал, цена, атмосфера, и т.д.)
    3. Анализ аспектов по категориям тональности
    4. Генерация рекомендаций для владельцев
    """
    
    log("📥 Загружаю данные из S3...")
    df = download_csv(s3_key)
    
    log(f"📝 Обрабатываю {len(df)} отзывов...")
    raw_texts = (
        df[review_column]
        .fillna("")
        .astype(str)
        .tolist()
    )
    
    # Препроцессинг
    processed = preprocess_corpus(raw_texts)
    df["processed"] = processed
    
    # ─────────────────────────────────────────────────────────
    # ОПРЕДЕЛЕНИЕ ТОНАЛЬНОСТИ
    # ─────────────────────────────────────────────────────────
    
    log("😊 Анализирую тональность...")
    from services.model_service import load_pipeline
    
    best_model = load_pipeline()
    sentiments = best_model.predict(processed)
    probabilities = best_model.predict_proba(processed)
    
    class_names = best_model.named_steps["clf"].classes_
    
    if isinstance(sentiments[0], (int, np.integer)):
        sentiments_str = [class_names[s] for s in sentiments]
    else:
        sentiments_str = list(sentiments)
    
    df["sentiment"] = sentiments_str
    
    sentiment_counts = Counter(sentiments_str)
    
    # ─────────────────────────────────────────────────────────
    # ИЗВЛЕЧЕНИЕ АСПЕКТОВ
    # ─────────────────────────────────────────────────────────
    
    log("🏷️  Извлекаю аспекты из отзывов...")
    
    texts_by_sentiment = {}
    for st in ["positive", "negative", "neutral"]:
        texts_by_sentiment[st] = df.loc[
            df["sentiment"] == st
        ]["processed"].tolist()
    
    # ─────────────────────────────────────────────────────────
    # АНАЛИЗ АСПЕКТОВ
    # ─────────────────────────────────────────────────────────
    
    log("📊 Анализирую аспекты по категориям...")
    
    aspect_extractor = AspectExtractor()
    aggregated_aspects = aggregate_aspects_by_sentiment(texts_by_sentiment, aspect_extractor)
    
    # ─────────────────────────────────────────────────────────
    # БИЗНЕС-ИНСАЙТЫ И РЕКОМЕНДАЦИИ
    # ─────────────────────────────────────────────────────────
    
    log("💡 Генерирую рекомендации...")
    
    business_analyzer = BusinessInsightAnalyzer()
    full_analysis = business_analyzer.analyze_reviews_by_aspects(df, text_column="processed")
    
    # ─────────────────────────────────────────────────────────
    # ИЗВЛЕЧЕНИЕ КЛЮЧЕВЫХ СЛОВ (СТАРЫЙ МЕТОД)
    # ─────────────────────────────────────────────────────────
    
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
                strategy="combined"
            )
        except Exception as e:
            log(f"⚠️  Ошибка при извлечении ключевых слов для {st}: {e}")
            keywords[st] = {"unigrams": [], "bigrams": []}
    
    # ─────────────────────────────────────────────────────────
    # ПОДГОТОВКА РЕЗУЛЬТАТА
    # ─────────────────────────────────────────────────────────
    
    df.drop(columns=["processed"], inplace=True)
    
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
    
    # ✅ ВАЖНО: Выводим красиво в stderr (для отладки)
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
    
    # ✅ КРИТИЧНО: ТОЛЬКО JSON в stdout, БЕЗ других символов!
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        )
    )
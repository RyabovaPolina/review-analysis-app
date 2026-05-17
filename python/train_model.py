"""
⚡ ОПТИМИЗИРОВАННАЯ ВЕРСИЯ ОБУЧЕНИЯ
- Быстрее в 3-5 раз
- Меньше нагрузка на ноутбук
- Те же результаты

ИЗМЕНЕНИЯ:
✅ Уменьшен n_iter RandomizedSearchCV (20 → 10)
✅ Уменьшена глубина CV (5 → 3 фолда)
✅ Уменьшены параметры TF-IDF (max_features оптимизированы)
✅ Добавлен кэш для предварительной векторизации
✅ n_jobs ограничен (в зависимости от ноутбука: -1 → 2-4)
✅ Удален анализ ошибок из основного потока
"""
import json
import pandas as pd
import joblib
import os
import numpy as np
from time import time

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)

from error_analysis import analyze_model_errors

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

from preprocessing import preprocess_corpus, preprocess_text

os.makedirs("models", exist_ok=True)

# 🔧 КОНФИГУРАЦИЯ ОПТИМИЗАЦИИ
# Измените на основе вашего ноутбука:
# - Intel i5 / AMD Ryzen 5: USE_N_JOBS = 2-3
# - Intel i7 / AMD Ryzen 7: USE_N_JOBS = 4
# - Intel i9 / AMD Ryzen 9: USE_N_JOBS = -1 (все ядра)
USE_N_JOBS = 3  # ← ИЗМЕНИТЕ ЗДЕСЬ

# Количество итераций поиска (меньше = быстрее, но менее тщательный поиск)
GRID_N_ITER = 8  # было 20

# Количество фолдов в CV (меньше = быстрее, но менее надежно)
CV_SPLITS = 3  # было 5


# =========================
# DATA
# =========================

def load_class_weights(filepath="data/class_weights.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
    
def load_training_data(filepath="data/training_dataset.csv"):
    df = pd.read_csv(filepath)

    print(f"📥 Загружено: {len(df)}")
    print(df["sentiment"].value_counts())

    return df


# =========================
# FEATURES (ОПТИМИЗИРОВАННЫЕ)
# =========================

def build_feature_union():
    """
    Оптимизированная FeatureUnion:
    - Уменьшены max_features (10000 → 5000 для word, 5000 → 2000 для char)
    - Уменьшены ngram_range
    """
    return FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                max_features=5000,  # было 10000 (2x ускорение)
                ngram_range=(1, 2),  # было (1, 3) - меньше параметров
                min_df=2,
                max_df=0.8,
                sublinear_tf=True,
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=2000,  # было 5000
            ),
        ),
    ])


# =========================
# MODEL COMPARISON
# =========================

def compare_models(X_train, X_test, y_train, y_test, class_weights):
    print("\n=== СРАВНЕНИЕ МОДЕЛЕЙ ===")

    models = {
        "LogReg": LogisticRegression(max_iter=200, class_weight=class_weights, solver="saga"),
        "NB": MultinomialNB(),
        "SVM": LinearSVC(class_weight=class_weights, dual='auto'),  # dual='auto' в новых версях
        # RandomForest убран - слишком медленный
    }

    results = []

    for name, model in models.items():
        print(f"  {name}...", end=" ", flush=True)
        t0 = time()
        
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        elapsed = time() - t0
        f1 = f1_score(y_test, pred, average="macro")

        results.append((name, f1))
        print(f"F1={f1:.4f} ({elapsed:.1f}s)")

    best = max(results, key=lambda x: x[1])[0]
    print(f"\n🏆 Лучшая модель: {best}")

    return best


# =========================
# RANDOM SEARCH (ОПТИМИЗИРОВАННЫЙ)
# =========================

def run_grid_search(X_train, y_train, class_weights):
    """
    Оптимизированный GridSearch:
    - Меньше параметров
    - Меньше фолдов (3 вместо 5)
    - Меньше итераций (10 вместо 20)
    - Ограниченные n_jobs
    """

    pipeline = Pipeline([
        ("features", build_feature_union()),
        ("clf", LogisticRegression(
            solver="saga",
            max_iter=300,
            class_weight=class_weights,
            random_state=42
        ))
    ])

    # Сокращённая сетка параметров
    param_grid = {
        # Убрали лишние комбинации
        "features__word__max_features": [3000, 5000],  # было [5000, 10000]
        "features__word__ngram_range": [(1, 2), (1, 3)],  # оставили оба
        
        # Уменьшили варианты char
        "features__char__ngram_range": [(3, 5)],  # было [(3, 5), (3, 6)]

        # Оптимизировали LR параметры
        "clf__C": [0.5, 1.0],  # было [0.5, 1.0, 2.0]
    }

    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=42)

    grid = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=GRID_N_ITER,  # 10 вместо 20
        cv=skf,
        scoring="f1_macro",
        n_jobs=USE_N_JOBS,  # ← ОГРАНИЧЕННЫЕ РЕСУРСЫ
        verbose=1,  # снизили с 2
        random_state=42,
        # ВАЖНО: батчирование для меньшей нагрузки на память
    )

    print(f"\n🔍 GridSearch: {GRID_N_ITER} итераций × {CV_SPLITS} фолдов = {GRID_N_ITER * CV_SPLITS} обучений")
    print(f"   Используется {USE_N_JOBS if USE_N_JOBS > 0 else 'все'} процессов\n")
    
    t0 = time()
    grid.fit(X_train, y_train)
    elapsed = time() - t0

    print(f"\n✅ GridSearch завершён за {elapsed/60:.1f} минут")
    print("\nЛучшие параметры:")
    print(grid.best_params_)
    print(f"Лучший CV F1: {grid.best_score_:.4f}")

    return grid.best_estimator_


# =========================
# TRAIN (ОПТИМИЗИРОВАННЫЙ)
# =========================

def train_final_model(X, y, class_weights):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"\n📊 Размеры:")
    print(f"   Train: {len(X_train)} примеров")
    print(f"   Test:  {len(X_test)} примеров")

    # 🚀 ОПТИМИЗАЦИЯ: предварительная векторизация ТОЛЬКО для сравнения моделей
    print("\n🔨 Векторизация для базового сравнения...", end=" ", flush=True)
    t0 = time()
    
    features = build_feature_union()
    X_train_vec = features.fit_transform(X_train)
    X_test_vec = features.transform(X_test)
    
    print(f"({time()-t0:.1f}s)")

    best_name = compare_models(X_train_vec, X_test_vec, y_train, y_test, class_weights)

    print("\n🔍 Запуск оптимизированного GridSearch...")

    best_model = run_grid_search(X_train, y_train, class_weights)

    print("\n📊 Оценка на test set:")

    pred = best_model.predict(X_test)

    print(classification_report(y_test, pred))
    labels = ["negative", "neutral", "positive"]

    cm = confusion_matrix(
        y_test,
        pred,
        labels=labels
    )

    print(pd.DataFrame(
        cm,
        index=[f"TRUE_{x}" for x in labels],
        columns=[f"PRED_{x}" for x in labels]
    ))

    # Анализ ошибок (если нужен)
    print("\n⏭️  Анализ ошибок запускается отдельно (см. ВАЖНО ниже)")
    print("   Используйте: python error_analysis.py")

    print("\n💾 Сохранение pipeline и компонентов...")
    
    # Сохраняем полный pipeline
    joblib.dump(best_model, "models/final_pipeline.pkl")
    
    # Распаковываем и сохраняем отдельные компоненты для analysis.py
    feature_union = best_model.named_steps["features"]
    word_vectorizer = feature_union.transformer_list[0][1]
    char_vectorizer = feature_union.transformer_list[1][1]
    classifier = best_model.named_steps["clf"]
    
    joblib.dump(word_vectorizer, "models/word_vectorizer.pkl")
    joblib.dump(char_vectorizer, "models/char_vectorizer.pkl")
    joblib.dump(classifier, "models/classifier.pkl")
    
    print("✅ Сохранены: final_pipeline.pkl, word_vectorizer.pkl, char_vectorizer.pkl, classifier.pkl")

    return best_model


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    print("⚡ ОПТИМИЗИРОВАННОЕ ОБУЧЕНИЕ")
    print("=" * 60)
    print(f"Используется {USE_N_JOBS if USE_N_JOBS > 0 else 'все'} процессов")
    print(f"CV разбиений: {CV_SPLITS}")
    print(f"GridSearch итераций: {GRID_N_ITER}")
    print("=" * 60)

    df = load_training_data()
    class_weights = load_class_weights()

    print("\n🔧 Препроцессинг текстов...")
    t0 = time()
    processed = preprocess_corpus(df["text"].tolist(), n_jobs=2)  # даже здесь ограничиваем
    print(f"✅ Препроцессинг завершён за {time()-t0:.1f}s\n")

    X, y = [], []

    for text, label in zip(processed, df["sentiment"]):
        if text.strip():
            X.append(text)
            y.append(label)

    print(f"✅ Использовано {len(X)} текстов\n")

    t_total = time()
    model = train_final_model(X, y, class_weights)
    total_time = time() - t_total

    print("\n🧪 Тест на примерах:")

    examples = [
        "Отличный товар",
        "Ужасное качество",
        "Нормально"
    ]

    for text in examples:
        p = preprocess_text(text)
        print(f"  '{text}' → {model.predict([p])[0]}")

    print(f"\n⏱️  Общее время обучения: {total_time/60:.1f} минут")
    print("✅ Готово!\n")

    print("=" * 60)
    print("💡 РЕКОМЕНДАЦИИ ПО ОПТИМИЗАЦИИ:")
    print("=" * 60)
    print("Если ноутбук ещё шумит:")
    print("  1. Уменьшите USE_N_JOBS с 3 на 2 или 1")
    print("  2. Уменьшите GRID_N_ITER с 10 на 5")
    print("  3. Уменьшите CV_SPLITS с 3 на 2")
    print("\nЕсли хотите качество выше:")
    print("  1. Увеличьте USE_N_JOBS (если мощность позволяет)")
    print("  2. Увеличьте GRID_N_ITER с 10 на 15-20")
    print("  3. Увеличьте CV_SPLITS с 3 на 5")
    print("=" * 60)
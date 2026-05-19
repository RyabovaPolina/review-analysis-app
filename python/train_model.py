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

USE_N_JOBS = 3 

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
# FEATURES
# =========================

def build_feature_union():
    return FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                max_features=5000,  
                ngram_range=(1, 2), 
                min_df=3,
                max_df=0.8,
                sublinear_tf=True,
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=2000, 
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
        "features__word__max_features": [3000, 5000],  # было [5000, 10000]
        "features__word__ngram_range": [(1, 2), (1, 3)],  # оставили оба
        "features__char__ngram_range": [(3, 5)],  # было [(3, 5), (3, 6)]
        "clf__C": [0.5, 1.0],  # было [0.5, 1.0, 2.0]
    }

    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=42)

    grid = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=GRID_N_ITER, 
        cv=skf,
        scoring="f1_macro",
        n_jobs=USE_N_JOBS,
        verbose=1,
        random_state=42,
    )

    print(f"\n🔍 RandomizedSearchCV: {GRID_N_ITER} итераций × {CV_SPLITS} фолдов = {GRID_N_ITER * CV_SPLITS} обучений")
    print(f"   Используется {USE_N_JOBS if USE_N_JOBS > 0 else 'все'} процессов\n")
    
    t0 = time()
    grid.fit(X_train, y_train)
    elapsed = time() - t0

    print(f"\n✅ RandomizedSearchCV завершён за {elapsed/60:.1f} минут")
    print("\nЛучшие параметры:")
    print(grid.best_params_)
    print(f"Лучший CV F1: {grid.best_score_:.4f}")

    return grid.best_estimator_


# =========================
# TRAIN
# =========================

def train_final_model(X, y, class_weights):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"\n📊 Размеры:")
    print(f"   Train: {len(X_train)} примеров")
    print(f"   Test:  {len(X_test)} примеров")

    print("\n🔨 Векторизация для базового сравнения...", end=" ", flush=True)
    t0 = time()
    
    print(f"({time()-t0:.1f}s)")

    print("\n🔍 Запуск оптимизированного RandomizedSearchCV...")

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

    print("ОБУЧЕНИЕ")
    print("=" * 60)
    print(f"Используется {USE_N_JOBS if USE_N_JOBS > 0 else 'все'} процессов")
    print(f"CV разбиений: {CV_SPLITS}")
    print(f"GridSearch итераций: {GRID_N_ITER}")
    print("=" * 60)

    df = load_training_data()
    class_weights = load_class_weights()

    print("\n🔧 Препроцессинг текстов...")
    t0 = time()
    processed = preprocess_corpus(df["text"].tolist(), n_jobs=2) 
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

    print(f"\n⏱️  Общее время обучения: {total_time/60:.1f} минут")
    print("Готово!\n")
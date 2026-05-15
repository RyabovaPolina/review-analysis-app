"""
ФИНАЛЬНЫЙ СКРИПТ ОБУЧЕНИЯ
- Без утечек данных
- Сравнение моделей
- RandomizedSearchCV для LogisticRegression (лучшая модель)
- StratifiedKFold
"""
import json
import pandas as pd
import joblib
import os

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
                max_features=10000,
                ngram_range=(1, 3),
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
                max_features=5000,
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
        "SVM": LinearSVC(class_weight=class_weights),
        "RF": RandomForestClassifier(n_estimators=100, n_jobs=-1)
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)


        f1 = f1_score(y_test, pred, average="macro")


        results.append((name, f1))
        print(f"{name}: F1={f1:.4f}")

    best = max(results, key=lambda x: x[1])[0]
    print(f"\n🏆 Лучшая модель: {best}")

    return best


# =========================
# RANDOM SEARCH (LR)
# =========================

def run_grid_search(X_train, y_train, class_weights):

    pipeline = Pipeline([
        ("features", build_feature_union()),
        ("clf", LogisticRegression(
            solver="saga",
            max_iter=300,
            class_weight=class_weights,
            random_state=42
        ))
    ])

    param_grid = {
        # WORD TF-IDF
        "features__word__max_features": [5000, 10000],
        "features__word__ngram_range": [(1, 2), (1, 3)],
        "features__word__min_df": [1, 2],

        # CHAR TF-IDF
        "features__char__ngram_range": [(3, 5), (3, 6)],

        # Logistic Regression
        "clf__C": [0.5, 1.0, 2.0]
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=20,
        cv=skf,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=2,
        random_state=42
    )

    grid.fit(X_train, y_train)

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

    print("\nПримеры:")
    for i in range(5):
        print(X_train[i])

    # базовая векторизация для сравнения
    features = build_feature_union()
    X_train_vec = features.fit_transform(X_train)
    X_test_vec = features.transform(X_test)

    best_name = compare_models(X_train_vec, X_test_vec, y_train, y_test, class_weights)

    print("\n🔍 Запуск GridSearch для LogisticRegression...")

    best_model = run_grid_search(X_train, y_train, class_weights)

    print("\n📊 Оценка на test:")

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

    analyzer = analyze_model_errors(best_model, X_test, y_test, pred)

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

    df = load_training_data()
    class_weights = load_class_weights()

    print("\n🔧 Препроцессинг...")
    processed = preprocess_corpus(df["text"].tolist())

    X, y = [], []

    for text, label in zip(processed, df["sentiment"]):
        if text.strip():
            X.append(text)
            y.append(label)

    model = train_final_model(X, y, class_weights)

    print("\n🧪 Тест:")

    examples = [
        "Отличный товар",
        "Ужасное качество",
        "Нормально"
    ]

    for text in examples:
        p = preprocess_text(text)
        print(text, "->", model.predict([p])[0])

    print("\n✅ Готово")
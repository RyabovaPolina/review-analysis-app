"""
Скрипт для обучения модели классификации тональности
Эксперимент 3.3.5 — word + char n-grams
"""

import pandas as pd
import numpy as np
import joblib
import os

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from preprocessing import preprocess_corpus, preprocess_text


# создаём директорию для моделей
os.makedirs("models", exist_ok=True)


def load_training_data(filepath="data/training_dataset.csv"):

    print(f"📥 Загрузка данных из {filepath}...")

    df = pd.read_csv(filepath)

    print(f"✅ Загружено {len(df)} отзывов")

    print("📊 Распределение классов:")

    print(df["sentiment"].value_counts())

    return df


def compare_models(X_train, X_test, y_train, y_test):

    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 70)

    models = {

        "LogisticRegression": LogisticRegression(
            solver="saga",
            max_iter=200,
            class_weight="balanced",
            random_state=42
        ),

        "MultinomialNB": MultinomialNB(),

        "LinearSVC": LinearSVC(
            class_weight="balanced",
            random_state=42,
            max_iter=2000
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    results = []

    for name, model in models.items():

        print(f"\n🔧 Обучение {name}...")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        f1_macro = f1_score(
            y_test,
            y_pred,
            average="macro"
        )

        f1_weighted = f1_score(
            y_test,
            y_pred,
            average="weighted"
        )

        results.append({

            "Model": name,
            "Accuracy": accuracy,
            "F1-Score (macro)": f1_macro,
            "F1-Score (weighted)": f1_weighted
        })

        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score (macro): {f1_macro:.4f}")
        print(f"  F1-Score (weighted): {f1_weighted:.4f}")

    print("\n" + "=" * 70)
    print("ИТОГОВАЯ ТАБЛИЦА СРАВНЕНИЯ")
    print("=" * 70)

    df_results = pd.DataFrame(results)

    print(df_results.to_string(index=False))

    best_model_name = df_results.loc[
        df_results["F1-Score (macro)"].idxmax(),
        "Model"
    ]

    print(f"\n🏆 Лучшая модель: {best_model_name}")

    return models[best_model_name], best_model_name


def train_final_model(X, y):

    print("\n" + "=" * 70)
    print("ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ")
    print("=" * 70)

    print("\n📊 Создание TF-IDF признаков...")

    # WORD n-grams

    word_vectorizer = TfidfVectorizer(

        max_features=10000,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.8,
        sublinear_tf=True
    )

    X_word = word_vectorizer.fit_transform(X)

    # CHAR n-grams

    char_vectorizer = TfidfVectorizer(

        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=5000
    )

    X_char = char_vectorizer.fit_transform(X)

    # объединение признаков

    X_vectorized = hstack([X_word, X_char])

    print(
        f"✅ Размерность матрицы признаков: "
        f"{X_vectorized.shape}"
    )

    # train / test

    X_train, X_test, y_train, y_test = train_test_split(

        X_vectorized,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    best_model, best_model_name = compare_models(

        X_train,
        X_test,
        y_train,
        y_test
    )

    print(f"\n📋 Детальный отчёт для {best_model_name}:")

    y_pred = best_model.predict(X_test)

    print(classification_report(y_test, y_pred))

    print("\n📊 Confusion Matrix:")

    cm = confusion_matrix(y_test, y_pred)

    print(cm)

    print("\n🔄 Кросс-валидация (5-fold)...")

    cv_scores = cross_val_score(

        best_model,
        X_vectorized,
        y,
        cv=5,
        scoring="f1_macro"
    )

    print(f"F1 по фолдам: {cv_scores}")

    print(
        f"Средний F1: "
        f"{cv_scores.mean():.4f} "
        f"(+/- {cv_scores.std() * 2:.4f})"
    )

    print("\n🔧 Переобучение на всех данных...")

    final_model = best_model

    final_model.fit(X_vectorized, y)

    print("\n💾 Сохранение моделей...")

    joblib.dump(
        word_vectorizer,
        "models/word_vectorizer.pkl"
    )

    joblib.dump(
        char_vectorizer,
        "models/char_vectorizer.pkl"
    )

    joblib.dump(
        final_model,
        "models/classifier.pkl"
    )

    print("✅ Модели сохранены")

    if hasattr(final_model, "coef_"):

        print("\n🔍 Топ признаков")

        word_features = (
            word_vectorizer
            .get_feature_names_out()
        )

        char_features = (
            char_vectorizer
            .get_feature_names_out()
        )

        feature_names = np.concatenate(
            [word_features, char_features]
        )

        for idx, class_label in enumerate(
            final_model.classes_
        ):

            print(f"\nКласс: {class_label}")

            coefs = final_model.coef_[idx]

            top_indices = np.argsort(
                coefs
            )[-10:][::-1]

            for i in top_indices:

                print(
                    f"{feature_names[i]}: "
                    f"{coefs[i]:.4f}"
                )

    return word_vectorizer, char_vectorizer, final_model


if __name__ == "__main__":

    print(
        "🚀 ОБУЧЕНИЕ МОДЕЛИ "
        "КЛАССИФИКАЦИИ ТОНАЛЬНОСТИ"
    )

    print("=" * 70)

    df = load_training_data()

    print("\n🔧 Препроцессинг текстов...")

    X = preprocess_corpus(
        df["text"].tolist()
    )

    y = df["sentiment"].tolist()

    word_vectorizer, char_vectorizer, model = \
        train_final_model(X, y)

    print("\n" + "=" * 70)
    print("ТЕСТИРОВАНИЕ НА ПРИМЕРАХ")
    print("=" * 70)

    test_examples = [

        "Отличный товар! Очень доволен покупкой.",
        "Ужасное качество, не рекомендую!",
        "Нормально, ничего особенного.",
        "Не плохо, но могло быть лучше",
        "Цена высокая, но качество супер!"
    ]

    for text in test_examples:

        processed = preprocess_text(text)

        X_word = word_vectorizer.transform(
            [processed]
        )

        X_char = char_vectorizer.transform(
            [processed]
        )

        vectorized = hstack(
            [X_word, X_char]
        )

        prediction = model.predict(
            vectorized
        )[0]

        proba = model.predict_proba(
            vectorized
        )[0]

        print(f"\nТекст: {text}")
        print(f"Предсказание: {prediction}")

        print(
            "Вероятности:",
            dict(
                zip(
                    model.classes_,
                    proba
                )
            )
        )

    print("\n✅ Обучение завершено!")
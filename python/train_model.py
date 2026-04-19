"""
Скрипт для обучения модели классификации тональности
"""
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from preprocessing import preprocess_corpus

# Создаём директорию для моделей
os.makedirs('models', exist_ok=True)


def load_training_data(filepath='data/training_dataset.csv'):
    """
    Загрузка тренировочных данных
    
    Ожидаемый формат CSV:
    text,sentiment
    "Отличный товар",positive
    "Ужасно",negative
    """
    print(f"📥 Загрузка данных из {filepath}...")
    df = pd.read_csv(filepath)
    
    print(f"✅ Загружено {len(df)} отзывов")
    print(f"📊 Распределение классов:")
    print(df['sentiment'].value_counts())
    
    return df


def compare_models(X_train, X_test, y_train, y_test):
    """
    Сравнение нескольких классификаторов
    """
    print("\n" + "="*70)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("="*70)
    
    models = {
        'LogisticRegression': LogisticRegression(
            solver='saga',
            max_iter=200,
            class_weight='balanced',
            random_state=42
        ),
        'MultinomialNB': MultinomialNB(),
        'LinearSVC': LinearSVC(
            class_weight='balanced',
            random_state=42,
            max_iter=2000
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\n🔧 Обучение {name}...")
        
        # Обучение
        model.fit(X_train, y_train)
        
        # Предсказания
        y_pred = model.predict(X_test)
        
        # Метрики
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'F1-Score (macro)': f1_macro,
            'F1-Score (weighted)': f1_weighted
        })
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score (macro): {f1_macro:.4f}")
        print(f"  F1-Score (weighted): {f1_weighted:.4f}")
    
    # Таблица результатов
    print("\n" + "="*70)
    print("ИТОГОВАЯ ТАБЛИЦА СРАВНЕНИЯ")
    print("="*70)
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    # Лучшая модель
    best_model_name = df_results.loc[df_results['F1-Score (macro)'].idxmax(), 'Model']
    print(f"\n🏆 Лучшая модель: {best_model_name}")
    
    return models[best_model_name], best_model_name


def train_final_model(X, y):
    """
    Обучение финальной модели на всех данных
    """
    print("\n" + "="*70)
    print("ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ")
    print("="*70)
    
    # Векторизатор TF-IDF
    print("\n📊 Создание TF-IDF векторизатора...")
    vectorizer = TfidfVectorizer(
        max_features=5000,      # Максимум 5000 признаков
        ngram_range=(1, 1),     # uni-grams
        min_df=2,               # Слово должно встретиться минимум в 2 документах
        max_df=0.8,             # Игнорировать слова, встречающиеся более чем в 80% документов
        sublinear_tf=True       # Применить логарифмическое масштабирование TF
    )
    
    X_vectorized = vectorizer.fit_transform(X)
    print(f"✅ Размерность матрицы признаков: {X_vectorized.shape}")
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    # Сравнение моделей
    best_model, best_model_name = compare_models(X_train, X_test, y_train, y_test)
    
    # Детальный отчёт по лучшей модели
    print(f"\n📋 Детальный отчёт для {best_model_name}:")
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    print("\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Кросс-валидация
    print("\n🔄 Кросс-валидация (5-fold)...")
    cv_scores = cross_val_score(best_model, X_vectorized, y, cv=5, scoring='f1_macro')
    print(f"  F1-Score (macro) по фолдам: {cv_scores}")
    print(f"  Средний F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Переобучаем на всех данных
    print("\n🔧 Переобучение на всех данных...")
    final_model = LogisticRegression(
        solver='saga',
        max_iter=200,
        class_weight='balanced',
        random_state=42
    )
    final_model.fit(X_vectorized, y)
    
    # Сохранение
    print("\n💾 Сохранение моделей...")
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
    joblib.dump(final_model, 'models/classifier.pkl')
    print("✅ Модели сохранены в директорию models/")
    
    # Топ важных признаков
    print("\n🔍 Топ-20 признаков для каждого класса:")
    feature_names = vectorizer.get_feature_names_out()
    
    for idx, class_label in enumerate(final_model.classes_):
        print(f"\n  Класс '{class_label}':")
        coefs = final_model.coef_[idx]
        top_indices = np.argsort(coefs)[-20:][::-1]
        top_features = [(feature_names[i], coefs[i]) for i in top_indices]
        for feature, coef in top_features[:10]:
            print(f"    {feature}: {coef:.4f}")
    
    return vectorizer, final_model


if __name__ == "__main__":
    print("🚀 ОБУЧЕНИЕ МОДЕЛИ КЛАССИФИКАЦИИ ТОНАЛЬНОСТИ")
    print("="*70)
    
    # 1. Загрузка данных
    df = load_training_data()
    
    # 2. Препроцессинг
    print("\n🔧 Препроцессинг текстов...")
    X = preprocess_corpus(df['text'].tolist())
    y = df['sentiment'].tolist()
    
    # 3. Обучение
    vectorizer, model = train_final_model(X, y)
    
    # 4. Тестирование на примерах
    print("\n" + "="*70)
    print("ТЕСТИРОВАНИЕ НА ПРИМЕРАХ")
    print("="*70)
    
    test_examples = [
        "Отличный товар! Очень доволен покупкой.",
        "Ужасное качество, не рекомендую!",
        "Нормально, ничего особенного.",
        "Не плохо, но могло быть лучше",
        "Цена высокая, но качество супер!"
    ]
    
    from preprocessing import preprocess_text
    
    for text in test_examples:
        processed = preprocess_text(text)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized)[0]
        
        print(f"\nТекст: {text}")
        print(f"Предсказание: {prediction}")
        print(f"Вероятности: {dict(zip(model.classes_, proba))}")
    
    print("\n✅ Обучение завершено!")
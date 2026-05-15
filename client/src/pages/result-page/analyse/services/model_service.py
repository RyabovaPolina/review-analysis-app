import os
import joblib

"""
✅ ИСПРАВЛЕННЫЙ model_service.py
Загружает модели и pipeline для анализа отзывов
"""

# Определяем пути к моделям
SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Поднимаемся на 2 уровня: services -> analyse -> project_root
BASE_DIR = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        "..",
        ".."
    )
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)


def load_pipeline():
    """
    ✅ ИСПРАВЛЕНИЕ П.1, П.2, П.3:
    
    Загружает полный обученный pipeline с правильными параметрами.
    Pipeline содержит:
      - FeatureUnion с word TfidfVectorizer и char TfidfVectorizer
      - LogisticRegression классификатор
    
    Все компоненты обучены вместе и имеют совместимые параметры.
    
    Это гарантирует:
    1. ✅ Правильное сохранение компонентов (П.1)
    2. ✅ Совместимые параметры векторайзеров (П.2)
    3. ✅ Правильную комбинацию признаков (П.3)
    
    Returns:
        Pipeline: Обученный pipeline с features + classifier
    """
    pipeline_path = os.path.join(
        MODELS_DIR,
        "final_pipeline.pkl"
    )
    
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(
            f"Pipeline не найден: {pipeline_path}\n"
            f"Запустите train_model.py для создания моделей"
        )
    
    pipeline = joblib.load(pipeline_path)
    return pipeline


def load_models():
    """
    Загружает отдельные компоненты для обратной совместимости.
    
    ⚠️ УСТАРЕЛАЯ ФУНКЦИЯ - используйте load_pipeline() вместо этого
    
    Returns:
        tuple: (word_vectorizer, char_vectorizer, classifier)
    """
    word_vectorizer = joblib.load(
        os.path.join(
            MODELS_DIR,
            "word_vectorizer.pkl"
        )
    )

    char_vectorizer = joblib.load(
        os.path.join(
            MODELS_DIR,
            "char_vectorizer.pkl"
        )
    )

    classifier = joblib.load(
        os.path.join(
            MODELS_DIR,
            "classifier.pkl"
        )
    )

    return (
        word_vectorizer,
        char_vectorizer,
        classifier
    )
import os
import sys
import joblib
import threading
import pickle
import hashlib

"""
✅ Потокобезопасный кэш pipeline и TF-IDF матриц
"""

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ----------------------- Pipeline Cache -----------------------
class ModelCache:
    """Потокобезопасный кэш моделей в памяти"""
    
    def __init__(self):
        self._cache = {}
        self._lock = threading.RLock()
    
    def get_pipeline(self):
        """Ленивая загрузка + кэширование"""
        if "pipeline" in self._cache:
            return self._cache["pipeline"]
        
        with self._lock:
            if "pipeline" in self._cache:
                return self._cache["pipeline"]
            
            pipeline_path = os.path.join(MODELS_DIR, "final_pipeline.pkl")
            print("⏳ Загружаю модель...", file=sys.stderr)
            pipeline = joblib.load(pipeline_path)
            self._cache["pipeline"] = pipeline
            print("✅ Модель загружена!", file=sys.stderr)
            return pipeline

_model_cache = ModelCache()

def load_pipeline():
    """Возвращает кэшированный pipeline"""
    return _model_cache.get_pipeline()


# ----------------------- TF-IDF Cache -----------------------
class VectorizerCache:
    """Кэширует трансформированные TF-IDF матрицы"""
    
    def __init__(self, cache_dir="/tmp/vectorizer_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._lock = threading.RLock()
    
    def get_cache_key(self, texts, vectorizer):
        content_hash = hashlib.md5(str(texts).encode()).hexdigest()
        return os.path.join(self.cache_dir, f"tfidf_{content_hash}.pkl")
    
    def transform_cached(self, texts, vectorizer):
        cache_key = self.get_cache_key(texts, vectorizer)
        
        with self._lock:
            if os.path.exists(cache_key):
                return pickle.load(open(cache_key, "rb"))
            
            result = vectorizer.transform(texts)
            pickle.dump(result, open(cache_key, "wb"))
            return result

tfidf_cache = VectorizerCache()


# ----------------------- Старый метод для обратной совместимости -----------------------
def load_models():
    word_vectorizer = joblib.load(os.path.join(MODELS_DIR, "word_vectorizer.pkl"))
    char_vectorizer = joblib.load(os.path.join(MODELS_DIR, "char_vectorizer.pkl"))
    classifier = joblib.load(os.path.join(MODELS_DIR, "classifier.pkl"))
    return word_vectorizer, char_vectorizer, classifier
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
import heapq
import os
import pickle

import hashlib


# ═════════════════════════════════════════════════════════════════════
# СТОП-СЛОВА И ФИЛЬТРЫ
# ═════════════════════════════════════════════════════════════════════

# Расширенные русские стоп-слова (служебная лексика, глаголы-связки)
RUSSIAN_STOPWORDS = {
    # Предлоги
    "в", "на", "по", "с", "к", "от", "из", "для", "без", "об", "обо",
    "перед", "перо", "при", "про", "если", "как", "что", "когда",
    
    # Союзы
    "и", "или", "а", "но", "то", "что", "как", "если", "чтобы", "хотя",
    "поскольку", "потому", "также", "еще", "ещё", "же", "ли", "бы",
    
    # Частицы
    "не", "ни", "уж", "только", "лишь", "даже", "вот", "ну", "вот",
    
    # Местоимения
    "я", "ты", "он", "она", "оно", "мы", "вы", "они", "это", "то",
    "который", "какой", "где", "когда", "почему", "кто", "что", "какой",
    "тот", "такой", "весь", "каждый", "любой", "свой", "сам",
    
    # Глаголы связки (особенно важно)
    "быть", "есть", "был", "была", "было", "были", "быте", "были", "быстро",
    "есть", "есть", "ест", "едят",
    
    # Вспомогательные глаголы
    "может", "нужно", "можно", "надо", "хотел", "хочет", "хотят",
    "должен", "должна", "должно", "должны",
    
    # Числа и буквы
    "один", "два", "три", "два", "два", "пять", "раз", "разу",
    
    # Показатели (очень частые, мало информативные)
    "там", "тут", "здесь", "куда", "откуда", "туда", "отсюда", "вверх", "вниз",
    "вчера", "завтра", "сегодня", "ночью", "днём", "утром", "вечером",
    
    # Модальные слова (служебные)
    "похоже", "кажется", "видимо", "вероятно", "наверное", "может",
    
    # Звукоподражания и междометия
    "ой", "ах", "ох", "ой", "ну", "блин", "черт", "да", "нет", "угу", "ага",
    
    # Английские артикли (могут быть в тексте)
    "a", "the", "is", "are", "am", "be", "been", "being",
}


# Фильтры для выделения ключевых слов
def is_valid_word(token: str, min_len: int = 3) -> bool:
    """
    Проверка, является ли токен значимым словом.
    
    Фильтрует:
    - Стоп-слова
    - Очень короткие слова
    - Числа (чистые)
    - Токены с доменными признаками (обрабатываются отдельно)
    
    Пропускает:
    - Слова с доменными маркерами (товар_*, качество_* и т.д.)
    - Отрицания (не_*, совсем_* и т.д.)
    """
    if len(token) < min_len:
        return False
    
    if token in RUSSIAN_STOPWORDS:
        return False
    
    # Пропускаем чистые числа
    if token.isdigit():
        return False
    
    # Пропускаем английские буквы (шум в русском тексте)
    if token.isascii() and len(token) <= 2:
        return False
    
    # Сохраняем доменные признаки (они информативны)
    # Они обозначают важные аспекты (качество, цена, доставка)
    
    return True

def is_meaningful_ngram(text: str, min_words: int = 2) -> bool:

    # Список бесполезных биграмм (очень частые, служебные)
    USELESS_BIGRAMS = {
        "что то", "как будто", "из за", "так как", "то что",
        "это то", "это как", "как то", "но это", "и то", "или то",
        "что это", "как это", "куда то", "зачем то", "почему то",
        "да это", "нет это", "вот это", "вот как", "вот что",
        "все что", "все как", "тот кто", "то кто", "кто то",
        "это то", "это что", "это как", "ну это", "вот и",
        "а то", "или то", "и то", "но то", "если то",
        "не что", "не как", "не то", "не это",
        "все равно", "все равно", "нет нет", "да да",
    }
    
    words = text.split()
    
    # Отфильтровываем слова состоящие из чистых стоп-слов
    meaningful_words = [w for w in words if w not in RUSSIAN_STOPWORDS]
    
    if len(meaningful_words) < min_words:
        return False
    
    # Специальная проверка для 2-грамм
    if len(words) == 2 and text in USELESS_BIGRAMS:
        return False
    
    return True


class VectorizerCache:

    def __init__(self, cache_dir="/tmp/vectorizer_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_cache_key(self, texts, vectorizer):
        """Генерирует уникальный ключ для кэша"""
        content_hash = hashlib.md5(str(texts).encode()).hexdigest()
        return os.path.join(self.cache_dir, f"tfidf_{content_hash}.pkl")

    def transform_cached(self, texts, vectorizer):
        """Возвращает кэшированную матрицу если существует"""
        cache_key = self.get_cache_key(texts, vectorizer)

        if os.path.exists(cache_key):
            return pickle.load(open(cache_key, 'rb'))

        result = vectorizer.transform(texts)
        pickle.dump(result, open(cache_key, 'wb'))
        return result


def extract_keywords_v2(
    texts_by_sentiment,
    sentiment_type,
    word_vectorizer,
    top_n=15,
    strategy="combined",
    verbose=False,
    cache=None 
):
    group = texts_by_sentiment.get(sentiment_type, [])
    other = [t for k, v in texts_by_sentiment.items() if k != sentiment_type for t in v]

    if not group:
        return {"unigrams": [], "bigrams": [], "trigrams": [], "domain_keywords": []}

    feature_names = word_vectorizer.get_feature_names_out()

    # ───────────── Трансформация с кэшем ─────────────
    if cache is None:
        X_group = word_vectorizer.transform(group)
        X_other = word_vectorizer.transform(other) if other else None
    else:
        X_group = cache.transform_cached(group, word_vectorizer)
        X_other = cache.transform_cached(other, word_vectorizer) if other else None

    freq_group = np.asarray(X_group.sum(axis=0)).flatten() / (len(group) + 1e-6)
    freq_other = np.asarray(X_other.sum(axis=0)).flatten() / (len(other) + 1e-6) if X_other is not None else np.zeros_like(freq_group)

    # ───────────── Метрики ─────────────
    tfidf_scores = np.asarray(X_group.mean(axis=0)).flatten()
    specificity = freq_group / (freq_other + 1e-6)
    bm25_scores = np.log1p(freq_group / (freq_other + 1e-6))

    labels = np.array([1]*len(group) + [0]*len(other))
    X_full = word_vectorizer.transform(group + other)
    X_arr = X_full.toarray()
    pw_and_s = (X_arr * labels[:, np.newaxis]).sum(axis=0) / len(labels)
    pw = X_arr.sum(axis=0) / len(labels)
    ps = labels.mean()
    pmi_scores = np.log((pw_and_s + 1e-10) / (pw * ps + 1e-10))

    if strategy == "combined":
        combined_scores = freq_group * np.sqrt(specificity)
    elif strategy == "tfidf":
        combined_scores = tfidf_scores
    elif strategy == "frequency":
        combined_scores = freq_group
    elif strategy == "specificity":
        combined_scores = specificity
    elif strategy == "bm25":
        combined_scores = bm25_scores
    elif strategy == "pmi":
        combined_scores = pmi_scores
    else:
        combined_scores = freq_group * np.sqrt(specificity)

    # ───────────── Heap top-N ─────────────
    heaps = {"unigrams": [], "bigrams": [], "trigrams": [], "domain_keywords": []}

    for idx, feature in enumerate(feature_names):
        score = combined_scores[idx]
        if score < 1e-8:
            continue
        word_count = len(feature.split())

        if "_" in feature and not feature.startswith("не_"):
            heap = heaps["domain_keywords"]
        elif word_count == 1:
            if not is_valid_word(feature):
                continue
            heap = heaps["unigrams"]
        elif word_count == 2:
            if not is_meaningful_ngram(feature, min_words=2):
                continue
            heap = heaps["bigrams"]
        elif word_count >= 3:
            if not is_meaningful_ngram(feature, min_words=3):
                continue
            heap = heaps["trigrams"]
        else:
            continue

        if len(heap) < top_n:
            heapq.heappush(heap, (score, idx, feature))
        elif score > heap[0][0]:
            heapq.heapreplace(heap, (score, idx, feature))

    results = {}
    for key, heap in heaps.items():
        sorted_items = sorted(heap, reverse=True)
        results[key] = [
            {"text": f, "score": round(float(s), 6), "frequency": round(float(freq_group[idx]), 6),
             "specificity": round(float(specificity[idx]), 2)}
            for s, idx, f in sorted_items
        ]

    results["metadata"] = {
        "sentiment": sentiment_type,
        "group_size": len(group),
        "total_other_size": len(other),
        "strategy": strategy,
        "total_unique_features": len(feature_names),
    }

    if verbose:
        print(f"\n📊 Статистика для {sentiment_type}:")
        for k in ["unigrams", "bigrams", "trigrams", "domain_keywords"]:
            print(f"  {k}: {len(results[k])} элементов")

    return results
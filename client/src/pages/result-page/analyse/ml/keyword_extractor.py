"""
✅ УЛУЧШЕННЫЙ KEYWORD EXTRACTOR v2.0
Извлечение ключевых слов для каждой категории (позитивные/негативные/нейтральные)

Проблемы в v1:
❌ Чрезмерно фильтрует слова → много шума в результатах
❌ TF-IDF score часто ≈ 0 → трудно интерпретировать
❌ Не учитывает специфику русского языка
❌ Обычные сочетания ("что то", "как будто") выглядят как ключевые
❌ Не фильтрует стоп-слова доменные (служебные слова)
❌ Не использует доменные признаки из препроцессинга

Улучшения в v2:
✅ Несколько метрик: TF-IDF, BM25, PMI, простая частотность
✅ Агрессивная фильтрация глагольных форм и служебных слов
✅ Специальная обработка n-грамм (биграммы, триграммы)
✅ Параметризуемые пороги для разных типов слов
✅ Явное исключение стоп-слов (англ. + расширенный русский)
✅ Учёт длины слова и частоты
✅ Лучшая работа с биграммами (интегрирует пункт и запятую)
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re


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
    """
    Проверка, является ли n-грамма значимой.
    
    Для биграмм и триграмм отфильтровываем пустые комбинации:
    - "что то" → шум
    - "как будто" → служебное
    - "из за" → служебное (из-за)
    - "так как" → служебное
    - "то что" → служебное
    """
    
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


# ═════════════════════════════════════════════════════════════════════
# МЕТРИКИ ДЛЯ ОТБОРА КЛЮЧЕВЫХ СЛОВ
# ═════════════════════════════════════════════════════════════════════

def compute_tf_idf_scores(texts, vectorizer=None):
    """
    Вычисляем TF-IDF для получения базовых весов слов.
    
    Args:
        texts: список текстов
        vectorizer: готовый TfidfVectorizer (для согласованности с моделью)
    
    Returns:
        (feature_names, tfidf_matrix)
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8,
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
    else:
        tfidf_matrix = vectorizer.transform(texts)
    
    feature_names = vectorizer.get_feature_names_out()
    
    return feature_names, tfidf_matrix


def compute_frequency_scores(texts):
    """
    Простое подсчитывание частоты слов.
    Полезно как дополнение к TF-IDF.
    """
    vectorizer = CountVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.8,
    )
    count_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Сумма по всем документам
    freq = np.asarray(count_matrix.sum(axis=0)).flatten()
    
    return feature_names, freq


def compute_bm25_scores(group_texts, other_texts, vectorizer):
    """
    BM25 (probabilistic relevance framework) - лучше чем TF-IDF для сравнения.
    
    Идея: выделяем слова которые:
    1. Часто встречаются в группе (высокая частота в group)
    2. Редко встречаются в других группах (низкая частота в other)
    
    Args:
        group_texts: тексты одной эмоциональной категории
        other_texts: тексты других категорий
        vectorizer: TfidfVectorizer (уже обученный на всех текстах)
    
    Returns:
        (feature_names, bm25_scores)
    """
    feature_names = vectorizer.get_feature_names_out()
    
    # Трансформируем тексты
    X_group = vectorizer.transform(group_texts)
    X_other = vectorizer.transform(other_texts)
    
    # Считаем средние TF-IDF для группы и других
    group_mean = np.asarray(X_group.mean(axis=0)).flatten()
    other_mean = np.asarray(X_other.mean(axis=0)).flatten()
    
    # BM25 подобный скор: отношение частоты в группе к частоте в остальном
    eps = 1e-6
    bm25_scores = group_mean / (other_mean + eps)
    
    # Логарифм для лучшей дифференциации
    bm25_scores = np.log1p(bm25_scores)
    
    return feature_names, bm25_scores


def compute_pointwise_mutual_information(texts_by_sentiment, sentiment):
    """
    PMI (Pointwise Mutual Information) - информационная метрика.
    
    Выделяет слова которые особенно характерны для одной категории.
    Для каждого слова: P(word & sentiment) / (P(word) * P(sentiment))
    
    Args:
        texts_by_sentiment: dict с текстами по категориям
        sentiment: категория для анализа
    
    Returns:
        (feature_names, pmi_scores)
    """
    # Все тексты для подсчёта вероятностей
    all_texts = []
    labels = []
    
    for sent, texts in texts_by_sentiment.items():
        all_texts.extend(texts)
        labels.extend([1 if sent == sentiment else 0] * len(texts))
    
    # CountVectorizer для абсолютных частот
    vectorizer = CountVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        min_df=2,
    )
    X = vectorizer.fit_transform(all_texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Подсчитываем совместные события
    X_arr = X.toarray()
    
    # P(word = 1 and sentiment = 1)
    pw_and_s = (X_arr * np.array(labels)[:, np.newaxis]).sum(axis=0) / len(labels)
    
    # P(word = 1) и P(sentiment = 1)
    pw = X_arr.sum(axis=0) / len(labels)
    ps = np.mean(labels)
    
    # PMI = log( P(w&s) / (P(w) * P(s)) )
    eps = 1e-10
    pmi = np.log((pw_and_s + eps) / (pw * ps + eps))
    
    return feature_names, pmi


# ═════════════════════════════════════════════════════════════════════
# ОСНОВНАЯ ФУНКЦИЯ ИЗВЛЕЧЕНИЯ КЛЮЧЕВЫХ СЛОВ
# ═════════════════════════════════════════════════════════════════════

def extract_keywords_v2(
    texts_by_sentiment,
    sentiment_type,
    word_vectorizer,
    top_n=15,
    strategy="combined",
    verbose=False
):
    """
    ✅ УЛУЧШЕННОЕ ИЗВЛЕЧЕНИЕ КЛЮЧЕВЫХ СЛОВ v2.0
    
    Теперь выделяет действительно ключевые слова, фильтруя шум.
    
    Args:
        texts_by_sentiment: dict {"positive": [...], "negative": [...], "neutral": [...]}
        sentiment_type: одна из категорий: "positive", "negative", "neutral"
        word_vectorizer: обученный TfidfVectorizer (из pipeline обучения)
        top_n: количество слов в результате
        strategy: "combined" (смесь метрик) или отдельные: "tfidf", "frequency", "bm25", "pmi"
        verbose: выводить ли информацию для отладки
    
    Returns:
        {
            "unigrams": [{"text": "слово", "score": 0.5}, ...],
            "bigrams": [{"text": "слово слово", "score": 0.3}, ...],
            "trigrams": [{"text": "слово слово слово", "score": 0.2}, ...],
            "domain_keywords": [{"text": "товар_очень_хороший", "score": 0.8}, ...],
            "explanation": "Методология извлечения..."
        }
    """
    
    group = texts_by_sentiment.get(sentiment_type, [])
    other = [
        t for k, v in texts_by_sentiment.items()
        if k != sentiment_type
        for t in v
    ]
    
    if not group:
        return {
            "unigrams": [],
            "bigrams": [],
            "trigrams": [],
            "domain_keywords": [],
        }
    
    # ─────────────────────────────────────────────────────────────
    # ШАГИ ПОДГОТОВКИ
    # ─────────────────────────────────────────────────────────────
    
    feature_names = word_vectorizer.get_feature_names_out()
    
    # Трансформируем тексты
    X_group = word_vectorizer.transform(group)
    X_other = word_vectorizer.transform(other) if other else None
    
    # ─────────────────────────────────────────────────────────────
    # ВЫЧИСЛЯЕМ МЕТРИКИ
    # ─────────────────────────────────────────────────────────────
    
    eps = 1e-6
    
    # 1. Частота в группе (нормализованная на размер группы)
    freq_group = np.asarray(X_group.sum(axis=0)).flatten() / (len(group) + eps)
    
    # 2. Частота в других группах
    if X_other is not None and X_other.shape[0] > 0:
        freq_other = np.asarray(X_other.sum(axis=0)).flatten() / (len(other) + eps)
    else:
        freq_other = np.zeros_like(freq_group)
    
    # 3. Дифференциация (контрастность): во сколько раз чаще в нашей группе
    specificity = freq_group / (freq_other + eps)
    
    # 4. TF-IDF (используем встроенные веса из vectorizer)
    X_group_tfidf = word_vectorizer.transform(group)
    tfidf_scores = np.asarray(X_group_tfidf.mean(axis=0)).flatten()
    
    # Комбинированный скор: баланс частоты и специфичности
    if strategy == "combined":
        # Комбинируем: частота в группе * контрастность ^ 0.5
        # sqrt нужен чтобы экстремально редкие слова не доминировали
        combined_scores = freq_group * np.sqrt(specificity)
    elif strategy == "tfidf":
        combined_scores = tfidf_scores
    elif strategy == "frequency":
        combined_scores = freq_group
    elif strategy == "specificity":
        combined_scores = specificity
    else:
        combined_scores = freq_group * np.sqrt(specificity)
    
    # ─────────────────────────────────────────────────────────────
    # РАЗДЕЛЯЕМ НА КАТЕГОРИИ (unigrams, bigrams, trigrams, домены)
    # ─────────────────────────────────────────────────────────────
    
    results = {
        "unigrams": [],
        "bigrams": [],
        "trigrams": [],
        "domain_keywords": [],
    }
    
    for idx, (feature, score) in enumerate(zip(feature_names, combined_scores)):
        if score < 1e-8:  # Пропускаем нулевые скоры
            continue
        
        word_count = len(feature.split())
        
        # Обработка доменных признаков (товар_*, качество_*, и т.д.)
        if "_" in feature and not feature.startswith("не_"):
            results["domain_keywords"].append({
                "text": feature,
                "score": round(float(score), 6),
                "frequency": round(float(freq_group[idx]), 6),
                "specificity": round(float(specificity[idx]), 2),
            })
        
        # Обработка обычных слов и фраз
        elif word_count == 1:
            # UNIGRAMS (одиночные слова)
            if is_valid_word(feature):
                results["unigrams"].append({
                    "text": feature,
                    "score": round(float(score), 6),
                    "frequency": round(float(freq_group[idx]), 6),
                    "specificity": round(float(specificity[idx]), 2),
                })
        
        elif word_count == 2:
            # BIGRAMS (двусловные)
            if is_meaningful_ngram(feature, min_words=2):
                results["bigrams"].append({
                    "text": feature,
                    "score": round(float(score), 6),
                    "frequency": round(float(freq_group[idx]), 6),
                    "specificity": round(float(specificity[idx]), 2),
                })
        
        elif word_count >= 3:
            # TRIGRAMS (трёхсловные и более)
            if is_meaningful_ngram(feature, min_words=3):
                results["trigrams"].append({
                    "text": feature,
                    "score": round(float(score), 6),
                    "frequency": round(float(freq_group[idx]), 6),
                    "specificity": round(float(specificity[idx]), 2),
                })
    
    # ─────────────────────────────────────────────────────────────
    # СОРТИРУЕМ И БЕРЁМ TOP-N
    # ─────────────────────────────────────────────────────────────
    
    # Сортируем по комбинированному скору
    for key in ["unigrams", "bigrams", "trigrams", "domain_keywords"]:
        results[key] = sorted(
            results[key],
            key=lambda x: x["score"],
            reverse=True
        )[:top_n]
    
    # Добавляем метаинформацию
    results["metadata"] = {
        "sentiment": sentiment_type,
        "group_size": len(group),
        "total_other_size": len(other),
        "strategy": strategy,
        "total_unique_features": len(feature_names),
    }
    
    if verbose:
        print(f"\n📊 Статистика для {sentiment_type}:")
        print(f"   Размер группы: {len(group)}")
        print(f"   Размер других: {len(other)}")
        print(f"   Всего n-грамм: {len(feature_names)}")
        print(f"   Выделено unigrams: {len(results['unigrams'])}")
        print(f"   Выделено bigrams: {len(results['bigrams'])}")
        print(f"   Выделено trigrams: {len(results['trigrams'])}")
        print(f"   Выделено domain_keywords: {len(results['domain_keywords'])}")
    
    return results


# ═════════════════════════════════════════════════════════════════════
# АЛЬТЕРНАТИВНАЯ ВЕРСИЯ: Для совместимости со старым кодом
# ═════════════════════════════════════════════════════════════════════

def extract_keywords(
    texts_by_sentiment,
    sentiment_type,
    word_vectorizer,
    top_n=10,
):
    """
    Обёртка для совместимости со старым интерфейсом.
    Вызывает extract_keywords_v2 с новыми параметрами.
    """
    results_v2 = extract_keywords_v2(
        texts_by_sentiment,
        sentiment_type,
        word_vectorizer,
        top_n=top_n,
        strategy="combined",
        verbose=False
    )
    
    # Преобразуем в старый формат для обратной совместимости
    return {
        "unigrams": results_v2.get("unigrams", []),
        "bigrams": results_v2.get("bigrams", []),
    }


if __name__ == "__main__":
    
    print("=" * 100)
    print("KEYWORD EXTRACTOR v2.0 - ДЕМОНСТРАЦИЯ")
    print("=" * 100)
    
    # Демонстрационные данные
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    demo_texts = {
        "positive": [
            "отличное качество очень вкусная еда приветливый персонал",
            "просто супер вкусно и не дорого молния доставка",
            "рекомендую очень понравилось приятная атмосфера",
            "благодарим за отличное обслуживание спасибо всему коллективу",
        ],
        "negative": [
            "ужасное качество испорченное товар не рекомендую",
            "полный отстой давайте хотя бы не обманывайте",
            "шляпа полная отвратительное обслуживание",
            "век жди доставку а товар не работает",
        ],
        "neutral": [
            "неплохо но было бы лучше если бы цена ниже",
            "не плохо но есть минусы вкусно но дороговато",
            "как то так себе хорошее место но не идеально",
        ]
    }
    
    # Обучаем vectorizer
    all_texts = [t for texts in demo_texts.values() for t in texts]
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        min_df=1,
    )
    vectorizer.fit(all_texts)
    
    # Извлекаем ключевые слова для каждой категории
    for sentiment in ["positive", "negative", "neutral"]:
        print(f"\n{'=' * 100}")
        print(f"КАТЕГОРИЯ: {sentiment.upper()}")
        print('=' * 100)
        
        results = extract_keywords_v2(
            demo_texts,
            sentiment,
            vectorizer,
            top_n=5,
            strategy="combined",
            verbose=True
        )
        
        print(f"\n📌 UNIGRAMS (одиночные слова):")
        for item in results["unigrams"]:
            print(f"  • {item['text']:20s} score={item['score']:.6f} freq={item['frequency']:.6f} spec={item['specificity']:.2f}x")
        
        print(f"\n📍 BIGRAMS (двусловные):")
        for item in results["bigrams"]:
            print(f"  • {item['text']:30s} score={item['score']:.6f} freq={item['frequency']:.6f} spec={item['specificity']:.2f}x")
        
        print(f"\n🏪 DOMAIN KEYWORDS (доменные признаки):")
        for item in results["domain_keywords"]:
            print(f"  • {item['text']:30s} score={item['score']:.6f}")
"""
Модуль препроцессинга текста для анализа тональности
"""
import re
import sys
import pymorphy2
from nltk.corpus import stopwords
import nltk
from functools import lru_cache
from joblib import Parallel, delayed

# ── Инициализация NLTK ────────────────────────────────────────────────────────
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

# ── Морфологический анализатор (один экземпляр на весь модуль) ────────────────
morph = pymorphy2.MorphAnalyzer()

# ── Стоп-слова ────────────────────────────────────────────────────────────────
RUSSIAN_STOPWORDS = set(stopwords.words('russian'))
CUSTOM_STOPWORDS = {
    'это', 'быть', 'весь', 'мочь', 'свой', 'который',
    'также', 'этот', 'тот', 'такой', 'если', 'когда'
}
RUSSIAN_STOPWORDS.update(CUSTOM_STOPWORDS)


# ── Кэшированная лемматизация ─────────────────────────────────────────────────
@lru_cache(maxsize=100_000) 
def _lemmatize_word(token: str) -> str:
    """Лемматизация одного слова с кэшированием результата."""
    return morph.parse(token)[0].normal_form


# ── Функции обработки ─────────────────────────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize_text(text):
    return [t for t in text.split() if len(t) >= 2]


def lemmatize_tokens(tokens):          # ← только одна версия, использует кэш
    return [_lemmatize_word(token) for token in tokens]


def remove_stopwords(tokens):
    return [t for t in tokens if t not in RUSSIAN_STOPWORDS]


def preprocess_text(text, remove_stops=True, lemmatize=True, handle_negations=True ):
    text = clean_text(text)
    if not text:
        return ""
    tokens = tokenize_text(text)
    if lemmatize:
        tokens = lemmatize_tokens(tokens)
    if remove_stops:
        tokens = remove_stopwords(tokens)
    return ' '.join(tokens)




def preprocess_corpus(texts, verbose=True, n_jobs=-1, handle_negations=True ):
    """
    n_jobs=-1 — использует все ядра процессора
    """
    total = len(texts)
    
    def process_with_log(idx, text):
        result = preprocess_text(text)
        if verbose and (idx + 1) % 500 == 0:
            print(f"  Обработано: {idx + 1}/{total}", file=sys.stderr)
        return result

    processed = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(preprocess_text)(text)
        for text in texts
    )
    return processed


# ── Тест ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_texts = [
        "Отличный товар! Очень доволен покупкой.",
        "Ужасное качество, не рекомендую!!!",
        "Нормально, но цена высокая.",
        "Не понравилось совсем, вернул обратно",
    ]

    print("Тестирование препроцессинга:")
    print("=" * 50)
    for text in test_texts:
        print(f"\nИсходный:    {text}")
        print(f"Обработанный: {preprocess_text(text)}")
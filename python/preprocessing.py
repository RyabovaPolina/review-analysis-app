"""
Модуль препроцессинга текста для анализа тональности
Оптимизированная версия для проведения экспериментов
"""

import re
import sys
import pymorphy2
from nltk.corpus import stopwords
import nltk
from functools import lru_cache
from joblib import Parallel, delayed


# ─────────────────────────────────────────────────────────────
# Инициализация NLTK
# ─────────────────────────────────────────────────────────────

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
    nltk.download("punkt")


# ─────────────────────────────────────────────────────────────
# Морфологический анализатор
# ─────────────────────────────────────────────────────────────

morph = pymorphy2.MorphAnalyzer()


# ─────────────────────────────────────────────────────────────
# Отрицания (НЕ удаляются)
# ─────────────────────────────────────────────────────────────

NEGATIONS = {
    "не",
    "нет",
    "ни",
    "никогда",
    "ничего",
    "никакой",
    "нигде",
    "никуда"
}


# ─────────────────────────────────────────────────────────────
# Пользовательские стоп-слова
# ─────────────────────────────────────────────────────────────

CUSTOM_STOPWORDS = {
    "это",
    "быть",
    "весь",
    "мочь",
    "свой",
    "который",
    "также",
    "этот",
    "тот",
    "такой",
    "если",
    "когда"
}


# ─────────────────────────────────────────────────────────────
# Итоговый список стоп-слов
# ─────────────────────────────────────────────────────────────

RUSSIAN_STOPWORDS = (
    set(stopwords.words("russian"))
    | CUSTOM_STOPWORDS
) - NEGATIONS


# ─────────────────────────────────────────────────────────────
# Кэшированная лемматизация
# ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=100_000)
def _lemmatize_word(token: str) -> str:
    """
    Лемматизация одного слова с кэшированием.
    """
    return morph.parse(token)[0].normal_form


def lemmatize_tokens(tokens):
    """
    Лемматизация списка токенов.
    """
    return [_lemmatize_word(token) for token in tokens]


# ─────────────────────────────────────────────────────────────
# Обработка отрицаний
# ─────────────────────────────────────────────────────────────

def combine_negations(tokens):
    """
    Объединяет конструкцию:

    не хороший → не_хороший
    """

    result = []

    i = 0

    while i < len(tokens):

        if tokens[i] == "не" and i + 1 < len(tokens):

            combined = f"не_{tokens[i + 1]}"
            result.append(combined)

            i += 2

        else:

            result.append(tokens[i])
            i += 1

    return result


# ─────────────────────────────────────────────────────────────
# Очистка текста
# ─────────────────────────────────────────────────────────────

def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\S+@\S+", "", text)

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ─────────────────────────────────────────────────────────────
# Токенизация
# ─────────────────────────────────────────────────────────────

def tokenize_text(text):

    return [
        token
        for token in text.split()
        if len(token) >= 2
    ]


# ─────────────────────────────────────────────────────────────
# Удаление стоп-слов
# ─────────────────────────────────────────────────────────────

def remove_stopwords(tokens):

    return [
        token
        for token in tokens
        if token not in RUSSIAN_STOPWORDS
    ]


# ─────────────────────────────────────────────────────────────
# Основная функция preprocessing
# ─────────────────────────────────────────────────────────────

def preprocess_text(
    text,
    remove_stops=True,
    lemmatize=True,
    use_negations=True
):

    text = clean_text(text)

    if not text:
        return ""

    tokens = tokenize_text(text)

    if use_negations:
        tokens = combine_negations(tokens)

    if lemmatize:
        tokens = lemmatize_tokens(tokens)

    if remove_stops:
        tokens = remove_stopwords(tokens)

    return " ".join(tokens)


# ─────────────────────────────────────────────────────────────
# Обработка корпуса текстов
# ─────────────────────────────────────────────────────────────

def preprocess_corpus(
    texts,
    verbose=True,
    n_jobs=-1,
    use_negations=True
):
    """
    Параллельная обработка списка текстов.
    """

    total = len(texts)

    def process_with_log(idx, text):

        result = preprocess_text(
            text,
            use_negations=use_negations
        )

        if verbose and (idx + 1) % 500 == 0:
            print(
                f"Обработано: {idx + 1}/{total}",
                file=sys.stderr
            )

        return result

    processed = Parallel(
        n_jobs=n_jobs,
        backend="threading"
    )(
        delayed(process_with_log)(idx, text)
        for idx, text in enumerate(texts)
    )

    return processed


# ─────────────────────────────────────────────────────────────
# Тестирование
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    test_texts = [

        "Отличный товар! Очень доволен покупкой.",

        "Ужасное качество, не рекомендую!!!",

        "Нормально, но цена высокая.",

        "Не понравилось совсем, вернул обратно",

        "Мне не очень понравился магазин",

    ]

    print("Тестирование препроцессинга")
    print("=" * 50)

    for text in test_texts:

        processed = preprocess_text(text)

        print()
        print("Исходный:")
        print(text)

        print("Обработанный:")
        print(processed)
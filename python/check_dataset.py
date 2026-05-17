import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer
)
import numpy as np
import re

# =========================
# НАСТРОЙКИ
# =========================

FILE_PATH = "C:/projects/git/review-analysis-app/python/data/restaurants_full_dataset.csv"

TEXT_COLUMN = "text"
RATING_COLUMN = "rating"

RAW_EXAMPLES_COUNT = 10

# =========================
# ЗАГРУЗКА ДАННЫХ
# =========================

df = pd.read_csv(FILE_PATH)

print("=" * 90)
print("ИНФОРМАЦИЯ О ДАТАСЕТЕ")
print("=" * 90)

print(f"\nРазмер датасета: {df.shape}")

print("\nНазвания колонок:")
print(df.columns.tolist())

print("\nПервые 5 строк:")
print(df.head())

# =========================
# ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ
# =========================

print("\n" + "=" * 90)
print("ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ")
print("=" * 90)

missing_values = df.isnull().sum()

print("\nКоличество пропусков:")
print(missing_values)

missing_percent = round(
    (missing_values / len(df)) * 100,
    2
)

print("\nПроцент пропусков:")
print(missing_percent)

# =========================
# СОЗДАНИЕ КЛАССОВ
# =========================

def rating_to_sentiment(rating):

    try:
        rating = float(rating)

        if rating >= 4:
            return "positive"

        elif rating == 3:
            return "neutral"

        else:
            return "negative"

    except:
        return "unknown"

df["sentiment"] = df[
    RATING_COLUMN
].apply(rating_to_sentiment)

LABEL_COLUMN = "sentiment"

# =========================
# РАСПРЕДЕЛЕНИЕ КЛАССОВ
# =========================

print("\n" + "=" * 90)
print("СТАТИСТИКА КЛАССОВ")
print("=" * 90)

class_counts = df[
    LABEL_COLUMN
].value_counts()

print("\nКоличество отзывов по классам:")
print(class_counts)

class_percentages = round(
    df[LABEL_COLUMN]
    .value_counts(normalize=True) * 100,
    2
)

print("\nПроцентное распределение:")
print(class_percentages)

max_class = class_counts.max()
min_class = class_counts.min()

imbalance_ratio = round(
    max_class / min_class,
    2
)

print(f"\nКоэффициент дисбаланса: "
      f"{imbalance_ratio}")

dominant_ratio = round(
    (max_class / len(df)) * 100,
    2
)

print(f"Доля доминирующего класса: "
      f"{dominant_ratio}%")

# =========================
# ВИЗУАЛИЗАЦИЯ КЛАССОВ
# =========================

plt.figure(figsize=(8, 5))

class_counts.plot(kind="bar")

plt.title("Распределение классов")
plt.xlabel("Класс")
plt.ylabel("Количество")

plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

# =========================
# ОБРАБОТКА ТЕКСТОВ
# =========================

texts = df[
    TEXT_COLUMN
].dropna().astype(str)

# =========================
# АНАЛИЗ ШУМА
# =========================

print("\n" + "=" * 90)
print("АНАЛИЗ ШУМА")
print("=" * 90)

noise_examples = []

url_count = 0
emoji_count = 0
caps_count = 0
short_reviews = 0

for text in texts:

    if "http" in text.lower():
        url_count += 1

    if re.search(r"[😀-🙏]", text):
        emoji_count += 1

    if re.search(r"\b[A-ZА-Я]{3,}\b", text):
        caps_count += 1

    if len(text.split()) < 3:
        short_reviews += 1

    has_noise = (
        "<" in text or
        ">" in text or
        "http" in text.lower() or
        "#" in text or
        "@" in text or
        "!!!" in text or
        len(text.split()) < 3
    )

    if has_noise:
        noise_examples.append(text)

noise_percent = round(
    len(noise_examples) / len(texts) * 100,
    2
)

print(f"\nШумных отзывов: "
      f"{len(noise_examples)}")

print(f"Процент шумных отзывов: "
      f"{noise_percent}%")

print(f"\nОтзывы с URL: {url_count}")
print(f"Отзывы с emoji: {emoji_count}")
print(f"Отзывы с CAPS WORDS: {caps_count}")
print(f"Очень короткие отзывы: "
      f"{short_reviews}")

print("\nПРИМЕРЫ ШУМНЫХ ОТЗЫВОВ:")

for i, example in enumerate(
        noise_examples[:RAW_EXAMPLES_COUNT], 1):

    print(f"\nПример {i}:")
    print(example)
    print("-" * 50)

# =========================
# АНАЛИЗ LABEL NOISE
# =========================

print("\n" + "=" * 90)
print("LABEL NOISE ANALYSIS")
print("=" * 90)

negative_words = [
    "bad",
    "terrible",
    "awful",
    "worst",
    "disgusting",
    "horrible",
    "poor"
]

positive_words = [
    "great",
    "excellent",
    "amazing",
    "perfect",
    "love",
    "wonderful",
    "best"
]

suspicious_samples = []

for _, row in df.iterrows():

    text = str(
        row[TEXT_COLUMN]
    ).lower()

    rating = row[RATING_COLUMN]

    # высокий рейтинг + негативные слова
    if rating >= 4:

        if any(
            word in text
            for word in negative_words
        ):

            suspicious_samples.append(
                (rating, text[:200])
            )

    # низкий рейтинг + позитивные слова
    elif rating <= 2:

        if any(
            word in text
            for word in positive_words
        ):

            suspicious_samples.append(
                (rating, text[:200])
            )

print(f"\nПодозрительных примеров: "
      f"{len(suspicious_samples)}")

print("\nПРИМЕРЫ LABEL NOISE:")

for i, sample in enumerate(
        suspicious_samples[:10], 1):

    print(f"\nПример {i}")
    print(f"Rating: {sample[0]}")
    print(sample[1])
    print("-" * 50)

# =========================
# ДЛИНА ТЕКСТОВ
# =========================

print("\n" + "=" * 90)
print("СТАТИСТИКА ДЛИНЫ ТЕКСТОВ")
print("=" * 90)

df["text_length_chars"] = df[
    TEXT_COLUMN
].astype(str).apply(len)

df["text_length_words"] = df[
    TEXT_COLUMN
].astype(str).apply(
    lambda x: len(x.split())
)

print(f"\nСредняя длина (символы): "
      f"{round(df['text_length_chars'].mean(), 2)}")

print(f"Медианная длина (символы): "
      f"{df['text_length_chars'].median()}")

print(f"Минимальная длина: "
      f"{df['text_length_chars'].min()}")

print(f"Максимальная длина: "
      f"{df['text_length_chars'].max()}")

print(f"Стандартное отклонение: "
      f"{round(df['text_length_chars'].std(), 2)}")

print(f"\nСреднее количество слов: "
      f"{round(df['text_length_words'].mean(), 2)}")

print("\nКвантили длины:")

print(
    df["text_length_chars"].quantile(
        [0.25, 0.5, 0.75, 0.95]
    )
)

# =========================
# ВЫБРОСЫ
# =========================

print("\n" + "=" * 90)
print("АНАЛИЗ ВЫБРОСОВ")
print("=" * 90)

very_long_reviews = df[
    df["text_length_chars"] > 1000
]

very_short_reviews = df[
    df["text_length_words"] <= 2
]

print(f"\nОчень длинных отзывов: "
      f"{len(very_long_reviews)}")

print(f"Очень коротких отзывов: "
      f"{len(very_short_reviews)}")

# =========================
# ВИЗУАЛИЗАЦИЯ ДЛИНЫ
# =========================

plt.figure(figsize=(8, 5))

plt.hist(
    df["text_length_chars"],
    bins=30
)

plt.title(
    "Распределение длины отзывов"
)

plt.xlabel("Количество символов")
plt.ylabel("Количество отзывов")

plt.tight_layout()
plt.show()

# =========================
# ДУБЛИКАТЫ
# =========================

print("\n" + "=" * 90)
print("АНАЛИЗ ДУБЛИКАТОВ")
print("=" * 90)

duplicates = df.duplicated(
    subset=[TEXT_COLUMN]
).sum()

duplicates_percent = round(
    duplicates / len(df) * 100,
    2
)

print(f"\nКоличество дубликатов: "
      f"{duplicates}")

print(f"Процент дубликатов: "
      f"{duplicates_percent}%")

# =========================
# ЛЕКСИЧЕСКИЙ АНАЛИЗ
# =========================

print("\n" + "=" * 90)
print("ЛЕКСИЧЕСКИЙ АНАЛИЗ")
print("=" * 90)

all_words = " ".join(
    texts
).lower().split()

word_freq = Counter(all_words)

total_words = len(all_words)

unique_words = len(set(all_words))

print(f"\nОбщее количество слов: "
      f"{total_words}")

print(f"Количество уникальных слов: "
      f"{unique_words}")

# Type Token Ratio

ttr = round(
    unique_words / total_words,
    4
)

print(f"\nType-Token Ratio: {ttr}")

# Hapax Legomena

hapax = len([
    word for word, count
    in word_freq.items()
    if count == 1
])

print(f"Слов, встречающихся 1 раз: "
      f"{hapax}")

# OOV Rate

rare_words = [
    word for word, count
    in word_freq.items()
    if count == 1
]

oov_rate = round(
    len(rare_words) /
    unique_words * 100,
    2
)

print(f"OOV Rate: {oov_rate}%")

# Vocabulary coverage

top_1000 = sum([
    count for word, count
    in word_freq.most_common(1000)
])

coverage = round(
    top_1000 / total_words * 100,
    2
)

print(f"Coverage top-1000 words: "
      f"{coverage}%")

# =========================
# ТОП СЛОВ
# =========================

print("\n" + "=" * 90)
print("ТОП-20 СЛОВ")
print("=" * 90)

for word, count in word_freq.most_common(20):

    print(f"{word}: {count}")

# =========================
# ТОП СЛОВ ПО КЛАССАМ
# =========================

for label in df[LABEL_COLUMN].unique():

    print("\n" + "=" * 90)
    print(f"TOP WORDS FOR "
          f"{label.upper()}")
    print("=" * 90)

    class_text = " ".join(
        df[
            df[LABEL_COLUMN] == label
        ][TEXT_COLUMN]
        .astype(str)
    ).lower().split()

    class_freq = Counter(
        class_text
    )

    for word, count in class_freq.most_common(15):

        print(f"{word}: {count}")

# =========================
# УНИКАЛЬНЫЕ СЛОВА ПО КЛАССАМ
# =========================

print("\n" + "=" * 90)
print("УНИКАЛЬНЫЕ СЛОВА ПО КЛАССАМ")
print("=" * 90)

for label in df[LABEL_COLUMN].unique():

    class_words = " ".join(
        df[
            df[LABEL_COLUMN] == label
        ][TEXT_COLUMN]
        .astype(str)
    ).lower().split()

    unique_class_words = len(
        set(class_words)
    )

    print(f"{label}: "
          f"{unique_class_words}")

# =========================
# TF-IDF SPARSITY
# =========================

print("\n" + "=" * 90)
print("TF-IDF АНАЛИЗ")
print("=" * 90)

vectorizer = TfidfVectorizer(
    max_features=5000
)

X = vectorizer.fit_transform(
    texts
)

sparsity = 1.0 - (
    X.count_nonzero() /
    (X.shape[0] * X.shape[1])
)

print(f"\nРазмер TF-IDF матрицы: "
      f"{X.shape}")

print(f"Разреженность матрицы: "
      f"{round(sparsity * 100, 2)}%")

# =========================
# ТОП TF-IDF ТЕРМИНОВ
# =========================

feature_names = np.array(
    vectorizer.get_feature_names_out()
)

tfidf_scores = np.asarray(
    X.mean(axis=0)
).ravel()

top_tfidf_idx = tfidf_scores.argsort()[
                 -20:
                 ][::-1]

print("\nТОП TF-IDF ТЕРМИНОВ:")

for idx in top_tfidf_idx:

    print(
        f"{feature_names[idx]}: "
        f"{round(tfidf_scores[idx], 4)}"
    )

# =========================
# TF-IDF ПО КЛАССАМ
# =========================

print("\n" + "=" * 90)
print("TF-IDF ПО КЛАССАМ")
print("=" * 90)

for label in df[LABEL_COLUMN].unique():

    print("\n" + "=" * 50)
    print(f"{label.upper()}")
    print("=" * 50)

    class_texts = df[
        df[LABEL_COLUMN] == label
    ][TEXT_COLUMN].astype(str)

    class_vectorizer = TfidfVectorizer(
        max_features=1000
    )

    class_X = class_vectorizer.fit_transform(
        class_texts
    )

    class_scores = np.asarray(
        class_X.mean(axis=0)
    ).ravel()

    class_features = np.array(
        class_vectorizer
        .get_feature_names_out()
    )

    top_idx = class_scores.argsort()[
              -10:
              ][::-1]

    for idx in top_idx:

        print(
            f"{class_features[idx]}: "
            f"{round(class_scores[idx], 4)}"
        )

# =========================
# BIGRAMS
# =========================

print("\n" + "=" * 90)
print("TOP BIGRAMS")
print("=" * 90)

bigram_vectorizer = CountVectorizer(
    ngram_range=(2, 2),
    stop_words="english",
    max_features=20
)

bigram_matrix = (
    bigram_vectorizer
    .fit_transform(texts)
)

bigram_counts = np.asarray(
    bigram_matrix.sum(axis=0)
).ravel()

bigrams = (
    bigram_vectorizer
    .get_feature_names_out()
)

bigram_freq = list(
    zip(bigrams, bigram_counts)
)

bigram_freq = sorted(
    bigram_freq,
    key=lambda x: x[1],
    reverse=True
)

for bigram, count in bigram_freq:

    print(f"{bigram}: {count}")

# =========================
# TRIGRAMS
# =========================

print("\n" + "=" * 90)
print("TOP TRIGRAMS")
print("=" * 90)

trigram_vectorizer = CountVectorizer(
    ngram_range=(3, 3),
    stop_words="english",
    max_features=20
)

trigram_matrix = (
    trigram_vectorizer
    .fit_transform(texts)
)

trigram_counts = np.asarray(
    trigram_matrix.sum(axis=0)
).ravel()

trigrams = (
    trigram_vectorizer
    .get_feature_names_out()
)

trigram_freq = list(
    zip(trigrams, trigram_counts)
)

trigram_freq = sorted(
    trigram_freq,
    key=lambda x: x[1],
    reverse=True
)

for trigram, count in trigram_freq:

    print(f"{trigram}: {count}")

# =========================
# СТАТИСТИКА РЕЙТИНГОВ
# =========================

print("\n" + "=" * 90)
print("СТАТИСТИКА РЕЙТИНГОВ")
print("=" * 90)

print(f"\nСредний рейтинг: "
      f"{round(df[RATING_COLUMN].mean(), 2)}")

print(f"Медианный рейтинг: "
      f"{df[RATING_COLUMN].median()}")

print(f"Минимальный рейтинг: "
      f"{df[RATING_COLUMN].min()}")

print(f"Максимальный рейтинг: "
      f"{df[RATING_COLUMN].max()}")

# =========================
# ДЛИНА ПО КЛАССАМ
# =========================

print("\n" + "=" * 90)
print("ДЛИНА ТЕКСТОВ ПО КЛАССАМ")
print("=" * 90)

length_stats = df.groupby(
    LABEL_COLUMN
)["text_length_words"].agg([
    "mean",
    "median",
    "min",
    "max"
])

print(length_stats)

# =========================
# КОРРЕЛЯЦИЯ
# =========================

print("\n" + "=" * 90)
print("КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
print("=" * 90)

correlation = df[
    "text_length_words"
].corr(
    df[RATING_COLUMN]
)

print(f"\nКорреляция длины "
      f"и рейтинга: "
      f"{round(correlation, 4)}")

# =========================
# СОХРАНЕНИЕ
# =========================

OUTPUT_FILE = "processed_reviews.csv"

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 90)
print(f"Файл сохранён: "
      f"{OUTPUT_FILE}")
print("=" * 90)
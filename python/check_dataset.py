import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# =========================
# НАСТРОЙКИ
# =========================

# Путь к CSV-файлу
FILE_PATH = "C:/projects/git/review-analysis-app/python/data/restaurants_full_dataset.csv"

# Названия колонок
TEXT_COLUMN = "text"      # колонка с текстами отзывов
RATING_COLUMN = "rating"  # колонка с рейтингом

# Сколько примеров сырых отзывов вывести
RAW_EXAMPLES_COUNT = 10


# =========================
# ЗАГРУЗКА ДАННЫХ
# =========================

df = pd.read_csv(FILE_PATH)

print("=" * 60)
print("ИНФОРМАЦИЯ О ДАТАСЕТЕ")
print("=" * 60)

print(f"\nРазмер датасета: {df.shape}")

print("\nНазвания колонок:")
print(df.columns.tolist())

print("\nПервые 5 строк:")
print(df.head())


# =========================
# СОЗДАНИЕ КЛАССОВ ТОНАЛЬНОСТИ
# =========================

def rating_to_sentiment(rating):
    """
    Преобразование рейтинга в класс тональности
    """

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


# Создаем колонку sentiment
df["sentiment"] = df[RATING_COLUMN].apply(rating_to_sentiment)

LABEL_COLUMN = "sentiment"


# =========================
# СТАТИСТИКА КЛАССОВ
# =========================

print("\n" + "=" * 60)
print("СТАТИСТИКА КЛАССОВ")
print("=" * 60)

# Количество объектов по классам
class_counts = df[LABEL_COLUMN].value_counts()

print("\nКоличество отзывов по классам:")
print(class_counts)

# Процентное распределение
class_percentages = round(
    df[LABEL_COLUMN].value_counts(normalize=True) * 100,
    2
)

print("\nПроцентное распределение:")
print(class_percentages)

# Проверка дисбаланса классов
max_class = class_counts.max()
min_class = class_counts.min()

imbalance_ratio = round(max_class / min_class, 2)

print(f"\nКоэффициент дисбаланса: {imbalance_ratio}")

if imbalance_ratio > 1.5:
    print("ВНИМАНИЕ: классы несбалансированы.")
else:
    print("Классы распределены относительно равномерно.")


# =========================
# ВИЗУАЛИЗАЦИЯ
# =========================

plt.figure(figsize=(8, 5))

class_counts.plot(kind="bar")

plt.title("Распределение классов отзывов")
plt.xlabel("Класс")
plt.ylabel("Количество отзывов")

plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# =========================
# ПЕРВИЧНЫЙ АНАЛИЗ ТЕКСТОВ
# =========================

print("\n" + "=" * 60)
print("ПРИМЕРЫ СЫРЫХ ОТЗЫВОВ")
print("=" * 60)

# Удаляем пустые значения
texts = df[TEXT_COLUMN].dropna().astype(str)

noise_examples = []

for text in texts:

    has_noise = (
        "<" in text or ">" in text or
        "http" in text.lower() or
        "#" in text or
        "@" in text or
        "!!!" in text or
        len(text.split()) < 3
    )

    if has_noise:
        noise_examples.append(text)

print(f"\nНайдено потенциально шумных отзывов: {len(noise_examples)}")

# Если шумных мало — вывести обычные примеры
if len(noise_examples) == 0:

    print("\nЯвный шум не найден.")
    print("Примеры сырых отзывов:\n")

    for i, example in enumerate(texts.head(RAW_EXAMPLES_COUNT), 1):
        print(f"Пример {i}:")
        print(example)
        print("-" * 50)

else:

    for i, example in enumerate(noise_examples[:RAW_EXAMPLES_COUNT], 1):
        print(f"\nПример {i}:")
        print(example)
        print("-" * 50)


# =========================
# ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ
# =========================

print("\n" + "=" * 60)
print("ДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА")
print("=" * 60)

# Длина отзывов
df["text_length"] = df[TEXT_COLUMN].astype(str).apply(len)

print(f"\nСредняя длина отзыва: {round(df['text_length'].mean(), 2)}")

print(f"Минимальная длина: {df['text_length'].min()}")

print(f"Максимальная длина: {df['text_length'].max()}")


# =========================
# ЧАСТОТНЫЙ АНАЛИЗ СЛОВ
# =========================

all_words = " ".join(
    df[TEXT_COLUMN].dropna().astype(str)
).lower().split()

word_freq = Counter(all_words)

print("\nТоп-20 самых частых слов:\n")

for word, count in word_freq.most_common(20):
    print(f"{word}: {count}")


# =========================
# СОХРАНЕНИЕ ОБНОВЛЕННОГО CSV
# =========================

OUTPUT_FILE = "processed_reviews.csv"

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("\n" + "=" * 60)
print(f"Обработанный файл сохранён: {OUTPUT_FILE}")
print("=" * 60)
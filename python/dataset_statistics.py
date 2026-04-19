"""
Скрипт для вывода статистики датасета отзывов
"""

import pandas as pd

INPUT_FILE = "data/geo-reviews-dataset-2023.csv"


def rating_to_sentiment(rating):

    if rating <= 2:
        return "negative"

    elif rating <= 4:
        return "neutral"

    else:
        return "positive"


def main():

    print("=" * 70)
    print("ОПИСАНИЕ НАБОРА ДАННЫХ")
    print("=" * 70)

    print("\nЗагрузка файла...")

    df = pd.read_csv(
        INPUT_FILE,
        sep=",",
        quoting=1,
        encoding="utf-8"
    )

    # удалить пустые тексты

    df = df[df["text"].notna()]

    print("\nИсточник данных:")
    print(INPUT_FILE)

    print("\nРазмер датасета:")

    print("Количество записей:", len(df))
    print("Количество столбцов:", len(df.columns))

    print("\nСтруктура данных:")

    print("\nСписок столбцов:")

    for column in df.columns:
        print("-", column)

    print("\nПример записи:")

    print(df.iloc[0])

    # -------------------------
    # ПРАВИЛЬНАЯ ОБРАБОТКА RATING
    # -------------------------

    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )

    df["rating"] = df["rating"].astype(int)

    # -------------------------
    # РАСПРЕДЕЛЕНИЕ РЕЙТИНГОВ
    # -------------------------

    print("\n" + "=" * 70)
    print("РАСПРЕДЕЛЕНИЕ РЕЙТИНГОВ")
    print("=" * 70)

    rating_counts = (
        df["rating"]
        .value_counts()
        .sort_index()
    )

    rating_table = pd.DataFrame({

        "Rating": rating_counts.index,

        "Количество": rating_counts.values

    })

    print("\nТаблица:")

    print(
        rating_table
        .to_string(index=False)
    )

    # -------------------------
    # SENTIMENT
    # -------------------------

    df["sentiment"] = df["rating"].apply(
        rating_to_sentiment
    )

    print("\n" + "=" * 70)
    print("РАСПРЕДЕЛЕНИЕ КЛАССОВ SENTIMENT")
    print("=" * 70)

    sentiment_counts = (
        df["sentiment"]
        .value_counts()
    )

    sentiment_table = pd.DataFrame({

        "Класс": sentiment_counts.index,

        "Количество": sentiment_counts.values

    })

    print("\nТаблица:")

    print(
        sentiment_table
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("Готово")
    print("=" * 70)


if __name__ == "__main__":
    main()
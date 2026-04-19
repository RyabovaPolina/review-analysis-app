import pandas as pd

INPUT_FILE = "data/geo-reviews-dataset-2023.csv"

OUTPUT_FILE = "data/training_dataset.csv"

SAMPLES_PER_CLASS = 5600


def rating_to_sentiment(rating):

    try:

        rating = float(str(rating).strip().replace(",", "."))

        if rating <= 2.5:
            return 'negative'
        elif rating >= 4.0:
            return 'positive'
        else:
            return 'neutral'
    except:
        return 'neutral'
    
print("Загрузка файла...")

df = pd.read_csv(
    INPUT_FILE,
    sep=",",
    quoting=1,
    encoding="utf-8"
)

print("Всего строк:", len(df))

# удалить пустые тексты

df = df[df["text"].notna()]

# фильтр коротких

df = df[df["text"].str.split().str.len() >= 5]

print("После фильтра:", len(df))

# диагностика

print("\nРаспределение рейтингов:")

print(
    df["rating"]
    .astype(str)
    .str.strip()
    .value_counts()
    .sort_index()
)

# создать sentiment

df["sentiment"] = df["rating"].apply(
    rating_to_sentiment
)

df = df[df["sentiment"].notna()]

print("\nРаспределение sentiment:")

print(df["sentiment"].value_counts())

# балансировка

sampled = []

for label in [
    "negative",
    "neutral",
    "positive",
]:

    subset = df[df["sentiment"] == label]

    print(label, len(subset))

    sampled.append(
        subset.sample(
            n=min(
                SAMPLES_PER_CLASS,
                len(subset)
            ),
            random_state=42
        )
    )

final_df = pd.concat(sampled)

final_df = final_df.sample(
    frac=1,
    random_state=42
)

final_df = final_df.reset_index(drop=True)

final_df[["text", "sentiment"]].to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nГотово")

print(final_df["sentiment"].value_counts())
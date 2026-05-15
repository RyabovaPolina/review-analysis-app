import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import json

INPUT_FILE = "data/geo-reviews-dataset-2023.csv"
OUTPUT_FILE = "data/training_dataset.csv"
WEIGHTS_FILE = "data/class_weights.json"
FULL_OUTPUT_FILE = "data/restaurants_full_dataset.csv"

TARGET_RUBRICS = {"Кафе", "Ресторан"}


# =========================
# HELPERS
# =========================

def rating_to_sentiment(rating):
    try:
        rating = int(float(str(rating).strip().replace(",", ".")))

        if rating in [1, 2]:
            return "negative"
        elif rating in [3, 4]:
            return "neutral"
        elif rating == 5:
            return "positive"
        else:
            return None
    except Exception:
        return None


def has_target_rubric(rubrics_str):
    if pd.isna(rubrics_str):
        return False

    rubrics = [r.strip() for r in rubrics_str.split(";")]
    return any(r in TARGET_RUBRICS for r in rubrics)


# =========================
# LOAD
# =========================

print("Загрузка файла...")

df = pd.read_csv(
    INPUT_FILE,
    sep=",",
    quoting=1,
    encoding="utf-8"
)

print("Всего строк:", len(df))


# =========================
# FILTER
# =========================

df = df[df["rubrics"].apply(has_target_rubric)]
print("После фильтра по рубрикам:", len(df))

# =========================
# SAVE RAW (ТОЛЬКО КАФЕ/РЕСТОРАНЫ, БЕЗ ИЗМЕНЕНИЙ)
# =========================

df_raw = df.copy()

df_raw = df_raw.sample(frac=1, random_state=42).reset_index(drop=True)

df_raw.to_csv(
    FULL_OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nСырой датасет (только кафе/рестораны) сохранён:")
print(FULL_OUTPUT_FILE)

df = df[df["text"].notna()]
df = df[df["text"].str.split().str.len() >= 5]

print("После фильтра текста:", len(df))


# =========================
# SENTIMENT
# =========================

df["sentiment"] = df["rating"].apply(rating_to_sentiment)
df = df[df["sentiment"].notna()]

print("\nРаспределение sentiment (до):")
print(df["sentiment"].value_counts())


# =========================
# SMART SAMPLING
# =========================

print("\nУмное сокращение датасета...")

# длина текста = proxy информативности
df["text_len"] = df["text"].str.split().str.len()

df_neg = df[df["sentiment"] == "negative"]
df_neu = df[df["sentiment"] == "neutral"]
df_pos = df[df["sentiment"] == "positive"]

# цели (примерно в 2 раза меньше исходного)
TARGET_NEG = len(df_neg)        # оставляем всё
TARGET_NEU = 8000              # почти всё
TARGET_POS = 20000             # сильно режем

# сортировка по "значимости"
df_neu = df_neu.sort_values("text_len", ascending=False).head(TARGET_NEU)
df_pos = df_pos.sort_values("text_len", ascending=False).head(TARGET_POS)

df = pd.concat([df_neg, df_neu, df_pos])

print("\nПосле сокращения:")
print(df["sentiment"].value_counts())


# =========================
# CLASS WEIGHTS
# =========================

classes = np.array(["negative", "neutral", "positive"])

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=df["sentiment"]
)

class_weights = dict(zip(classes, weights))

print("\nВеса классов:")
for k, v in class_weights.items():
    print(f"{k}: {v:.4f}")


# =========================
# SAVE
# =========================

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df[["text", "sentiment"]].to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

with open(WEIGHTS_FILE, "w", encoding="utf-8") as f:
    json.dump(class_weights, f, ensure_ascii=False, indent=2)


print("\nГотово")
print("Датасет:", OUTPUT_FILE)
print("Веса:", WEIGHTS_FILE)
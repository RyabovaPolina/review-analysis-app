import sys
import re
import pandas as pd

def filter_low_quality_reviews(df: pd.DataFrame, text_column: str, min_length: int = 10) -> pd.DataFrame:
    original_count = len(df)

    # 1️⃣ Пустые и короткие
    df = df[df[text_column].str.len() > min_length]

    # 2️⃣ Дубликаты
    df = df.drop_duplicates(subset=[text_column], keep='first')

    # 3️⃣ Тексты без букв
    df = df[df[text_column].str.contains(r'[а-яa-z]', regex=True, case=False)]

    # 4️⃣ Повторяющиеся символы (спам)
    df = df[~df[text_column].str.contains(r'(.)\1{5,}', regex=True)]

    removed = original_count - len(df)
    print(f"Удалено {removed} низкокачественных отзывов ({removed/original_count*100:.1f}%)", file=sys.stderr)

    return df
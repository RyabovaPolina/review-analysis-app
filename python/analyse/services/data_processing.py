import pandas as pd
from io import BytesIO
from preprocessing import preprocess_corpus
from services.s3_service import _s3_pool 
from services.data_filter import filter_low_quality_reviews

def process_csv_streaming(s3_key, review_column, chunk_size=1000):
    # Получаем CSV через пул S3
    response = _s3_pool.client.get_object(
        Bucket=_s3_pool.bucket_name,
        Key=s3_key
    )
    csv_content = response['Body'].read()

    processed_texts = []
    valid_indices = []
    total_index = 0

    reader = pd.read_csv(
        BytesIO(csv_content),
        chunksize=chunk_size,
        dtype={review_column: 'object'}
    )

    for chunk in reader:
        # Фильтруем низкокачественные отзывы
        chunk_filtered = filter_low_quality_reviews(chunk, review_column)

        # Если после фильтрации нет строк, пропускаем
        if chunk_filtered.empty:
            total_index += len(chunk)
            continue

        # Препроцессинг
        raw_texts = chunk_filtered[review_column].fillna("").astype(str).tolist()
        processed = preprocess_corpus(raw_texts)
        processed_texts.extend(processed)

        # Сохраняем индексы valid_rows относительно всего CSV
        valid_indices.extend([total_index + i for i in chunk_filtered.index])

        total_index += len(chunk)

        del chunk, chunk_filtered
    return processed_texts, valid_indices
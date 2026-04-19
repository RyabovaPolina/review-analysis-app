"""
Скрипт анализа отзывов с использованием обученной ML-модели
"""
import sys
import json
import pandas as pd
import boto3
from io import BytesIO, StringIO
import os
from dotenv import load_dotenv
import joblib
import numpy as np
from collections import Counter
from preprocessing import preprocess_corpus
from scipy.sparse import hstack

# Загружаем переменные окружения
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=env_path)

# Инициализация S3 клиента
s3_client = boto3.client(
    's3',
    endpoint_url=os.getenv('S3_ENDPOINT'),
    aws_access_key_id=os.getenv('S3_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('S3_SECRET_KEY')
)
bucket_name = os.getenv('S3_BUCKET_NAME')

# Загрузка обученных моделей
print("📥 Загрузка моделей...", file=sys.stderr)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')

try:
    vectorizer_path = os.path.join(MODELS_DIR, 'vectorizer.pkl')
    classifier_path = os.path.join(MODELS_DIR, 'classifier.pkl')

    print(f"🔍 Ищем модели в: {MODELS_DIR}", file=sys.stderr)
    print(f"  Vectorizer: {os.path.exists(vectorizer_path)}", file=sys.stderr)
    print(f"  Classifier: {os.path.exists(classifier_path)}", file=sys.stderr)

    vectorizer = joblib.load(vectorizer_path)
    classifier = joblib.load(classifier_path)
    print("✅ Модели загружены успешно", file=sys.stderr)
except Exception as e:
    print(f"❌ Ошибка загрузки моделей: {e}", file=sys.stderr)
    print(f"💡 Текущая директория: {os.getcwd()}", file=sys.stderr)
    print(f"💡 Директория скрипта: {SCRIPT_DIR}", file=sys.stderr)
    print("💡 Запустите сначала train_model.py для обучения модели", file=sys.stderr)
    sys.exit(1)


def download_from_s3(s3_key):
    """Скачивание CSV из S3"""
    try:
        print(f"📥 Скачивание из S3: bucket={bucket_name}, key={s3_key}", file=sys.stderr)
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        csv_content = response['Body'].read()

        for sep in [None, ',', ';']:
            try:
                df = pd.read_csv(
                    BytesIO(csv_content),
                    sep=sep,
                    engine='python',
                    on_bad_lines='warn',
                    quotechar='"',
                    encoding='utf-8'
                )
                return df
            except Exception as e:
                print(f"⚠️ Попытка с sep={sep!r} не удалась: {e}", file=sys.stderr)

        raise Exception("Не удалось прочитать CSV ни с одним разделителем")

    except Exception as e:
        raise Exception(f"Ошибка скачивания из S3: {str(e)}")


def upload_to_s3(df, original_key):
    """Загрузка результатов в S3"""
    result_key = original_key.replace('uploads/', 'results/')
    if not result_key.endswith('_analyzed.csv'):
        result_key = result_key.replace('.csv', '_analyzed.csv')

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False, quoting=1)

    print(f"📤 Загружаем результаты в S3: {result_key}", file=sys.stderr)
    s3_client.put_object(
        Bucket=bucket_name,
        Key=result_key,
        Body=csv_buffer.getvalue().encode('utf-8'),
        ContentType='text/csv'
    )
    return result_key


def extract_keywords(processed_texts, top_n=15):

    if not processed_texts:
        return []

    X_word = vectorizer["word"].transform(processed_texts)

    feature_names = vectorizer["word"].get_feature_names_out()

    scores = np.asarray(
        X_word.sum(axis=0)
    ).flatten()

    top_indices = scores.argsort()[-top_n:][::-1]

    return [
        (feature_names[i], float(scores[i]))
        for i in top_indices
    ]

def analyze_reviews(s3_key, review_column):
    """Основная функция анализа"""
    try:
        print(f"🚀 Запуск ML-анализа...", file=sys.stderr)
        print(f"  Bucket: {bucket_name}", file=sys.stderr)
        print(f"  S3 Key: {s3_key}", file=sys.stderr)
        print(f"  Column: {review_column}", file=sys.stderr)

        # 1. Скачиваем файл
        df = download_from_s3(s3_key)
        print(f"✅ Файл загружен, строк: {len(df)}", file=sys.stderr)
        print(f"📋 Колонки: {list(df.columns)}", file=sys.stderr)

        if review_column not in df.columns:
            raise Exception(
                f"Колонка '{review_column}' не найдена. "
                f"Доступные: {list(df.columns)}"
            )

        # 2. Препроцессинг ОДИН РАЗ для всего датасета
        print(f"🔧 Препроцессинг текстов...", file=sys.stderr)
        raw_texts = df[review_column].fillna("").astype(str).tolist()
        processed_texts = preprocess_corpus(raw_texts, verbose=True, n_jobs=4)
        df['processed'] = processed_texts  # временная колонка

        # 3. Классификация
        print(f"🤖 Классификация тональности...", file=sys.stderr)


        X_word = vectorizer["word"].transform(processed_texts)

        X_char = vectorizer["char"].transform(processed_texts)

        X = hstack([
            X_word,
            X_char
        ])
        sentiments = classifier.predict(X)
        probabilities = classifier.predict_proba(X)

        df['sentiment'] = sentiments
        df['sentiment_score'] = probabilities.max(axis=1)
        sentiment_counts = Counter(sentiments)

        # 4. Ключевые слова — используем уже обработанные тексты
        print(f"📊 Извлечение ключевых слов...", file=sys.stderr)
        keywords = {}
        for sentiment_type in ['positive', 'negative', 'neutral']:
            mask = df['sentiment'] == sentiment_type
            processed_for_sentiment = df.loc[mask, 'processed'].tolist()
            kw = extract_keywords(processed_for_sentiment, top_n=15)
            keywords[sentiment_type] = [word for word, score in kw]

        # Убираем временную колонку перед сохранением
        df.drop(columns=['processed'], inplace=True)

        # 5. Загружаем результаты в S3
        print(f"📤 Загружаем результаты обратно в S3...", file=sys.stderr)
        result_key = upload_to_s3(df, s3_key)

        stats = {
            'positive_count': int(sentiment_counts.get('positive', 0)),
            'negative_count': int(sentiment_counts.get('negative', 0)),
            'neutral_count': int(sentiment_counts.get('neutral', 0)),
            'total_reviews': len(df),
            'avg_sentiment_score': float(probabilities.max(axis=1).mean()),
            'result_key': result_key,
            'keywords': keywords
        }

        print(f"✅ Анализ завершен успешно!", file=sys.stderr)
        print(f"  Положительных: {stats['positive_count']}", file=sys.stderr)
        print(f"  Отрицательных: {stats['negative_count']}", file=sys.stderr)
        print(f"  Нейтральных: {stats['neutral_count']}", file=sys.stderr)

        return stats

    except Exception as e:
        print(f"❌ Ошибка: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Недостаточно аргументов"}), file=sys.stderr)
        sys.exit(1)

    s3_key = sys.argv[1]
    review_column = sys.argv[2]

    try:
        result = analyze_reviews(s3_key, review_column)
        print(json.dumps(result, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)
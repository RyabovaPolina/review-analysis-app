import os
import io
import boto3
import pandas as pd
from io import BytesIO, StringIO
from botocore.config import Config
from dotenv import load_dotenv
import threading

env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

class S3ClientPool:
    """Пулинг S3 соединений для повторного использования"""
    
    def __init__(self):
        self._lock = threading.RLock()
        config = Config(
            max_pool_connections=50,

            connect_timeout=60,
            read_timeout=300,

            retries={
                'max_attempts': 3,
                'mode': 'adaptive'
            }
        )
        self.client = boto3.client(
            's3',
            endpoint_url=os.getenv('S3_ENDPOINT'),
            aws_access_key_id=os.getenv('S3_ACCESS_KEY'),
            aws_secret_access_key=os.getenv('S3_SECRET_KEY'),
            config=config
        )
        self.bucket_name = os.getenv('S3_BUCKET_NAME')

        
    def download_csv(self, s3_key):

        with self._lock:
            response = self.client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )

        return pd.read_csv(
            io.BytesIO(response['Body'].read()),
            encoding="utf-8"
        )
        
    def upload_csv(self, df, original_key):
        with self._lock:
            result_key = original_key.replace("uploads/", "results/")
            if not result_key.endswith("_analyzed.csv"):
                result_key = result_key.replace(".csv", "_analyzed.csv")
            buffer = StringIO()
            df.to_csv(buffer, index=False)
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=result_key,
                Body=buffer.getvalue().encode("utf-8"),
                ContentType="text/csv"
            )
        return result_key

# Глобальный пул, который переиспользуется
_s3_pool = S3ClientPool()

def download_csv(s3_key):
    return _s3_pool.download_csv(s3_key)

def upload_csv(df, s3_key):
    return _s3_pool.upload_csv(df, s3_key)
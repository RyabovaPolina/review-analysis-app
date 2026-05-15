import os
import boto3
import pandas as pd
from io import BytesIO, StringIO
from dotenv import load_dotenv

env_path = os.path.join(
    os.path.dirname(__file__),
    '..',
    '.env'
)

load_dotenv(env_path)

s3_client = boto3.client(
    's3',
    endpoint_url=os.getenv('S3_ENDPOINT'),
    aws_access_key_id=os.getenv('S3_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('S3_SECRET_KEY')
)

bucket_name = os.getenv('S3_BUCKET_NAME')


def download_csv(s3_key):

    response = s3_client.get_object(
        Bucket=bucket_name,
        Key=s3_key
    )

    csv_content = response['Body'].read()

    for sep in [None, ',', ';']:

        try:

            return pd.read_csv(
                BytesIO(csv_content),
                sep=sep,
                engine="python",
                encoding="utf-8"
            )

        except Exception:
            pass

    raise Exception("Cannot read CSV")


def upload_csv(df, original_key):

    result_key = original_key.replace(
        "uploads/",
        "results/"
    )

    if not result_key.endswith("_analyzed.csv"):

        result_key = result_key.replace(
            ".csv",
            "_analyzed.csv"
        )

    buffer = StringIO()

    df.to_csv(buffer, index=False)

    s3_client.put_object(

        Bucket=bucket_name,
        Key=result_key,
        Body=buffer.getvalue().encode("utf-8"),
        ContentType="text/csv"
    )

    return result_key
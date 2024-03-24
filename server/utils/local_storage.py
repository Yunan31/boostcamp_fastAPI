import boto3
import os
from dotenv import load_dotenv

load_dotenv()

s3 = None
BUCKET_NAME = None

def connect_s3():
    global s3, BUCKET_NAME
    s3 = boto3.client('s3')
    BUCKET_NAME = os.getenv("BUCKET_NAME")

def upload_file(file_path, object_path):
    s3.upload_file(file_path, BUCKET_NAME, object_path)

def download_file(object_path, file_path):
    s3.download_file(BUCKET_NAME, object_path, file_path)
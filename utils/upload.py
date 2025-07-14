import uuid
from google.cloud import storage
from dotenv import load_dotenv
import os
from datetime import timedelta
import logging

load_dotenv()
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

def upload_file_to_gcs(file_bytes: bytes, filename: str) -> str:

    file_type = filename.split(".")[-1]
    if file_type == "pdf":
        content_type = "application/pdf"
    elif file_type == "jpg" or file_type == "jpeg":
        content_type = "image/jpeg"
    elif file_type == "png":
        content_type = "image/png"
    elif file_type == "docx" or file_type == "doc":
        content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif file_type == "pptx" or file_type == "ppt":
        content_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    elif file_type == "txt":
        content_type = "text/plain"
    elif file_type == "csv":
        content_type = "text/csv"
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    
    filename = f"{uuid.uuid4()}_{filename}"
    blob = bucket.blob(filename)

    blob.upload_from_string(file_bytes, content_type=content_type)
    url = blob.generate_signed_url(expiration=timedelta(minutes=30))
    logging.info(f"File uploaded to GCS bucket {BUCKET_NAME}")

    if content_type == "image/png":
        url += "&ext=.png"
    elif content_type == "image/jpeg":
        url += "&ext=.jpg"

    return {
        "file_url": url,
        "file_type": file_type,
        "filename": filename
    }
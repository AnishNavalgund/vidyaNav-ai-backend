import uuid
import os
import logging
from datetime import datetime

def upload_file_to_gcs(file_bytes: bytes, filename: str) -> str:
    # Save file locally in 'uploads/' directory
    uploads_dir = os.path.join(os.getcwd(), 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)
    # Add timestamp and uuid to filename for uniqueness
    unique_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4()}_{filename}"
    file_path = os.path.join(uploads_dir, unique_filename)
    with open(file_path, 'wb') as f:
        f.write(file_bytes)
    logging.info(f"File saved locally at {file_path}")
    file_type = filename.split('.')[-1]
    return {
        "file_url": file_path,
        "file_type": file_type,
        "filename": unique_filename
    }
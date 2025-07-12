import uuid
from pathlib import Path

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def save_file(file_bytes: bytes, filename: str) -> str:
    upload_file_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{upload_file_id}_{filename}"
    with open(save_path, "wb") as f:
        f.write(file_bytes)
    return str(save_path)
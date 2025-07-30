import asyncio
from vertex_ai_imagen import ImagenClient
from visualaid_service.schema import VisualAidRequest, VisualAidOutput
from dotenv import load_dotenv
import logging
import os
import uuid
from google.cloud import storage
from datetime import timedelta

logger = logging.getLogger("visual_aid_service")
load_dotenv()


if os.getenv("DOCKER") == "true":
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/app/creds.json"
else:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

project_id = os.getenv("GCS_PROJECT_ID")

client = ImagenClient(project_id=project_id)
client.setup_credentials(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

# Note: not using pydantic‑ai’s Agent() because that is meant for LLMs and not image generators. so using simple Pydantic‑typed async function instead.

async def generate_visual_aid(request: VisualAidRequest) -> VisualAidOutput:

    full_prompt = (
        f"{request.prompt.strip().capitalize()}. Draw this as a simple, chalkboard-style educational diagram. "
        f"As a multigrade teacher in rural India, I want to use this image for my students on blackboard."
    )

    image = await client.generate(
        model="imagen-3.0-fast-generate-001",
        prompt=full_prompt,
        aspect_ratio="1:1",
        count=request.count
    )

    image_list = image if isinstance(image, list) else [image]
    
    outputs = []

    for idx, img in enumerate(image_list):
        # Save locally
        local_path = f"generated_image_{uuid.uuid4().hex}.png"
        img.save(local_path)

        # Upload to GCS
        with open(local_path, "rb") as f:
            file_bytes = f.read()
        upload_result = upload_file_to_gcs(file_bytes, filename=f"visual_aid_{uuid.uuid4().hex}.png")
        image_url = upload_result["file_url"]

        outputs.append(VisualAidOutput(
            image_url=image_url,
            caption=request.prompt,
            topic=request.prompt.title(),
            grade_range="Primary School"
        ))
        
    # flattinng list of dicts
    return [o.model_dump() for o in outputs]

BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

def upload_file_to_gcs(file_bytes: bytes, filename: str) -> dict:
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    blob = bucket.blob(unique_filename)
    blob.upload_from_string(file_bytes, content_type="image/png")
    url = blob.generate_signed_url(expiration=timedelta(hours=12))
    logging.info(f"File uploaded to GCS bucket {BUCKET_NAME}")
    logging.info(f"Generated signed URL: {url}")
    return {
        "file_url": url,
        "file_type": "png",
        "filename": unique_filename
    }

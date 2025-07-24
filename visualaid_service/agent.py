import asyncio
from vertex_ai_imagen import ImagenClient
from visualaid_service.schema import VisualAidRequest, VisualAidOutput
from dotenv import load_dotenv
import logging
import os
from utils.upload import upload_file_to_gcs
import uuid

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

        # dont upload to GCS
        # with open(local_path, "rb") as f:
        #     file_bytes = f.read()

        # upload_result = upload_file_to_gcs(file_bytes, filename="visual_aid.png")
        # image_url = upload_result["file_url"]

        outputs.append(VisualAidOutput(
            image_url=local_path,
            caption=request.prompt,
            topic=request.prompt.title(),
            grade_range="Primary School"
        ))
        
    # flattinng list of dicts
    return [o.model_dump() for o in outputs]

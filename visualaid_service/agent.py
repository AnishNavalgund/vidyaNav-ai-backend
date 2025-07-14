import asyncio
from vertex_ai_imagen import ImagenClient
# from pydantic_ai import Agent
from visualaid_service.schema import VisualAidRequest, VisualAidOutput
from dotenv import load_dotenv
import logging
import os
from utils.upload import upload_file_to_gcs
import uuid

logger = logging.getLogger("visual_aid_service")
load_dotenv()

# get from env
#client = ImagenClient(project_id=os.getenv("GCS_PROJECT_ID"))   
#client.setup_credentials("creds.json")

if os.getenv("DOCKER") == "true":
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/app/creds.json"
else:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

project_id = os.getenv("GCS_PROJECT_ID")

client = ImagenClient(project_id=project_id)
client.setup_credentials(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

# Note: not using pydantic‑ai’s Agent() because that is meant for LLMs and not image generators. so using simple Pydantic‑typed async function instead.

async def generate_visual_aid(request: VisualAidRequest) -> VisualAidOutput:

    print(f"\n >>> generate_visual_aid()  ")
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
        print(" >>> Image Saved local ")

        # Upload to GCS
        with open(local_path, "rb") as f:
            file_bytes = f.read()

        upload_result = upload_file_to_gcs(file_bytes, filename="visual_aid.png")
        image_url = upload_result["file_url"]

        outputs.append(VisualAidOutput(
            image_url=image_url,
            caption=request.prompt,
            topic=request.prompt.title(),
            grade_range="Primary School"
        ))
        
    # flattinng list of dicts
    print(" >>>>> Output:", [o.model_dump() for o in outputs])
    
    return [o.model_dump() for o in outputs]

    #print(" \n >>> image =", image)
    #print(" \n >>> type(image) =", type(image))
    #local_path = "generated_image.png"
    #image.save(local_path)
    #print(f"\n >>> Image Saved local ")
#
    #with open(local_path, "rb") as f:
    #    file_bytes = f.read()
#
    #upload_result = upload_file_to_gcs(file_bytes, filename="visual_aid.png")
    #image_url = upload_result["file_url"]
#
    #output = VisualAidOutput(
    #    image_url= image_url,
    #    caption=request.prompt,
    #    topic=request.prompt.title(),
    #)


# visual_aid_agent = Agent(
#     model=ImagenModel(),
#     output_type=VisualAidOutput,
#     deps_type=None,
#     system_prompt=(
#         "You are a helpful AI assistant for teachers in rural Indian schools. "
#         "Given a short prompt like 'draw the water cycle' or 'show a food chain', "
#         "you must generate simple, clear, age-appropriate visual diagrams suitable for a blackboard or printed handout. "
#         "Return a short caption."
#     )
# )


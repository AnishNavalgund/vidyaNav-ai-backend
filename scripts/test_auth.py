from google.cloud import aiplatform
from dotenv import load_dotenv
import os

load_dotenv()

aiplatform.init(project=os.getenv("GCP_PROJECT_ID"), location="us-central1")
print("Authenticated and ready to use Vertex AI.")
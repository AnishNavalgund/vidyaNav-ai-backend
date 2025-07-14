from pydantic import BaseModel, HttpUrl

class VisualAidRequest(BaseModel):
    prompt: str
    count: int = 1  

class VisualAidOutput(BaseModel):
    image_url: HttpUrl
    caption: str
    topic: str
from pydantic import BaseModel, HttpUrl, Field

class VisualAidRequest(BaseModel):
    prompt: str = Field(..., description="The visual topic or concept to illustrate")
    count: int = Field(default=1, ge=1, le=5, description="Number of image variations to generate")

class VisualAidOutput(BaseModel):
    image_url: HttpUrl = Field(..., description="Public or signed URL to the generated image")
    caption: str = Field(..., description="Short description or original prompt")
    topic: str = Field(..., description="Title")
# instantknowledge_service/schema.py

from typing import Optional, List
from pydantic import BaseModel, Field

class QuestionRequest(BaseModel):
    question: str = Field(..., example="Why is the sky blue?")
    grade_level: int = Field(..., ge=1, le=12, example=5)
    language: str = Field(..., example="English")

class AnswerResponse(BaseModel):
    answer: str = Field(..., description="Simplified answer using analogy.")
    analogy_used: bool = Field(..., description="Was an analogy used?")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score.")
    model_used: Optional[str] = Field(default=None, description="LLM used for answer.")
    source_chunks: Optional[List[str]] = Field(
        default=None,
        description="Textbook chunks used as context for answering."
    )

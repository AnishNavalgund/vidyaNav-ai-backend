from pydantic import BaseModel, Field
from typing import Optional

class WorksheetOutput(BaseModel):
    grade_1: Optional[str] = Field(None, description="Worksheet for grade 1")
    grade_2: Optional[str] = Field(None, description="Worksheet for grade 2")
    grade_3: Optional[str] = Field(None, description="Worksheet for grade 3")
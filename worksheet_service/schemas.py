from pydantic import BaseModel, Field
from typing import Optional

class WorksheetOutput(BaseModel):
    grade_1: Optional[str] = Field(default=None, description="Worksheet content for Grade 1")
    grade_2: Optional[str] = Field(default=None, description="Worksheet content for Grade 2")
    grade_3: Optional[str] = Field(default=None, description="Worksheet content for Grade 3")
    grade_4: Optional[str] = Field(default=None, description="Worksheet content for Grade 4")
    grade_5: Optional[str] = Field(default=None, description="Worksheet content for Grade 5")
    grade_6: Optional[str] = Field(default=None, description="Worksheet content for Grade 6")


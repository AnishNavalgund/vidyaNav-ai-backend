from pydantic_ai import Agent, ImageUrl
from worksheet_service.schemas import WorksheetOutput

from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_vertex import GoogleVertexProvider

model = GeminiModel(
    model_name="gemini-2.5-flash",
    provider=GoogleVertexProvider(region="us-central1")
)
worksheet_agent = Agent(

    
    model=model,
    output_type=WorksheetOutput,
    deps_type=None,
    system_prompt=(
        "You are a helpful AI assistant for teachers in rural Indian schools. "
        "You are given an image of a textbook page. Generate simple, age-appropriate worksheets "
        "for the requested grade levels and return them as a JSON dictionary."
    )
)

async def generate_worksheets(image_url: str, grade_input: str, language: str = "English") -> dict:

    def process_grades(input):
        # print(f">>> In Func {input}")
        grades = []
        for part in input.split(','):
            # print(f">>>>>> Part  {part}")
            part = part.strip()
            if part.isdigit() and len(part) == 1:
                grades.append(part)
        return grades

    grades = process_grades(grade_input)
    
    input_parts = [
        f"Generate simple worksheets for the following grades: {', '.join(grades)}. "
        f"The content should be in {language}. Use the attached textbook image.",
        ImageUrl(url=image_url)
    ]

    result = await worksheet_agent.run(input_parts)

    #print("\n >>>>>>>>>>>>>>>>> RESULT FROM GEMINI: \n")
    #print(result)

    output = result.output

    #print("\n >>>>>>>>>>>>>>>>> OUTPUT FROM GEMINI: \n")
    #print(output)

    return {
        k: v for k, v in output.model_dump().items()
        if v is not None
    }
from instantknowledge_service.schema import AnswerResponse
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_vertex import GoogleVertexProvider

model = GeminiModel(
    model_name="gemini-2.0-flash-lite",
    provider=GoogleVertexProvider(region="us-central1")
)

# Define agent with structured system prompt
instant_knowledge_agent = Agent(
    model=model,
    output_type=AnswerResponse,
    deps_type=None,
    system_prompt=(
        "You are an educational assistant helping a teacher answer student questions. "
        "Use simple, clear language with relatable analogies suitable for a Grade {grade_level} student. "
        "Respond only in {language}. Make it culturally relevant to Indian classrooms. "
        "Return the result in the following JSON format:\n"
        "{\n"
        '  "answer": "your explanation here",\n'
        '  "analogy_used": true,\n'
        '  "confidence_score": 0.85\n'
        "}\n"
        "Only return valid JSON with no commentary or prefix."
    )
)

from instantknowledge_service.schema import AnswerResponse
from langchain_google_vertexai import VertexAI
from langchain.output_parsers import PydanticOutputParser

llm = VertexAI(model_name="gemini-2.0-flash-lite")
parser = PydanticOutputParser(pydantic_object=AnswerResponse)

def get_instant_knowledge_answer(prompt: str) -> AnswerResponse:
    llm_response = llm.invoke(f"{prompt}\n{parser.get_format_instructions()}")
    return parser.parse(llm_response)

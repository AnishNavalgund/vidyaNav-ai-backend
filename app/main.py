import os

import logging
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, Form, HTTPException, File, Body, Request
from fastapi.middleware.cors import CORSMiddleware

from utils.upload import upload_file_to_gcs

from worksheet_service.agents import generate_worksheets

from instantknowledge_service.local_rag import get_answer_with_uploaded_textbook
from instantknowledge_service.schema import AnswerResponse

from visualaid_service.agent import generate_visual_aid
from visualaid_service.schema import VisualAidRequest, VisualAidOutput

from typing import List, Optional
import tempfile
import traceback
import json
import re

from langchain_google_vertexai import VertexAI
from langchain.prompts import PromptTemplate
from worksheet_service.schemas import WorksheetOutput
from instantknowledge_service.schema import AnswerResponse
from visualaid_service.schema import VisualAidOutput
from fastapi import UploadFile, File, Form


#  Load .env at startup
load_dotenv()

def clean_llm_response(response: str) -> str:
    # Remove ```json and ``` from the string
    cleaned = re.sub(r"```json|```", "", response).strip()
    return json.loads(cleaned)

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))

app = FastAPI(title="VidyaNav-ai API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:9002"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/generate-worksheet/")
async def generate_worksheet(
    file: UploadFile,
    grades: str = Form(...),
    language: str = Form("English")
):
    logging.info(f"Received request to generate worksheet")
    try:
        file_bytes = await file.read()
        upload_result = upload_file_to_gcs(file_bytes, file.filename)
        logging.info(f"Upload result: file_url={upload_result['file_url']}, file_type={upload_result['file_type']}, filename={upload_result['filename']}")
        logging.info(f"Calling generate_worksheets with file_url={upload_result['file_url']}, grades={grades}, language={language}")
        output = await generate_worksheets(
            file_url=upload_result["file_url"],
            grade_input=grades,
            language=language
        )
        logging.info("Worksheet generation successful.")
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/instant-knowledge-upload", response_model=AnswerResponse)
async def ask_with_uploaded_textbook(
    question: str = Form(...),
    grade_level: int = Form(...),
    language: str = Form(...),
    textbook: UploadFile = File(...)
):
    if textbook.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        file_bytes = await textbook.read()
        upload_result = upload_file_to_gcs(file_bytes, textbook.filename)
        logging.info(f"Textbook uploaded to GCS")
        def save_temp_pdf(file_bytes: bytes, suffix: str = ".pdf") -> str:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode='wb') as tmp:
                tmp.write(file_bytes)
                tmp.flush()
                os.fsync(tmp.fileno())  # Ensure data is written to disk
            return tmp.name

        temp_pdf_path = save_temp_pdf(file_bytes)

        # Run RAG pipeline
        response = await get_answer_with_uploaded_textbook(
            question=question,
            grade_level=grade_level,
            language=language,
            pdf_path=temp_pdf_path
        )

        logging.info("Instant knowledge answer generated successfully.")
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/visual-aid", response_model=List[VisualAidOutput])
async def generate_visual_aid_endpoint(request: VisualAidRequest):
    logging.info(f"Received visual aid request: {request}")
    try:
        return await generate_visual_aid(request)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai-assistant/")
async def ai_assistant(
    prompt: str = Form(...),
    file: Optional[UploadFile] = File(None)

):
    """
    Unified endpoint for teacher prompt. Detects intent and routes to the correct agent.
    """
    llm = VertexAI(model_name="gemini-2.0-flash")
    intent_prompt = PromptTemplate(
        input_variables=["prompt"],
        template = (
        "You are an intent classifier and information extractor for a teacher's AI assistant.\n"
        "Given the teacher's prompt, classify the intent as one of:\n"
        "- worksheet\n"
        "- instant_knowledge\n"
        "- visual_aid\n\n"
        "Also extract these optional fields if they are present:\n"
        "- grade (number), language (word), count (number).\n"
        "If a field is not found, return null.\n\n"
        "Respond only in valid JSON like this:\n"
        '{{"intent": "worksheet", "grade": 5, "language": "English", "count": 10}}'
        '\n\n'
        "Prompt: {prompt}"
        )
    )
    intent_raw = llm.invoke(intent_prompt.format(prompt=prompt)).strip().lower()

    logging.info(f">>>>>>>>>>>>> Intent raw: {intent_raw}")
    logging.info(f">>>>>>>>>>>>> Type of intent_raw: {type(intent_raw)}")
    intent_data = clean_llm_response(intent_raw)
    logging.info(f">>>>>>>>>>>>> Intent data: {intent_data}")
    logging.info(f">>>>>>>>>>>>> Type of intent_data: {type(intent_data)}")


    try:
        intent = intent_data["intent"]
        grade = intent_data["grade"]
        language = intent_data["language"]
    
        count_raw = intent_data.get("count")
        count = int(count_raw) if count_raw is not None else None

        logging.info(f"Intent: {intent}, Grade: {grade}, Language: {language}, Count: {count}")
    except Exception:
        intent = ""
        grade = None
        language = None
        count = None
    logging.info(f">>>> Intent: {intent}, Grade: {grade}, Language: {language}, Count: {count}")
    # Helper: treat empty string filename as missing file
    file_missing = (file is None or getattr(file, 'filename', '') == '' or file.filename == '""')
    if intent == "worksheet":
        if file_missing or not grade:
            return {"error": "Worksheet generation requires a file and grades."}
        file_bytes = await file.read()
        from utils.upload import upload_file_to_gcs
        upload_result = upload_file_to_gcs(file_bytes, file.filename)
        return await generate_worksheets(
            file_url=upload_result["file_url"],
            grade_input=grade,
            language=language
        )
    elif intent == "instant_knowledge":
        if file_missing:
            return {"error": "Instant knowledge requires a PDF file."}
        file_bytes = await file.read()
        from utils.upload import upload_file_to_gcs
        upload_result = upload_file_to_gcs(file_bytes, file.filename)
        temp_pdf_path = upload_result["file_url"]
        return await get_answer_with_uploaded_textbook(
            question=prompt,
            grade_level=grade,
            language=language,
            pdf_path=temp_pdf_path
        )
    elif intent == "visual_aid":
        from visualaid_service.schema import VisualAidRequest
        req = VisualAidRequest(prompt=prompt, count=count)
        return await generate_visual_aid(req)
    else:
        return {"error": f"Could not classify intent. Detected: {intent}"}

@app.post("/chatbot/")
async def chatbot(
    request: Request,
    file: Optional[UploadFile] = File(None)
):
    """
    Chatbot endpoint: Accepts conversation history and optional file, detects intent, routes to agent, returns assistant reply.
    Accepts both application/json and multipart/form-data.
    """
    import json
    history = []
    # Try to get history from form or json
    if request.headers.get("content-type", "").startswith("multipart/form-data"):
        form = await request.form()
        history_str = form.get("history")
        if history_str:
            history = json.loads(history_str)
    else:
        body = await request.body()
        if body:
            try:
                data = json.loads(body)
                history = data.get("history", [])
            except Exception:
                history = []
    # If no history, start with a single user message if file is not None
    if not history:
        # Try to get user message from form or json
        user_message = None
        if request.headers.get("content-type", "").startswith("multipart/form-data"):
            form = await request.form()
            user_message = form.get("prompt")
        else:
            try:
                data = json.loads(body)
                user_message = data.get("prompt")
            except Exception:
                user_message = None
        if user_message:
            history = [{"role": "user", "content": user_message}]
        else:
            return {"role": "assistant", "content": "Please provide a message."}
    # Get the latest user message
    user_message = history[-1]["content"]
    # Build conversation context for LLM
    conversation = "\n".join([
        f"{msg['role'].capitalize()}: {msg['content']}" for msg in history
    ])
    # Intent detection prompt
    llm = VertexAI(model_name="gemini-2.5-flash")
    intent_prompt = PromptTemplate(
        input_variables=["conversation", "user_message"],
        template=(
            "You are an intent and parameter extractor for a teacher's AI assistant. "
            "Given the conversation so far and the latest user message, extract:\n"
            "- intent: one of 'worksheet', 'instant_knowledge', 'visual_aid'\n"
            "- grade: the grade level (if mentioned, else null)\n"
            "- language: the language (if mentioned, else null)\n"
            "- count: number of images (if visual aid, else null)\n"
            "Return a JSON object with keys 'intent', 'grade', 'language', 'count'.\n"
            "Conversation:\n{conversation}\nUser: {user_message}\nJSON:"
        )
    )
    intent_raw = llm.invoke(intent_prompt.format(conversation=conversation, user_message=user_message)).strip()
    def clean_llm_response(response: str) -> dict:
        import re, json
        cleaned = re.sub(r"```json|```", "", response).strip()
        return json.loads(cleaned)
    intent_data = clean_llm_response(intent_raw)
    intent = intent_data.get("intent", "").strip().lower()
    grade = intent_data.get("grade") or "3"
    language = intent_data.get("language") or "English"
    count_raw = intent_data.get("count")
    try:
        count = int(count_raw)
        if count < 1:
            count = 1
    except (TypeError, ValueError):
        count = 1
    # Helper: treat empty string filename as missing file
    file_missing = (file is None or getattr(file, 'filename', '') == '' or file.filename == '""')
    # Routing
    if intent == "worksheet":
        if file_missing or not grade:
            return {"role": "assistant", "content": "Please upload an image or textbook."}
        file_bytes = await file.read()
        from utils.upload import upload_file_to_gcs
        upload_result = upload_file_to_gcs(file_bytes, file.filename)
        result = await generate_worksheets(
            file_url=upload_result["file_url"],
            grade_input=grade,
            language=language
        )
        return {"role": "assistant", "content": result}
    elif intent == "instant_knowledge":
        if file_missing:
            return {"role": "assistant", "content": "Please upload a textbook PDF."}
        file_bytes = await file.read()
        from utils.upload import upload_file_to_gcs
        upload_result = upload_file_to_gcs(file_bytes, file.filename)
        temp_pdf_path = upload_result["file_url"]
        answer = await get_answer_with_uploaded_textbook(
            question=user_message,
            grade_level=int(grade) if grade and grade.isdigit() else 3,
            language=language,
            pdf_path=temp_pdf_path
        )
        return {"role": "assistant", "content": answer}
    elif intent == "visual_aid":
        from visualaid_service.schema import VisualAidRequest
        req = VisualAidRequest(prompt=user_message, count=count)
        visual_aid = await generate_visual_aid(req)
        return {"role": "assistant", "content": visual_aid}
    else:
        return {"role": "assistant", "content": {"error": f"Could not classify intent. Detected: {intent_data}"}}

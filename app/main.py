import os
from fastapi import FastAPI, UploadFile, Form, HTTPException, File
import tempfile
import logging
from dotenv import load_dotenv

from utils.upload import upload_file_to_gcs

from worksheet_service.agents import generate_worksheets

from instantknowledge_service.local_rag import get_answer_with_uploaded_textbook
from instantknowledge_service.schema import AnswerResponse


# from tts_service.agent import generate_speech_prompt, synthesize_speech
# from pydantic import BaseModel

#  Load .env at startup
load_dotenv()

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))

app = FastAPI(title="VidyaNav-ai API")

@app.post("/generate-worksheet/")
async def generate_worksheet(
    file: UploadFile,
    grades: str = Form(...),
    language: str = Form("English")
):
    try:
        file_bytes = await file.read()
        upload_result = upload_file_to_gcs(file_bytes, file.filename)

        output = await generate_worksheets(
            image_url=upload_result["file_url"],
            grade_input=grades,
            language=language
        )
        return output

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

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
        print("Uploaded to GCS:", upload_result["filename"])

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

        response.model_dump()["textbook_gcs_file"] = upload_result["filename"]
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#class TTSRequest(BaseModel):
#    text: str
#
#
#class TTSResponse(BaseModel):
#    prompt: str
#    audio_base64: str
#
#
#@app.post("/tts", response_model=TTSResponse)
#async def text_to_speech(request: TTSRequest):
#    spoken_prompt = await generate_speech_prompt(request.text)
#    audio_b64 = synthesize_speech(spoken_prompt)
#    return TTSResponse(prompt=spoken_prompt, audio_base64=audio_b64)
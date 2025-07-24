import os

import logging
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, Form, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware

from utils.upload import upload_file_to_gcs

from worksheet_service.agents import generate_worksheets

from instantknowledge_service.local_rag import get_answer_with_uploaded_textbook
from instantknowledge_service.schema import AnswerResponse

from visualaid_service.agent import generate_visual_aid
from visualaid_service.schema import VisualAidRequest, VisualAidOutput

from typing import List
import tempfile
import traceback


#  Load .env at startup
load_dotenv()


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

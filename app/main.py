import os
from fastapi import FastAPI, UploadFile, Form, HTTPException
from dotenv import load_dotenv

from utils.upload import upload_file_to_gcs
from worksheet_service.agents import generate_worksheets

from instantknowledge_service.agent import get_instant_answer
from instantknowledge_service.schema import AnswerResponse, QuestionRequest


# from tts_service.agent import generate_speech_prompt, synthesize_speech
# from pydantic import BaseModel

#  Load .env at startup
load_dotenv()

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
async def ask_question(request: QuestionRequest):
    try:
        response = await get_instant_answer(
            question=request.question,
            grade_level=request.grade_level,
            language=request.language
        )
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
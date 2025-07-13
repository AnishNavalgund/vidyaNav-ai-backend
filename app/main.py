import os
from fastapi import FastAPI, UploadFile, Form, HTTPException
from dotenv import load_dotenv

from utils.upload import upload_file_to_gcs
from worksheet_service.agents import generate_worksheets

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

        grade_list = [g.strip() for g in grades.split(",")]

        output = await generate_worksheets(
            image_url=upload_result["file_url"],
            grades=grade_list,
            language=language
        )
        return output

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

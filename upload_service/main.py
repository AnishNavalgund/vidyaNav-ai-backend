from fastapi import FastAPI, UploadFile, File
from upload_service.storage import save_file

app = FastAPI()

@app.post("/fileupload/")
async def upload(file: UploadFile = File(...)):
    file_bytes = await file.read()
    saved_path = save_file(file_bytes, file.filename)
    return {"file_path": saved_path}
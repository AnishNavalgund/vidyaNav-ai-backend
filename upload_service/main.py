from fastapi import FastAPI, UploadFile, File
from upload_service.storage import upload_file

app = FastAPI()

@app.post("/fileupload/")
async def fileupload(file: UploadFile = File(...)):
    # read the file as bytes
    file_bytes = await file.read() 
    result = upload_file(file_bytes, file.filename)
    return result
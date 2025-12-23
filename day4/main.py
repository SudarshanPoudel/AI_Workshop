import os
import tempfile
from typing import List
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from rag import load_pdf, get_answer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/uploadfile")
async def upload_file(file: UploadFile):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            file_content = await file.read()
            tmp.write(file_content)
            tmp_path = tmp.name
        
        filename = file.filename
        load_pdf(tmp_path, filename)
        os.unlink(tmp_path)
        return {"message": "Files uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

@app.get("/ask")
async def ask_question(question: str):
    try:
        resp = get_answer(question)
        return {"answer": resp.answer, "filename": resp.filename, "page_no": resp.page_no}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
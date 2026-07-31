import os
import shutil
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from parser import ResumeParser

app = FastAPI(
    title="Automated Resume Analyzer API",
    description="Upload a PDF or DOCX resume and receive structured JSON output.",
    version="1.0.0",
)

# Load the parser once at startup (spaCy model load is expensive).
resume_parser = ResumeParser()

ALLOWED_EXTENSIONS = {".pdf", ".docx"}


@app.get("/")
def read_root():
    return {
        "message": "Automated Resume Analyzer API is running.",
        "usage": "POST a .pdf or .docx file to /parse-resume/",
    }


@app.post("/parse-resume/")
async def parse_resume(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Only .pdf and .docx are allowed.",
        )

    # Save the uploaded file to a temporary path since the parser reads from disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = resume_parser.parse_file(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse resume: {exc}") from exc
    finally:
        os.remove(tmp_path)

    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
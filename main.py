import uuid
import logging
import logging.config

from typing import List
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks

from services.inference import Inference

logging.config.fileConfig("logging.ini", disable_existing_loggers=False)
logger = logging.getLogger(__name__)
logger.info("Starting local_rag...")

# silence noisy loggers
logging.getLogger('pdfinterp').setLevel(logging.ERROR)
logging.getLogger('ppocr').setLevel(logging.ERROR)

logger.info("Starting local_rag...")
app = FastAPI(title="Local RAG System with Ollama", version="0.0.1")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# initialize main inference service (includes database, vector store, LLM, and PDF processing)
inference = Inference()

# define endpoints
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_files(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    
    processed_files = []
    errors = []
    
    for file in files:
        if not file.filename or not file.filename.endswith('.pdf'):
            errors.append(f"{file.filename} is not a PDF file")
            continue
            
        try:
            file_id = str(uuid.uuid4())
            file_path = f"uploads/{file_id}_{file.filename}"
            
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # add file to database with unprocessed status
            file_data = {
                "file_id": file_id,
                "filename": file.filename,
                "file_path": file_path,
                "file_size": len(content)
            }
            
            await inference.database.add_file(file_data)
            
            background_tasks.add_task(inference.process_single_file, file, file_id, file_path)
            
            processed_files.append({
                "filename": file.filename,
                "file_id": file_id,
                "status": "processing"
            })
            
        except Exception as e:
            logger.error(f"Error uploading {file.filename}: {str(e)}")
            errors.append(f"Error uploading {file.filename}: {str(e)}")
    
    return {
        "processed_files": processed_files,
        "errors": errors,
        "message": f"Files uploaded successfully. {len(processed_files)} file(s) processing in background."
    }


@app.post("/ask")
async def ask_question(question: str = Form(...)):
    """Ask a question and get an answer with sources"""
    
    try:
        result = inference.ask_question(question)
        
        return {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "confidence": result.get("confidence", 0.0)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.get("/files")
async def get_files():
    """Get all uploaded files with their status"""
    try:
        files = await inference.get_files()
        return {"files": files}
    except Exception as e:
        logger.error(f"Error getting files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """Delete a file and all its chunks"""
    try:
        success = await inference.delete_file(file_id)
        
        if success:
            return {"message": "File deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        logger.error(f"Error deleting file {file_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/files")
async def delete_all_files():
    """Delete all files and chunks"""
    try:
        deleted_count = await inference.delete_all_files()
        return {"message": f"Successfully deleted {deleted_count} files and all chunks", "deleted_count": deleted_count}
    except Exception as e:
        logger.error(f"Error deleting all files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def get_status():
    """Get system status and document count"""
    try:
        return await inference.get_status()
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

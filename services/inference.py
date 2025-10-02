import logging

from fastapi import UploadFile
from typing import List, Dict, Any
from services.storage.chroma_vector_store import ChromaVectorStore
from services.storage.database import Database
from services.llm.ollama_service import OllamaService
from services.llm.rag_pipeline import RAGPipeline
from services.pdf_processor import PDFProcessor


logger = logging.getLogger(__name__)

# TODO: add async support for all methods
# TODO: change name, it is not exactly `inference`
# TODO: move away other services initilization

class Inference:
    
    def __init__(self):
        # MongoDB instance for persistant statuses, filenames, pdfs, etc.
        self.database = Database()

        # LLM-related services
        self.ollama_service = OllamaService()
        self.pdf_processor = PDFProcessor(self.ollama_service)
        self.ollama_service.ensure_models()

        # Vector and RAG pipeline
        self.vector_store = ChromaVectorStore()
        self.rag_pipeline = RAGPipeline(self.vector_store, self.ollama_service)

        logger.info("Inference service initialized successfully")

    async def process_single_file(self, file: UploadFile, file_id: str, file_path: str):
        try:
            # 1) upload PDF file to mongoDB
            await self.database.update_file_status(file_id, "processing")

            # 2) process PDF fiile
            result = self.pdf_processor.process_pdf(file_path, file_id)
            if not result["success"]:
                raise Exception(result.get("error", "Unknown error"))

            summary = result["summary"]
            documents_count = result["documents_count"]

            # 3) update file status in DB
            await self.database.update_file_status(
                file_id,
                "completed",
                chunks_count=documents_count,
                summary=summary,
                processed_text=result.get("content_length", 0),
            )

            logger.info(f"Successfully processed {file.filename} with {documents_count} documents")
            
            self.rag_pipeline.refresh_index()
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            await self.database.update_file_status(file_id, "error", error=str(e))

    def ask_question(self, question: str) -> Dict[str, Any]:
        result = self.rag_pipeline.answer_question(question)
        return result

    async def get_files(self) -> List[Dict[str, Any]]:
        return await self.database.get_all_files()

    async def delete_file(self, file_id: str) -> bool:
        if self.vector_store:
            self.vector_store.delete_documents_by_file(file_id)
        return await self.database.delete_file(file_id)

    async def delete_all_files(self) -> int:
        files = await self.database.get_all_files()
        if self.vector_store:
            for file in files:
                self.vector_store.delete_documents_by_file(file["file_id"])
        await self.database.files_collection.delete_many({})
        await self.database.chunks_collection.delete_many({})
        return len(files)

    async def get_status(self) -> Dict[str, Any]:
        files_count = await self.database.get_files_count()
        chunks_count = await self.database.get_total_chunks_count()
        vector_count = self.vector_store.get_document_count() if self.vector_store else 0

        return {
            "status": "ready",
            "files_count": files_count,
            "chunks_count": chunks_count,
            "vector_count": vector_count,
            "vector_store_ready": self.vector_store.is_ready() if self.vector_store else False,
        }



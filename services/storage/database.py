from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

class Database:
    def __init__(self, connection_string: str = None):
        if connection_string is None:
            connection_string = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
        
        self.client = AsyncIOMotorClient(connection_string)
        self.db = self.client.rag_system
        self.files_collection = self.db.files
        self.chunks_collection = self.db.chunks
    
    async def add_file(self, file_data: Dict[str, Any]) -> str:
        """Add file metadata to database"""
        file_data["created_at"] = datetime.now()
        file_data["status"] = "unprocessed"
        result = await self.files_collection.insert_one(file_data)
        return str(result.inserted_id)
    
    async def update_file_status(self, file_id: str, status: str, chunks_count: int = None, error: str = None, summary: str = None, processed_text: str = None):
        """Update file processing status"""
        update_data = {"status": status, "updated_at": datetime.now()}
        if chunks_count is not None:
            update_data["chunks_count"] = chunks_count
        if error:
            update_data["error"] = error
        if summary:
            update_data["summary"] = summary
        if processed_text:
            update_data["processed_text"] = processed_text
        
        await self.files_collection.update_one(
            {"file_id": file_id},
            {"$set": update_data}
        )
    
    async def get_unprocessed_files(self) -> List[Dict[str, Any]]:
        """Get all unprocessed files"""
        cursor = self.files_collection.find({"status": "unprocessed"})
        files = []
        async for file in cursor:
            file["_id"] = str(file["_id"])
            files.append(file)
        return files
    
    async def get_all_files(self) -> List[Dict[str, Any]]:
        """Get all files with their status"""
        cursor = self.files_collection.find().sort("created_at", -1)
        files = []
        async for file in cursor:
            file["_id"] = str(file["_id"])
            files.append(file)
        return files
    
    async def get_file_by_id(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get file by file_id"""
        file = await self.files_collection.find_one({"file_id": file_id})
        if file:
            file["_id"] = str(file["_id"])
        return file
    
    async def delete_file(self, file_id: str) -> bool:
        """Delete file and all its chunks"""
        file_result = await self.files_collection.delete_one({"file_id": file_id})
        chunks_result = await self.chunks_collection.delete_many({"file_id": file_id})
        
        return file_result.deleted_count > 0
    
    async def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Add document chunks to database"""
        if chunks:
            await self.chunks_collection.insert_many(chunks)
    
    async def get_chunks_by_file(self, file_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific file"""
        cursor = self.chunks_collection.find({"file_id": file_id})
        chunks = []
        async for chunk in cursor:
            chunk["_id"] = str(chunk["_id"])
            chunks.append(chunk)
        return chunks
    
    async def get_total_chunks_count(self) -> int:
        """Get total number of chunks across all files"""
        return await self.chunks_collection.count_documents({})
    
    async def get_files_count(self) -> int:
        """Get total number of files"""
        return await self.files_collection.count_documents({})
    
    async def close(self):
        """Close database connection"""
        self.client.close()

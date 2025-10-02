from typing import List, Dict, Any
import logging
from pathlib import Path
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

class ChromaVectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"description": "PDF document chunks for RAG"}
        )
        
        logger.info("ChromaDB vector store initialized")
    
    def is_ready(self) -> bool:
        """Check if vector store is ready"""
        try:
            return self.collection is not None
        except Exception:
            return False
    
    def get_document_count(self) -> int:
        """Get total number of documents in the store"""
        try:
            return self.collection.count()
        except Exception:
            return 0
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to ChromaDB vector store"""
        try:
            if not documents:
                return
            
            # Prepare data for ChromaDB
            texts = []
            metadatas = []
            ids = []
            
            for i, doc in enumerate(documents):
                text = doc.get("text", "")
                metadata = doc.get("metadata", {})
                
                # Generate unique ID
                doc_id = f"{metadata.get('file_id', 'unknown')}_{metadata.get('chunk_index', i)}"
                
                texts.append(text)
                metadatas.append(metadata)
                ids.append(doc_id)
            
            # Add to ChromaDB
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to ChromaDB")
            
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
    
    def search_similar(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using ChromaDB"""
        try:
            if self.collection.count() == 0:
                return []
            
            # Search in ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        "text": doc,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                        "score": results['distances'][0][i] if results['distances'] and results['distances'][0] else 0.0
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return []
    
    def delete_documents_by_file(self, file_id: str):
        """Delete documents by file ID"""
        try:
            # Get all documents with the file_id
            results = self.collection.get(
                where={"file_id": file_id}
            )
            
            if results['ids']:
                # Delete by IDs
                self.collection.delete(ids=results['ids'])
                deleted_count = len(results['ids'])
                logger.info(f"Deleted {deleted_count} documents for file {file_id}")
                return deleted_count
            
            return 0
            
        except Exception as e:
            logger.error(f"Error deleting documents from ChromaDB: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "index_type": "ChromaDB",
                "is_ready": self.is_ready()
            }
        except Exception as e:
            logger.error(f"Error getting ChromaDB stats: {e}")
            return {
                "total_documents": 0,
                "index_type": "ChromaDB",
                "is_ready": False
            }

import os
import logging
from typing import Dict, Any
from llama_index.core import Document
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama

from services.storage.chroma_vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, vector_store: ChromaVectorStore):
        self.vector_store = vector_store
        
        self._create_index_from_chroma()
    
    def _create_index_from_chroma(self):
        """Create LlamaIndex from ChromaDB collection"""
        try:
            # check if collection has any documents
            doc_count = self.vector_store.get_document_count()
            logger.info(f"ChromaDB document count: {doc_count}")
            
            if doc_count == 0:
                logger.info("No documents in ChromaDB, index will be None")
                self.index = None
                return
            
            # fet all documents from ChromaDB
            results = self.vector_store.collection.get()
            logger.info(f"Retrieved {len(results['ids'])} documents from ChromaDB")
            
            if not results['ids']:
                logger.warning("No document IDs found in ChromaDB results")
                self.index = None
                return
            
            # create documents for LlamaIndex
            documents = []
            
            for _, (doc_id, text, metadata) in enumerate(zip(results['ids'], results['documents'], results['metadatas'])):
                doc = Document(
                    text=text,
                    metadata=metadata or {},
                    id_=doc_id
                )
                documents.append(doc)
            
            logger.info(f"Created {len(documents)} LlamaIndex documents")
            
            # create index from documents
            self.index = VectorStoreIndex.from_documents(documents)
            logger.info("LlamaIndex created successfully")
            
        except Exception as e:
            logger.error(f"Error creating index from ChromaDB: {e}")
            self.index = None
    
    def answer_question(self, question: str, max_sources: int = 5) -> Dict[str, Any]:
        """Answer a question using LlamaIndex RAG pipeline"""
        try:
            if not self.index:
                return {
                    "answer": "No documents available. Please upload some documents first.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            llm = Ollama(
                model="llama3.2:3b",
                request_timeout=360.0,
                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            )            
            
            query_engine = self.index.as_query_engine(
                similarity_top_k=max_sources,
                response_mode="compact",
                llm=llm
            )
            
            logger.info(f"Querying: {question[:100]}...")

            response = query_engine.query(question)
            answer = str(response.response)
            logger.info("Query completed successfully")

            
            # get sources
            sources = []
            if 'response' in locals() and hasattr(response, 'source_nodes') and response.source_nodes:
                for node in response.source_nodes:
                    metadata = node.metadata or {}
                    sources.append({
                        "file": metadata.get('source_file', 'Unknown'),
                        "chunk_index": metadata.get('chunk_index', 0),
                        "text_preview": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                        "similarity_score": getattr(node, 'score', 0.8)
                    })
            elif 'nodes' in locals() and nodes:
                # TODO: do we need this?
                # fallback sources from retriever
                for i, node in enumerate(nodes[:max_sources]):
                    metadata = node.metadata or {}
                    sources.append({
                        "file": metadata.get('source_file', 'Unknown'),
                        "chunk_index": metadata.get('chunk_index', i),
                        "text_preview": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                        "similarity_score": getattr(node, 'score', 0.7)
                    })
            
            # TODO: recheck that, not sure it works as intended
            # calculate confidence
            confidence = 0.0
            if sources:
                scores = [s.get('similarity_score', 0) for s in sources]
                confidence = round((sum(scores) / len(scores)) * 100, 1)
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return {
                "answer": f"Error processing question: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }
    
    def refresh_index(self):
        """Refresh the index from ChromaDB (call this after adding new documents)"""
        logger.info("Refreshing LlamaIndex from ChromaDB...")
        self._create_index_from_chroma()


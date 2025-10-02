import ollama
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
import logging
import base64

logger = logging.getLogger(__name__)

class OllamaService:
    
    def __init__(self, model: str = "llama3.2:3b", embedding_model: str = "nomic-embed-text", vision_model: str = "llava:7b"):
        import os
        
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model
        self.embedding_model = embedding_model
        self.vision_model = vision_model
                
        self.client = ollama.Client(host=self.base_url)
        self.llm = Ollama(model=model, base_url=self.base_url, timeout=300.0)
        self.embed_model = OllamaEmbedding(model_name=embedding_model, base_url=self.base_url, timeout=300.0)
        
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        logger.info(f"Ollama service initialized with model: {model}, embedding: {embedding_model}, vision: {vision_model}")
    
    
    def pull_model(self, model_name: str) -> bool:
        """Pull model from ollama.com"""
        try:
            self.client.pull(model_name)
            logger.info(f"Successfully pulled model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    
    def ensure_models(self) -> None:
        """Check for required models and pull any that are missing."""
        models = self.client.list()
        existing = {m.get('name') for m in models.get('models', [])}
        required = {m for m in [self.model, self.embedding_model, self.vision_model] if m}

        for model_name in required:
            if model_name not in existing:
                logger.info(f"Pulling model: {model_name}")
                self.pull_model(model_name)
    
    
    def describe_image(self, image_path: str, context: str = "") -> str:
        """Describe an image using Ollama vision model"""
        try:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            if context:
                prompt = f"Describe this image in detail. Context: {context}"
            else:
                prompt = "Describe this image in detail, focusing on any diagrams, charts, or visual elements."
            
            response = self.client.generate(
                model=self.vision_model,
                prompt=prompt,
                images=[image_data],
                options={
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 500
                }
            )
            
            return response['response'].strip()
            
        except Exception as e:
            logger.error(f"Error describing image {image_path}: {e}")
            return f"[Error describing image: {e}]"
    
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using Ollama vision model"""
        try:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            prompt = "Extract all text from this image. Return only the text content, no additional description."
            
            response = self.client.generate(
                model=self.vision_model,
                prompt=prompt,
                images=[image_data],
                options={
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 500
                }
            )
            
            return response['response'].strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from image {image_path}: {e}")
            return f"[Error extracting text: {e}]"
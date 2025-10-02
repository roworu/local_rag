import numpy as np
from PIL import Image
import io
from typing import List, Dict, Any
import logging
from paddleocr import PaddleOCR


logger = logging.getLogger(__name__)        


class OCRService:

    def __init__(self):

        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en')
    
    def extract_text_from_image(self, image_data: bytes) -> List[Dict[str, Any]]:
        """Extract text from image using PaddleOCR"""
        
        # TODO: rewrite it, it looks ugly
        # TODO: add logging for each step, encoding, NumPy transformation, etc..
        try:
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image)
            result = self.ocr.ocr(image_array, cls=True)
            
            # process results
            extracted_texts = []
            if result and result[0] is not None:
                for detection in result[0]:
                    if detection:  # skip None detections
                        bbox = detection[0]
                        text_info = detection[1]
                        
                        if isinstance(text_info, tuple) and len(text_info) == 2:
                            text, confidence = text_info
                            
                            if confidence > 0.5 and text.strip():
                                extracted_texts.append({
                                    "text": text,
                                    "confidence": confidence,
                                    "bbox": bbox
                                })
            
            return extracted_texts
            
        except Exception as e:
            logger.error(f"Error in OCR processing: {e}")
            return []
    
    def process_images_in_markdown(self, md_path: str, images_data: List[Dict[str, Any]]) -> str:
        """Process images and insert OCR text into markdown"""

        # TODO: rewrite it, it looks ugly
        # TODO: add better logging
        # TODO: add better error handling
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # replace image placeholders with OCR text
            for img_data in images_data:
                page_num = img_data.get('page', 0)
                img_index = img_data.get('index', 0)
                placeholder = f"![Image {img_index}](image_placeholder_{page_num}_{img_index})"
                
                if placeholder in content:
                    ocr_results = self.extract_text_from_image(img_data['data'])
                    
                    if ocr_results:
                        ocr_text = " ".join([result['text'] for result in ocr_results])
                        replacement = f"**OCR Text from Image {img_index}:**\n{ocr_text}\n"
                        content = content.replace(placeholder, replacement)
                    else:
                        content = content.replace(placeholder, f"*[Image {img_index} - No text detected]*")
            
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return content
            
        except Exception as e:
            logger.error(f"Error processing images in markdown: {e}")
            return ""

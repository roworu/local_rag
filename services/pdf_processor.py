import fitz
import camelot
import logging
import pdfplumber

from pathlib import Path
from typing import List, Dict, Any

from services.ocr_service import OCRService
from services.markdown_cleaner import MarkdownCleaner
from services.storage.chroma_vector_store import ChromaVectorStore
from services.llm.ollama_service import OllamaService


logger = logging.getLogger(__name__)


class PDFProcessor:

    def __init__(self, ollama_service: OllamaService):
        self.ollama_service = ollama_service

        self.ocr_service = OCRService()
        self.markdown_cleaner = MarkdownCleaner()
        self.vector_store = ChromaVectorStore()

        logger.info("PDF processor initialized")

    def process_pdf(self, pdf_path: str, file_id: str) -> Dict[str, Any]:
        try:
            context: Dict[str, Any] = {"pdf_path": pdf_path, "file_id": file_id, "md_path": None}

            logger.info(f"PDFProcessor: Step 1 - extract_text for {pdf_path}")
            step = self.extract_text(context)
            if not step["result"]:
                return {"success": False, "error": step["error_message"], "failed_step": "extract_text"}

            logger.info("PDFProcessor: Step 2 - extract_images")
            step = self.extract_images(context)
            if not step["result"]:
                return {"success": False, "error": step["error_message"], "failed_step": "extract_images"}

            logger.info("PDFProcessor: Step 3 - generate_markdown")
            step = self.generate_markdown(context)
            if not step["result"]:
                return {"success": False, "error": step["error_message"], "failed_step": "generate_markdown"}
            self._save_debug_state(context.get("md_path"), "01_initial_markdown", context)

            logger.info("PDFProcessor: Step 4 - run_ocr")
            step = self.run_ocr(context)
            if not step["result"]:
                return {"success": False, "error": step["error_message"], "failed_step": "run_ocr"}
            self._save_debug_state(context.get("md_path"), "02_after_ocr", context)

            logger.info("PDFProcessor: Step 5 - clean_markdown")
            step = self.clean_markdown(context)
            if not step["result"]:
                return {"success": False, "error": step["error_message"], "failed_step": "clean_markdown"}
            self._save_debug_state(context.get("md_path"), "03_after_cleanup", context)

            logger.info("PDFProcessor: Step 6 - process_diagrams")
            step = self.process_diagrams(context)
            if not step["result"]:
                return {"success": False, "error": step["error_message"], "failed_step": "process_diagrams"}
            self._save_debug_state(context.get("md_path"), "04_after_diagrams", context)

            logger.info("PDFProcessor: Step 7 - create_vector_docs")
            step = self.create_vector_docs(context)
            if not step["result"]:
                return {"success": False, "error": step["error_message"], "failed_step": "create_vector_docs"}

            logger.info("PDFProcessor: Step 8 - generate_summary")
            step = self.generate_summary(context)
            if not step["result"]:
                return {"success": False, "error": step["error_message"], "failed_step": "generate_summary"}
            self._save_debug_state(context.get("md_path"), "05_final_state", context)

            logger.info("PDFProcessor: Done processing PDF")
            return {
                "success": True,
                "markdown_path": context.get("md_path"),
                "documents_count": len(context.get("documents", [])),
                "images_processed": len(context.get("images_data", [])),
                "summary": context.get("summary", ""),
                "content_length": len(context.get("cleaned_content", "")),
            }

        except Exception as e:
            logger.error(f"Error in PDF processing: {e}")
            return {"success": False, "error": str(e)}

    def extract_text(self, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            pdf_path: str = context["pdf_path"]
            pymupdf_content = self._extract_with_pymupdf(pdf_path)
            camelot_content = self._extract_with_camelot(pdf_path)
            combined = self._combine_content(pymupdf_content, camelot_content)
            context["combined_content"] = combined
            return {"result": True, "error_message": ""}
        except Exception as e:
            logger.error(f"extract_text failed: {e}")
            return {"result": False, "error_message": str(e)}

    def extract_images(self, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            combined = context.get("combined_content", {})
            images_data = combined.get("images", [])
            context["images_data"] = images_data
            return {"result": True, "error_message": ""}
        except Exception as e:
            logger.error(f"extract_images failed: {e}")
            return {"result": False, "error_message": str(e)}

    def run_ocr(self, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            images_data = context.get("images_data", [])
            md_path = context.get("md_path")
            if images_data and md_path:
                try:
                    self.ocr_service.process_images_in_markdown(md_path, images_data)
                except Exception as e:
                    logger.warning(f"OCR processing failed: {e}")
            return {"result": True, "error_message": ""}
        except Exception as e:
            logger.error(f"run_ocr failed: {e}")
            return {"result": False, "error_message": str(e)}

    def process_diagrams(self, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            images_data = context.get("images_data", [])
            md_path = context.get("md_path")
            if images_data and md_path:
                try:
                    self._process_diagrams_with_ollama(md_path, images_data)
                except Exception as e:
                    logger.warning(f"LLaVA processing via Ollama failed: {e}")
            return {"result": True, "error_message": ""}
        except Exception as e:
            logger.error(f"process_diagrams failed: {e}")
            return {"result": False, "error_message": str(e)}

    def generate_markdown(self, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            pdf_path: str = context["pdf_path"]
            combined = context.get("combined_content", {})
            md_path = str(self._create_markdown_file(pdf_path, combined))
            context["md_path"] = md_path
            return {"result": True, "error_message": ""}
        except Exception as e:
            logger.error(f"generate_markdown failed: {e}")
            return {"result": False, "error_message": str(e)}

    def clean_markdown(self, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            md_path = context.get("md_path")
            if not md_path:
                return {"result": False, "error_message": "Markdown path not set"}
            
            cleaned_content = self.markdown_cleaner.clean_markdown_file(md_path)
            context["cleaned_content"] = cleaned_content
            return {"result": True, "error_message": ""}
        except Exception as e:
            logger.error(f"clean_markdown failed: {e}")
            return {"result": False, "error_message": str(e)}

    def create_vector_docs(self, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            md_path = context.get("md_path")
            file_id = context.get("file_id")
            if not md_path or not file_id:
                return {"result": False, "error_message": "Missing md_path or file_id"}
            documents = self._create_documents_from_markdown(md_path, file_id)
            context["documents"] = documents
            simple_docs = [{"text": d.text, "metadata": d.metadata} for d in documents]
            if simple_docs:
                self.vector_store.add_documents(simple_docs)
            return {"result": True, "error_message": ""}
        except Exception as e:
            logger.error(f"create_vector_docs failed: {e}")
            return {"result": False, "error_message": str(e)}

    def generate_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            cleaned_content = context.get("cleaned_content", "")
            summary = self._generate_summary(cleaned_content)
            context["summary"] = summary
            return {"result": True, "error_message": ""}
        except Exception as e:
            logger.error(f"generate_summary failed: {e}")
            return {"result": False, "error_message": str(e)}

    def _create_documents_from_markdown(self, md_path: str, file_id: str) -> List[Any]:
        try:
            from llama_index.core.schema import Document

            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = self._split_markdown_content(content)

            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    text=chunk,
                    metadata={
                        "file_id": file_id,
                        "chunk_index": i,
                        "source_file": Path(md_path).name,
                        "total_chunks": len(chunks),
                    },
                )
                documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Error creating documents: {e}")
            return []

    def _split_markdown_content(self, content: str, chunk_size: int = 1000) -> List[str]:
        chunks = []
        current = []

        for para in content.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            if sum(len(p) for p in current) + len(para) + len(current) <= chunk_size:
                current.append(para)
            else:
                chunks.append("\n\n".join(current))
                current = [para]

        if current:
            chunks.append("\n\n".join(current))

        return chunks

    def _generate_summary(self, content: str) -> str:
        try:

            if len(content) > 4000:
                content = content[:4000]

            prompt = f"""
            Provide a short summary of the following document in 1 sentence. Focus on the main topics, key points, and important information. No introduction or conclusion, just the summary itself. Document:
            ---
            {content}
            ---
            Summary:"""

            response = self.ollama_service.client.generate(
                model=self.ollama_service.model,
                prompt=prompt,
                options={"temperature": 0.3, "top_p": 0.9, "num_predict": 30}
            )

            return response["response"].strip()

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Failed to generate summary: {e}"

    def _process_diagrams_with_ollama(self, md_path: str, images_data: List[Dict[str, Any]]) -> str:
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()

            import tempfile
            import os
            with tempfile.TemporaryDirectory() as tmp_dir:
                for img_data in images_data:
                    page_num = img_data.get('page', 0)
                    img_index = img_data.get('index', 0)
                    placeholder = f"![Image {img_index}](image_placeholder_{page_num}_{img_index})"

                    if placeholder in content:
                        context = self._get_image_context(content, placeholder)
                        tmp_path = os.path.join(tmp_dir, f"diagram_{page_num}_{img_index}.png")
                        with open(tmp_path, 'wb') as tmp_file:
                            tmp_file.write(img_data['data'])

                        description = self.ollama_service.describe_image(tmp_path, context)
                        replacement = f"**Diagram/Chart Description:**\n{description}\n"
                        content = content.replace(placeholder, replacement)

            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(content)

            return content
        except Exception as e:
            logger.error(f"Error processing diagrams via Ollama: {e}")
            return ""

    def _get_image_context(self, content: str, placeholder: str) -> str:
        try:
            pos = content.find(placeholder)
            if pos == -1:
                return ""
            start = max(0, pos - 200)
            end = min(len(content), pos + len(placeholder) + 200)
            context = content[start:end]
            context = content.replace(placeholder, "[IMAGE]")
            return " ".join(context.split())[:300]
        except Exception:
            return ""

    def _extract_with_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        doc = fitz.open(pdf_path)
        content: Dict[str, Any] = {"pages": [], "images": [], "text_blocks": []}

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            text = page.get_text()

            image_list = page.get_images(full=True)
            page_images = []
            for img_index, img in enumerate(image_list):
                xref = img[0]
                rects = page.get_image_rects(xref)
                bbox = rects[0] if rects else None
                pix = fitz.Pixmap(doc, xref)
                try:
                    if pix.n - pix.alpha < 4:
                        img_data = pix.tobytes("png")
                        page_images.append({
                            "index": img_index,
                            "data": img_data,
                            "page": page_num,
                            "bbox": [bbox.x0, bbox.y0, bbox.x1, bbox.y1] if bbox else None,
                        })
                finally:
                    pix = None

            text_blocks = page.get_text("dict")

            content["pages"].append({
                "page_num": page_num,
                "text": text,
                "images": page_images,
                "text_blocks": text_blocks,
            })
            content["images"].extend(page_images)
            content["text_blocks"].extend(text_blocks.get("blocks", []))

        doc.close()
        return content

    def _normalize_table(self, table: List[List[str]]) -> List[List[str]]:
        max_cols = max(len(row) for row in table)
        norm = []
        for row in table:
            clean_row = [(c or "").strip().replace("\n", " ") for c in row]
            while len(clean_row) < max_cols:
                clean_row.append("")
            norm.append(clean_row)
        return norm


    def _extract_with_camelot(self, pdf_path: str) -> Dict[str, Any]:
        content: Dict[str, Any] = {"pages": [], "tables": [], "text_objects": []}

        # TODO: need to implement more stramliened logic on how to pick between camelot and pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                tables = []
                camelot_tables = []
                try:
                    camelot_tables = camelot.read_pdf(
                        pdf_path, pages=str(page_num + 1), flavor="lattice"
                    )
                except Exception as e:
                    logger.warning(f"Camelot failed on page {page_num}: {e}")
    
                if camelot_tables:
                    for t in camelot_tables:
                        rows = self._normalize_table(t.df.values.tolist())
                        if len(rows) > 1 and len(set(map(len, rows))) == 1:
                            tables.append({"page": page_num, "data": rows})
                else:
                    # fallback: pdfplumber
                    try:
                        plumb_tables = page.extract_tables()
                        for t in plumb_tables:
                            rows = self._normalize_table(t)
                            if len(rows) > 1 and len(set(map(len, rows))) == 1:
                                tables.append({"page": page_num, "data": rows})
                    except Exception as e:
                        logger.warning(f"pdfplumber tables failed on page {page_num}: {e}")
    
                content["pages"].append({
                    "page_num": page_num,
                    "text": text,
                    "tables": tables,
                })
                content["tables"].extend(tables)
                content["text_objects"].extend(page.chars)
        return content


    def _combine_content(self, pymupdf_content: Dict[str, Any], camelot_content: Dict[str, Any]) -> Dict[str, Any]:

        combined: Dict[str, Any] = {
            "pages": [],
            "images": pymupdf_content.get("images", []),
            "tables": camelot_content.get("tables", []),
            "text_objects": camelot_content.get("text_objects", []),
        }

        max_pages = max(len(pymupdf_content.get("pages", [])), len(camelot_content.get("pages", [])))

        for i in range(max_pages):
            pymupdf_page = pymupdf_content.get("pages", [{}])[i] if i < len(pymupdf_content.get("pages", [])) else {}
            camelot_page = camelot_content.get("pages", [{}])[i] if i < len(camelot_content.get("pages", [])) else {}

            combined_page = {
                "page_num": i,
                "text": pymupdf_page.get("text", "") or camelot_page.get("text", ""),
                "images": pymupdf_page.get("images", []),
                "tables": camelot_page.get("tables", []),
                "text_blocks": pymupdf_page.get("text_blocks", []),
            }
            combined["pages"].append(combined_page)

        return combined

    def _create_markdown_file(self, pdf_path: str, content: Dict[str, Any]) -> Path:
        pdf_name = Path(pdf_path).stem
        output_dir = Path("processed_docs")
        output_dir.mkdir(exist_ok=True)
        md_path = output_dir / f"{pdf_name}.md"
    
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {pdf_name}\n\n")
    
            for page in content.get("pages", []):
                f.write(f"## Page {page['page_num'] + 1}\n\n")
    
                if page.get("text") and not page.get("tables"):
                    f.write(f"{page['text']}\n\n")
    
                for table in page.get("tables", []):
                    rows = self._normalize_table(table.get("data", []))
                    if not rows:
                        continue
    
                    f.write("### Table\n\n")
                    header = rows[0]
                    f.write("| " + " | ".join(header) + " |\n")
                    f.write("| " + " | ".join(["---"] * len(header)) + " |\n")
                    for row in rows[1:]:
                        f.write("| " + " | ".join(row) + " |\n")
                    f.write("\n")
    
                for img in page.get("images", []):
                    f.write(f"![Image {img['index']}](image_placeholder_{img['page']}_{img['index']})\n\n")
    
        return md_path


    def _save_debug_state(self, md_path: str, stage: str, context: Dict[str, Any]) -> None:
        """Save debug state of markdown file at different processing stages"""

        # TODO: that functionality should be hidden after some env option
        try:
            import os
            from pathlib import Path
            
            debug_dir = Path("debug_md_states")
            debug_dir.mkdir(exist_ok=True)
            
            original_name = Path(md_path).stem
            file_id = context.get("file_id", "unknown")
            
            debug_filename = f"{stage}_{file_id}_{original_name}.md"
            debug_path = debug_dir / debug_filename
            
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(f"# Debug State: {stage}\n")
                f.write(f"# File ID: {file_id}\n")
                f.write(f"# Original: {original_name}\n")
                f.write(f"# Generated: {Path(md_path).name}\n\n")
                f.write("---\n\n")
                f.write(content)
            
            logger.info(f"Debug state saved: {debug_filename}")
            
        except Exception as e:
            logger.warning(f"Failed to save debug state {stage}: {e}")




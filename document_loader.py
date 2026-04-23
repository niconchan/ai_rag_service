import ray
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader
import logging

logger = logging.getLogger(__name__)


class Document:
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(content_length={len(self.content)}, metadata={self.metadata})"


@ray.remote
class PdfProcessor:
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            reader = PdfReader(file_path)
            text_parts = []
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    text_parts.append({
                        "text": text,
                        "page": page_num + 1,
                        "source": str(file_path)
                    })
            chunks = self._create_chunks(text_parts)
            return chunks
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return []

    def _create_chunks(self, text_parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_chunks = []
        for part in text_parts:
            text = part["text"]
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end]
                all_chunks.append({
                    "content": chunk_text.strip(),
                    "page": part["page"],
                    "source": part["source"]
                })
                start += self.chunk_size - self.overlap
        return all_chunks


def load_documents_from_folder(folder_path: str, chunk_size: int = 512, overlap: int = 50) -> List[Document]:
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder not found: {folder_path}")

    pdf_files = list(folder.glob("**/*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files to process")

    if not ray.is_initialized():
        ray.init()

    processor = PdfProcessor.remote(chunk_size=chunk_size, overlap=overlap)
    futures = [processor.process_pdf.remote(str(f)) for f in pdf_files]
    results = ray.get(futures)

    documents = []
    for file_result in results:
        for chunk in file_result:
            doc = Document(
                content=chunk["content"],
                metadata={
                    "source": chunk["source"],
                    "page": chunk["page"]
                }
            )
            documents.append(doc)

    logger.info(f"Created {len(documents)} document chunks")
    return documents


def load_single_pdf(file_path: str, chunk_size: int = 512, overlap: int = 50) -> List[Document]:
    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"File not found: {file_path}")

    if not ray.is_initialized():
        ray.init()

    processor = PdfProcessor.remote(chunk_size=chunk_size, overlap=overlap)
    results = ray.get([processor.process_pdf.remote(str(path))])[0]

    documents = []
    for chunk in results:
        doc = Document(
            content=chunk["content"],
            metadata={
                "source": chunk["source"],
                "page": chunk["page"]
            }
        )
        documents.append(doc)

    return documents

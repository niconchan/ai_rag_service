import ray
from typing import List, Dict, Any
from pathlib import Path
import logging

from config import config
from document_loader import Document, load_documents_from_folder, load_single_pdf
from embedding_service import EmbeddingService
from milvus_store import MilvusStore

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        milvus_host: str = None,
        milvus_port: int = None,
        milvus_collection: str = None,
        embedding_model: str = None,
        embedding_device: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        embedding_workers: int = 4
    ):
        self.milvus_host = milvus_host or config.milvus.host
        self.milvus_port = milvus_port or config.milvus.port
        self.milvus_collection = milvus_collection or config.milvus.collection_name
        self.embedding_model = embedding_model or config.embedding.model_name
        self.embedding_device = embedding_device or config.embedding.device
        self.chunk_size = chunk_size or config.chunk.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk.overlap

        self.embedding_service = EmbeddingService(
            model_name=self.embedding_model,
            device=self.embedding_device,
            batch_size=config.embedding.batch_size,
            num_workers=embedding_workers
        )

        self.milvus_store = MilvusStore(
            host=self.milvus_host,
            port=self.milvus_port,
            collection_name=self.milvus_collection,
            vector_dim=self.embedding_service.dimension
        )

    def ingest_folder(self, folder_path: str, rebuild_collection: bool = False):
        self.milvus_store.create_collection(if_delete=rebuild_collection)

        documents = load_documents_from_folder(
            folder_path,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap
        )

        self._ingest_documents(documents)
        logger.info(f"Completed ingesting folder: {folder_path}")

    def ingest_pdf(self, file_path: str, rebuild_collection: bool = False):
        self.milvus_store.create_collection(if_delete=rebuild_collection)

        documents = load_single_pdf(
            file_path,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap
        )

        self._ingest_documents(documents)
        logger.info(f"Completed ingesting PDF: {file_path}")

    def _ingest_documents(self, documents: List[Document]):
        texts = [doc.content for doc in documents]
        metadata = [doc.metadata for doc in documents]

        logger.info(f"Generating embeddings for {len(texts)} documents...")
        embeddings = self.embedding_service.encode(texts)

        logger.info(f"Inserting documents into Milvus...")
        self.milvus_store.insert(texts, embeddings, metadata)

    def close(self):
        self.milvus_store.close()
        if ray.is_initialized():
            ray.shutdown()

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from config import config
from embedding_service import EmbeddingService
from milvus_store import MilvusStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    content: str
    distance: float
    metadata: Dict[str, Any]


class RAGRetriever:
    def __init__(
        self,
        milvus_host: str = None,
        milvus_port: int = None,
        milvus_collection: str = None,
        embedding_model: str = None,
        embedding_device: str = None
    ):
        self.milvus_host = milvus_host or config.milvus.host
        self.milvus_port = milvus_port or config.milvus.port
        self.milvus_collection = milvus_collection or config.milvus.collection_name
        self.embedding_model = embedding_model or config.embedding.model_name
        self.embedding_device = embedding_device or config.embedding.device

        self.embedding_service = EmbeddingService(
            model_name=self.embedding_model,
            device=self.embedding_device,
            batch_size=config.embedding.batch_size,
            num_workers=2
        )

        self.milvus_store = MilvusStore(
            host=self.milvus_host,
            port=self.milvus_port,
            collection_name=self.milvus_collection,
            vector_dim=self.embedding_service.dimension
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        logger.info(f"Retrieving documents for query: {query}")

        query_embedding = self.embedding_service.encode_single(query)

        results = self.milvus_store.search(query_embedding, top_k=top_k)

        retrieval_results = []
        for result in results:
            retrieval_results.append(RetrievalResult(
                content=result["content"],
                distance=result["distance"],
                metadata={
                    "source": result["source"],
                    "page": result["page"]
                }
            ))

        return retrieval_results

    def retrieve_with_context(self, query: str, top_k: int = 5, max_context_length: int = 2000) -> str:
        results = self.retrieve(query, top_k=top_k)

        context_parts = []
        current_length = 0

        for result in results:
            chunk_length = len(result.content)
            if current_length + chunk_length > max_context_length:
                break
            context_parts.append(f"[Source: {result.metadata['source']}, Page {result.metadata['page']}]\n{result.content}")
            current_length += chunk_length

        return "\n\n".join(context_parts)

    def close(self):
        self.milvus_store.close()

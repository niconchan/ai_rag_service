import logging
from typing import List, Dict, Any, Optional
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

logger = logging.getLogger(__name__)


class MilvusStore:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "documents",
        vector_dim: int = 384
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.collection: Optional[Collection] = None
        self._connect()

    def _connect(self):
        try:
            connections.connect(host=self.host, port=self.port)
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def create_collection(self, if_delete: bool = False):
        if utility.has_collection(self.collection_name):
            if if_delete:
                utility.drop_collection(self.collection_name)
                logger.info(f"Dropped existing collection: {self.collection_name}")
            else:
                self.collection = Collection(self.collection_name)
                logger.info(f"Using existing collection: {self.collection_name}")
                return

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="page", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim)
        ]
        schema = CollectionSchema(fields=fields, description="Document collection for RAG")
        self.collection = Collection(name=self.collection_name, schema=schema)

        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)
        logger.info(f"Created collection: {self.collection_name}")

    def insert(self, texts: List[str], embeddings: List[List[float]], metadata: List[Dict[str, Any]]):
        if self.collection is None:
            raise ValueError("Collection not initialized")

        data = [
            texts,
            [m.get("source", "") for m in metadata],
            [m.get("page", 0) for m in metadata],
            embeddings
        ]
        result = self.collection.insert(data)
        self.collection.flush()
        logger.info(f"Inserted {len(texts)} documents into Milvus")
        return result

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        if self.collection is None:
            raise ValueError("Collection not initialized")

        self.collection.load()
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["content", "source", "page"]
        )

        hits = []
        for hits_per_query in results:
            for hit in hits_per_query:
                hits.append({
                    "id": hit.id,
                    "distance": hit.distance,
                    "content": hit.entity.get("content"),
                    "source": hit.entity.get("source"),
                    "page": hit.entity.get("page")
                })
        return hits

    def delete_all(self):
        if self.collection is None:
            raise ValueError("Collection not initialized")
        self.collection.delete(expr="id >= 0")
        self.collection.flush()
        logger.info("Deleted all documents from collection")

    def close(self):
        connections.disconnect(alias="default")
        logger.info("Disconnected from Milvus")

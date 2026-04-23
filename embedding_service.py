import ray
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


@ray.remote
class EmbeddingWorker:
    def __init__(self, model_name: str, device: str, batch_size: int):
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def encode(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=False)
        return embeddings.tolist()


class EmbeddingService:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        num_workers: int = 2
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.workers: List[ray.actor.ActorHandle] = []

    def _init_workers(self):
        if not ray.is_initialized():
            ray.init()
        if not self.workers:
            self.workers = [
                EmbeddingWorker.remote(self.model_name, self.device, self.batch_size)
                for _ in range(self.num_workers)
            ]
            logger.info(f"Initialized {self.num_workers} embedding workers")

    def encode(self, texts: List[str]) -> List[List[float]]:
        self._init_workers()

        batch_size = (len(texts) + self.num_workers - 1) // self.num_workers
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        futures = [worker.encode.remote(batch) for worker, batch in zip(self.workers, batches)]
        results = ray.get(futures)

        all_embeddings = []
        for result in results:
            all_embeddings.extend(result)

        return all_embeddings

    def encode_single(self, text: str) -> List[float]:
        return self.encode([text])[0]

    @property
    def dimension(self) -> int:
        model = SentenceTransformer(self.model_name)
        return model.get_sentence_embedding_dimension()

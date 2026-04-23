import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class MilvusConfig:
    host: str
    port: int
    collection_name: str


@dataclass
class RayConfig:
    address: str
    num_cpus: int


@dataclass
class EmbeddingConfig:
    model_name: str
    device: str
    batch_size: int


@dataclass
class ChunkConfig:
    chunk_size: int
    overlap: int


@dataclass
class PdfConfig:
    max_workers: int


@dataclass
class Config:
    milvus: MilvusConfig
    ray: RayConfig
    embedding: EmbeddingConfig
    chunk: ChunkConfig
    pdf: PdfConfig


def load_config(config_path: Optional[str] = None) -> Config:
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    return Config(
        milvus=MilvusConfig(**data['milvus']),
        ray=RayConfig(**data['ray']),
        embedding=EmbeddingConfig(**data['embedding']),
        chunk=ChunkConfig(**data['chunk']),
        pdf=PdfConfig(**data['pdf'])
    )


config = load_config()

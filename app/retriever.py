"""Chroma vector store with model2vec static embeddings (CPU, no torch)."""
import threading
from functools import lru_cache

import numpy  # noqa: F401  (must be imported before dspy anywhere in the process)
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from model2vec import StaticModel

from app.config import CHROMA_DIR, COLLECTION, EMBED_MODEL, TOP_K


@lru_cache(maxsize=None)
def _static_model(name: str) -> StaticModel:
    return StaticModel.from_pretrained(name)


class StaticEmbedding(EmbeddingFunction[Documents]):
    def __init__(self, model_name: str = EMBED_MODEL):
        self.model_name = model_name
        self._model = _static_model(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        return [v.tolist() for v in self._model.encode(list(input))]

    @staticmethod
    def name() -> str:
        return "model2vec"

    def get_config(self) -> dict:
        return {"model_name": self.model_name}

    @staticmethod
    def build_from_config(config: dict) -> "StaticEmbedding":
        return StaticEmbedding(config["model_name"])


_lock = threading.Lock()
_collection = None


def get_collection():
    # Chroma's client setup is not thread-safe; graphs run in parallel threads during eval/optimization.
    global _collection
    with _lock:
        if _collection is None:
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            _collection = client.get_or_create_collection(
                COLLECTION, embedding_function=StaticEmbedding(), metadata={"hnsw:space": "cosine"}
            )
        return _collection


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    res = get_collection().query(query_texts=[query], n_results=k)
    return [
        {"title": m["title"], "text": d}
        for d, m in zip(res["documents"][0], res["metadatas"][0])
    ]


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{i + 1}] {c['title']}: {c['text']}" for i, c in enumerate(chunks))

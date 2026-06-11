"""Векторный индекс на ChromaDB поверх произвольного эмбеддера.

``VectorIndex`` принимает любой объект, реализующий протокол
:class:`web_ai_assistant.embeddings.Embedder` (``E5Embedder``, ``GigaChatEmbedder``,
кастомный).

Для обратной совместимости сохранён ``E5VectorIndex`` — тонкая обёртка,
которая собирает ``VectorIndex`` поверх локального ``E5Embedder``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .corpus import Chunk

if TYPE_CHECKING:
    from .embeddings import Embedder


class VectorIndex:
    """ChromaDB-коллекция + сменный эмбеддер."""

    def __init__(self, embedder: Embedder, collection_name: str = "web_courses") -> None:
        import chromadb

        self.embedder = embedder
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return self.collection.count()
        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=self.embedder.embed_passages([c.text for c in chunks]),
            metadatas=[
                {"doc_id": c.doc_id, "title": c.title, "url": c.url} for c in chunks
            ],
        )
        return self.collection.count()

    def query(self, question: str, k: int = 4):
        res = self.collection.query(
            query_embeddings=[self.embedder.embed_query(question)], n_results=k
        )
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        sims = [1 - d for d in res["distances"][0]]  # cosine_sim = 1 - cosine_dist
        return docs, metas, sims


class E5VectorIndex(VectorIndex):
    """Backwards-compatible alias: ``VectorIndex(E5Embedder(device=...))``."""

    EMB_MODEL = "intfloat/multilingual-e5-large"

    def __init__(self, collection_name: str = "web_courses", device: str = "cuda") -> None:
        from .embeddings.e5 import E5Embedder

        super().__init__(embedder=E5Embedder(device=device), collection_name=collection_name)

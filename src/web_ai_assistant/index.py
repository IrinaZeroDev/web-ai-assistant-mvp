"""Векторный индекс: e5-multilingual + ChromaDB.

Модель эмбеддингов и параметры — как в ``project_plan.md`` раздел 7.2.
"""

from __future__ import annotations

from collections.abc import Iterable

from .corpus import Chunk


class E5VectorIndex:
    """Тонкая обёртка над ChromaDB + sentence-transformers/e5-multilingual."""

    EMB_MODEL = "intfloat/multilingual-e5-large"

    def __init__(self, collection_name: str = "web_courses", device: str = "cuda"):
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.emb = SentenceTransformer(self.EMB_MODEL, device=device)
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # --------- эмбеддинги (с префиксами как требует e5) ---------
    def embed_passages(self, texts: Iterable[str]) -> list[list[float]]:
        return self.emb.encode(
            [f"passage: {t}" for t in texts],
            normalize_embeddings=True,
            batch_size=16,
            show_progress_bar=False,
        ).tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.emb.encode(f"query: {query}", normalize_embeddings=True).tolist()

    # --------- индексация ---------
    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=self.embed_passages([c.text for c in chunks]),
            metadatas=[
                {"doc_id": c.doc_id, "title": c.title, "url": c.url} for c in chunks
            ],
        )
        return self.collection.count()

    # --------- поиск ---------
    def query(self, question: str, k: int = 4):
        res = self.collection.query(
            query_embeddings=[self.embed_query(question)], n_results=k
        )
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        sims = [1 - d for d in res["distances"][0]]  # cosine_sim = 1 - cosine_dist
        return docs, metas, sims

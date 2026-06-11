"""Локальный эмбеддер ``intfloat/multilingual-e5-large`` (1024 dim)."""

from __future__ import annotations

from collections.abc import Iterable


class E5Embedder:
    """Sentence-transformers e5-multilingual-large.

    Требует префиксы ``query:`` / ``passage:`` (это часть протокола обучения e5).
    """

    MODEL_ID = "intfloat/multilingual-e5-large"
    dim: int = 1024

    def __init__(self, device: str = "cuda", model_id: str | None = None):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_id or self.MODEL_ID, device=device)

    def embed_passages(self, texts: Iterable[str]) -> list[list[float]]:
        return self._model.encode(
            [f"passage: {t}" for t in texts],
            normalize_embeddings=True,
            batch_size=16,
            show_progress_bar=False,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(
            f"query: {text}", normalize_embeddings=True
        ).tolist()

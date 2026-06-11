"""VectorIndex работает с любым эмбеддером через протокол."""

from __future__ import annotations

import pytest

pytest.importorskip("chromadb")

from web_ai_assistant.corpus import Chunk
from web_ai_assistant.index import VectorIndex


class _FakeEmbedder:
    """Простая «эмбеддер»-заглушка: дим = 3, кодирует первые 3 буквы."""

    dim = 3

    def embed_passages(self, texts):
        return [self._enc(t) for t in texts]

    def embed_query(self, text):
        return self._enc(text)

    @staticmethod
    def _enc(t: str) -> list[float]:
        out = [0.0, 0.0, 0.0]
        for i, ch in enumerate(t.lower()[:3]):
            out[i] = (ord(ch) % 10) / 10.0
        return out


def _chunks() -> list[Chunk]:
    return [
        Chunk(chunk_id="a#0", doc_id="a", title="A", url="https://x/a", text="flex"),
        Chunk(chunk_id="b#0", doc_id="b", title="B", url="https://x/b", text="grid"),
        Chunk(chunk_id="c#0", doc_id="c", title="C", url="https://x/c", text="form"),
    ]


def test_add_and_query_with_custom_embedder():
    idx = VectorIndex(_FakeEmbedder(), collection_name="t_custom")
    n = idx.add(_chunks())
    assert n == 3
    docs, metas, sims = idx.query("flex me", k=2)
    assert len(docs) == 2
    assert all(0.0 <= s <= 1.0 for s in sims)
    titles = {m["title"] for m in metas}
    assert "A" in titles  # ближайший к "flex" — это документ A


def test_empty_add_returns_existing_count():
    idx = VectorIndex(_FakeEmbedder(), collection_name="t_empty")
    assert idx.add([]) == 0
    idx.add(_chunks())
    assert idx.add([]) == 3  # add([]) не падает и возвращает текущий count

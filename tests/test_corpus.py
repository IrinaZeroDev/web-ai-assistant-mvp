"""Тесты сбора корпуса и чанкования — без обращений к сети."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from web_ai_assistant.corpus import Document, load_corpus, save_corpus, split_documents


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    docs = [
        Document(doc_id="a", url="https://x/a", title="A", text="hello"),
        Document(doc_id="b", url="https://x/b", title="B", text="world"),
    ]
    path = tmp_path / "corpus.jsonl"
    save_corpus(docs, path)
    loaded = load_corpus(path)
    assert loaded == docs
    # формат — JSONL
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["doc_id"] == "a"


@pytest.mark.skipif(
    not pytest.importorskip("langchain", reason="langchain not installed"),
    reason="langchain not installed",
)
def test_split_documents_produces_chunks() -> None:
    long_text = ("CSS flexbox layout. " * 200).strip()
    docs = [Document(doc_id="css/fb", url="https://x", title="CSS/flexbox", text=long_text)]
    chunks = split_documents(docs, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    # ids уникальны и имеют формат "{doc_id}#{i}"
    assert all(c.chunk_id.startswith("css/fb#") for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    # метаданные пробрасываются
    assert all(c.title == "CSS/flexbox" for c in chunks)

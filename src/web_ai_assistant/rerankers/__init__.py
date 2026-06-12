"""Cross-encoder реранкеры для подъёма faithfulness RAG.

Реранкер — это второй слой retrieval'а: после bi-encoder retrieval (ChromaDB
поверх e5/GigaChat embeddings) мы перепроверяем top-K фрагментов более точной
моделью, которая видит ``query`` и ``passage`` одновременно.

Все реализации удовлетворяют протоколу :class:`Reranker`:

.. code-block:: python

    class Reranker(Protocol):
        def rerank(self, query: str, candidates: list[str]) -> list[float]: ...

Возвращает relevance-скор для каждого кандидата (порядок сохраняется).

Доступные реализации:

- :class:`web_ai_assistant.rerankers.bge.BGEReranker` — локальный
  ``BAAI/bge-reranker-v2-m3`` (multilingual cross-encoder, GPU желателен).
- :class:`web_ai_assistant.rerankers.gigachat.GigaChatReranker` — GigaChat
  как LLM-judge (без GPU, проксируется через Sber).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

__all__ = ["Reranker", "BGEReranker", "GigaChatReranker"]


@runtime_checkable
class Reranker(Protocol):
    """Минимальный интерфейс реранкера."""

    def rerank(self, query: str, candidates: Sequence[str]) -> list[float]: ...


def __getattr__(name: str):
    if name == "BGEReranker":
        from .bge import BGEReranker

        return BGEReranker
    if name == "GigaChatReranker":
        from .gigachat import GigaChatReranker

        return GigaChatReranker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

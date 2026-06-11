"""Эмбеддер-провайдеры для :class:`web_ai_assistant.index.VectorIndex`.

Все эмбеддеры реализуют один протокол :class:`Embedder`:

- ``embed_passages(texts: Iterable[str]) -> list[list[float]]`` — для индексации;
- ``embed_query(text: str) -> list[float]`` — для запроса;
- атрибут ``dim: int`` — размерность векторов.

Готовые реализации:

- :class:`web_ai_assistant.embeddings.e5.E5Embedder` — локальный
  ``intfloat/multilingual-e5-large`` (1024 dim, GPU желателен).
- :class:`web_ai_assistant.embeddings.gigachat.GigaChatEmbedder` — облачный
  GigaChat (``Embeddings`` 1024 / ``EmbeddingsGigaR`` 2560).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

__all__ = ["Embedder", "E5Embedder", "GigaChatEmbedder"]


@runtime_checkable
class Embedder(Protocol):
    """Минимальный интерфейс эмбеддера."""

    dim: int

    def embed_passages(self, texts: Iterable[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


def __getattr__(name: str):
    if name == "E5Embedder":
        from .e5 import E5Embedder

        return E5Embedder
    if name == "GigaChatEmbedder":
        from .gigachat import GigaChatEmbedder

        return GigaChatEmbedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

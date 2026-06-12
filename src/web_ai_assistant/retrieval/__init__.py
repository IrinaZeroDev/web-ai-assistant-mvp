"""Алгоритмы пост-обработки retrieval-кандидатов.

Сейчас доступно:

- :func:`web_ai_assistant.retrieval.mmr.mmr_select` — **Maximal Marginal Relevance**
  (Carbonell & Goldstein, 1998): балансирует релевантность запросу и
  разнообразие выбранных фрагментов.
"""

from __future__ import annotations

__all__ = ["mmr_select"]


def __getattr__(name: str):
    if name == "mmr_select":
        from .mmr import mmr_select

        return mmr_select
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

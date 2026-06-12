"""Логирование запросов и аналитика затруднений.

Подмодули:

- :mod:`web_ai_assistant.analytics.storage` — SQLite-хранилище логов;
- :mod:`web_ai_assistant.analytics.clustering` — кластеризация запросов
  по эмбеддингам (KMeans / HDBSCAN).
"""

from __future__ import annotations

__all__ = [
    "QueryLog",
    "QueryStore",
    "cluster_queries",
    "ClusterResult",
    "suggest_threshold",
    "ThresholdSuggestion",
]


def __getattr__(name: str):
    if name in ("QueryLog", "QueryStore"):
        from .storage import QueryLog, QueryStore  # noqa: F401

        return locals()[name]
    if name in ("cluster_queries", "ClusterResult"):
        from .clustering import ClusterResult, cluster_queries  # noqa: F401

        return locals()[name]
    if name in ("suggest_threshold", "ThresholdSuggestion"):
        from .threshold import ThresholdSuggestion, suggest_threshold  # noqa: F401

        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

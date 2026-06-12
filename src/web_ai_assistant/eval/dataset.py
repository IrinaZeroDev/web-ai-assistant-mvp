"""Загрузка eval-набора: JSONL или последние N из SQLite-логов."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalItem:
    """Один вопрос для эвалюации."""

    question: str
    ground_truth: str | None = None      # для RAGAS (если есть)
    in_corpus: bool | None = None         # для refusal-accuracy (True = должен ответить)
    meta: dict = field(default_factory=dict)


def load_jsonl(path: str | Path) -> list[EvalItem]:
    """JSONL: ``{"question": ..., "ground_truth"?: ..., "in_corpus"?: bool, ...}``."""
    items: list[EvalItem] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        items.append(
            EvalItem(
                question=row["question"],
                ground_truth=row.get("ground_truth") or row.get("gt"),
                in_corpus=row.get("in_corpus"),
                meta={k: v for k, v in row.items() if k not in ("question", "ground_truth", "gt", "in_corpus")},
            )
        )
    return items


def load_from_db(db_path: str | Path, limit: int = 100, only_unblocked: bool = False) -> list[EvalItem]:
    """Последние N запросов из ``logs/queries.db`` (PII-редактированные)."""
    from ..analytics.storage import QueryStore

    store = QueryStore(db_path)
    try:
        rows = store.recent(limit=limit, include_blocked=not only_unblocked)
    finally:
        store.close()
    items: list[EvalItem] = []
    for r in rows:
        items.append(
            EvalItem(
                question=r["question"],
                in_corpus=(r.get("blocked") is None),
                meta={
                    "blocked_in_log": r.get("blocked"),
                    "ts": r.get("ts"),
                    "id": r.get("id"),
                },
            )
        )
    return items

"""Метрики для A/B-сравнения двух конфигов :class:`RAGAssistant`.

Два слоя:

1. **Fast custom metrics** — считаются из :class:`Answer` без LLM-судьи:

   - ``refusal_rate``     — доля ``blocked != None``;
   - ``mean_max_sim``     — среднее по ``max_sim`` (proxy для retrieval-качества);
   - ``mean_rerank_score``— среднее по top-1 rerank_score (если был реранкер);
   - ``mean_latency_s``   — среднее время ответа (s);
   - ``source_overlap``   — Jaccard-оверлап top-K url'ов между A и B (для пары).

2. **RAGAS (опционально, ``--ragas``)** — faithfulness, answer_relevancy,
   context_recall (если есть ground_truth). Использует LLM-судью.

Структура результата::

    {
      "per_item": [
        {"question": "...", "blocked": None, "max_sim": 0.71, "rerank": 0.92,
         "latency_s": 1.3, "answer": "...", "sources": [...]},
        ...
      ],
      "aggregate": {
        "n": 50,
        "refusal_rate": 0.12,
        "mean_max_sim": 0.68,
        "mean_rerank_score": 0.91,
        "mean_latency_s": 1.2,
      },
    }
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from .dataset import EvalItem


@dataclass
class RunResult:
    """Результат прогона одного конфига по всему набору."""

    name: str
    per_item: list[dict] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)


def run_assistant(assistant, items: list[EvalItem], name: str = "A") -> RunResult:
    """Прогоняет ``assistant.ask`` по списку ``items`` и считает быстрые метрики."""
    per_item: list[dict] = []
    for it in items:
        t0 = time.perf_counter()
        ans = assistant.ask(it.question)
        dt = time.perf_counter() - t0
        rerank_top1 = None
        sources = list(getattr(ans, "sources", []) or [])
        if sources and "rerank_score" in sources[0]:
            rerank_top1 = sources[0]["rerank_score"]
        per_item.append(
            {
                "question": it.question,
                "ground_truth": it.ground_truth,
                "in_corpus": it.in_corpus,
                "answer": ans.answer,
                "blocked": ans.blocked,
                "max_sim": ans.max_sim,
                "rerank_top1": rerank_top1,
                "latency_s": round(dt, 4),
                "sources": [
                    {"id": s.get("id"), "title": s.get("title"), "url": s.get("url")}
                    for s in sources
                ],
            }
        )
    aggregate = _aggregate_fast(per_item)
    return RunResult(name=name, per_item=per_item, aggregate=aggregate)


def _aggregate_fast(per_item: list[dict]) -> dict:
    n = len(per_item)
    if n == 0:
        return {"n": 0}
    refusals = sum(1 for r in per_item if r["blocked"])
    sims = [r["max_sim"] for r in per_item if r["max_sim"] is not None]
    reranks = [r["rerank_top1"] for r in per_item if r["rerank_top1"] is not None]
    lats = [r["latency_s"] for r in per_item if r["latency_s"] is not None]
    agg: dict[str, Any] = {
        "n": n,
        "refusal_rate": round(refusals / n, 4),
        "mean_max_sim": round(mean(sims), 4) if sims else None,
        "mean_rerank_score": round(mean(reranks), 4) if reranks else None,
        "mean_latency_s": round(mean(lats), 4) if lats else None,
    }
    # refusal-accuracy если у датасета размечен in_corpus
    labelled = [r for r in per_item if r["in_corpus"] is not None]
    if labelled:
        tp = sum(1 for r in labelled if r["in_corpus"] and not r["blocked"])
        tn = sum(1 for r in labelled if (not r["in_corpus"]) and r["blocked"])
        agg["refusal_accuracy"] = round((tp + tn) / len(labelled), 4)
    return agg


def source_overlap(a: RunResult, b: RunResult) -> dict:
    """Jaccard по top-K url'ам — насколько A и B retrieve одни и те же документы."""
    overlaps: list[float] = []
    for ra, rb in zip(a.per_item, b.per_item, strict=False):
        urls_a = {s.get("url") for s in ra["sources"] if s.get("url")}
        urls_b = {s.get("url") for s in rb["sources"] if s.get("url")}
        union = urls_a | urls_b
        if not union:
            continue
        overlaps.append(len(urls_a & urls_b) / len(union))
    if not overlaps:
        return {"mean_jaccard": None, "pairs": 0}
    return {"mean_jaccard": round(sum(overlaps) / len(overlaps), 4), "pairs": len(overlaps)}


# ----------------------------- RAGAS (optional) ----------------------------- #


def compute_ragas(result: RunResult) -> dict:
    """RAGAS-метрики (faithfulness, answer_relevancy, context_recall).

    Тяжёлая зависимость — импортируется только при ``--ragas``. Если ``ragas``
    или ``datasets`` не установлены, возвращает ``{}``.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_recall, faithfulness
    except ImportError:  # pragma: no cover
        return {}

    rows = []
    for r in result.per_item:
        if r["blocked"]:
            continue
        rows.append(
            {
                "question": r["question"],
                "answer": r["answer"],
                "contexts": [s.get("title", "") for s in r["sources"]],
                "ground_truth": r.get("ground_truth") or "",
            }
        )
    if not rows:
        return {}
    ds = Dataset.from_list(rows)
    metrics = [faithfulness, answer_relevancy]
    if all(r["ground_truth"] for r in rows):
        metrics.append(context_recall)
    scores = evaluate(ds, metrics=metrics)
    return {k: round(float(v), 4) for k, v in scores.items() if isinstance(v, (int, float))}

"""CLI: ``webai-threshold suggest`` — рекомендует sim_threshold по логам."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="webai-threshold",
        description="Подбор sim_threshold для RAGAssistant по распределению max_sim.",
    )
    p.add_argument(
        "--db",
        default="logs/queries.db",
        help="Путь к SQLite-логу запросов (default: %(default)s).",
    )
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("suggest", help="Рекомендовать порог по логам.")
    s.add_argument(
        "--method",
        choices=["auto", "otsu", "gmm", "percentile"],
        default="auto",
    )
    s.add_argument("--min-sample", type=int, default=30)
    s.add_argument("--percentile", type=float, default=5.0,
                   help="P-й перцентиль для percentile/fallback (default: %(default)s).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not Path(args.db).exists():
        print(f"База логов {args.db} не найдена.")
        return 1

    from ..analytics.storage import QueryStore
    from ..analytics.threshold import suggest_threshold

    store = QueryStore(args.db)
    try:
        dist = store.max_sim_distribution()
    finally:
        store.close()

    sug = suggest_threshold(
        dist["in_corpus"],
        dist["out_of_corpus"],
        method=args.method,
        min_sample=args.min_sample,
        fallback_percentile=args.percentile,
    )

    print(f"База:                 {args.db}")
    print(f"in-corpus запросов:   {sug.in_corpus_count}")
    print(f"out-of-corpus:        {sug.out_of_corpus_count}")
    print(f"Метод:                {sug.method}")
    print(f"Распределение:        {sug.distribution_quality}")
    print(f"Рекомендуемый порог:  {sug.threshold:.4f}")
    print()
    print(sug.rationale)
    if sug.distribution_quality == "too_few_samples":
        return 0
    print()
    print("Применить в админ-эндпоинте:")
    print("  curl -X POST -u admin:<password> -H 'Content-Type: application/json' \\")
    print(f"       -d '{{\"threshold\": {sug.threshold:.4f}}}' \\")
    print("       http://localhost:8000/admin/api/threshold/apply")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

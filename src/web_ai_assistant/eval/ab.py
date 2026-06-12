"""A/B-эвалюация двух конфигов :class:`RAGAssistant` + CLI ``webai-ab``.

Примеры использования:

.. code-block:: bash

    # Декларативные YAML-конфиги + JSONL-вопросы:
    webai-ab \\
        --a-config configs/baseline.yaml \\
        --b-config configs/with_reranker.yaml \\
        --questions data/eval_questions.jsonl \\
        --out-md reports/ab.md --out-html reports/ab.html --out-json reports/ab.json

    # Python-фабрики + последние 200 запросов из лог-БД:
    webai-ab \\
        --a-pyfunc myconfigs:build_baseline \\
        --b-pyfunc myconfigs:build_pilot \\
        --from-db logs/queries.db --db-limit 200 \\
        --ragas

Программный API: :func:`run_ab` принимает 2 уже собранных бота и список
:class:`web_ai_assistant.eval.dataset.EvalItem`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .dataset import EvalItem, load_from_db, load_jsonl
from .factories import build_from_yaml, load_from_pyfunc
from .metrics import RunResult, compute_ragas, run_assistant, source_overlap
from .report import render_markdown, write_reports
from .stats import paired_compare


def run_ab(
    assistant_a,
    assistant_b,
    items: list[EvalItem],
    *,
    name_a: str = "A",
    name_b: str = "B",
    ragas: bool = False,
) -> dict[str, Any]:
    """Прогоняет двух ассистентов по одному набору вопросов и возвращает агрегаты.

    Возвращает словарь со всем содержимым отчёта (``a``, ``b``, ``paired_stats``,
    ``source_overlap``, ``ragas_a``, ``ragas_b``).
    """
    a = run_assistant(assistant_a, items, name=name_a)
    b = run_assistant(assistant_b, items, name=name_b)

    stats: dict[str, dict] = {}
    for metric in ("max_sim", "rerank_top1", "latency_s"):
        vals_a = [r[metric] for r in a.per_item]
        vals_b = [r[metric] for r in b.per_item]
        st = paired_compare(vals_a, vals_b)
        if st.get("n_pairs", 0) > 0:
            stats[metric] = st

    overlap = source_overlap(a, b)

    ragas_a: dict = {}
    ragas_b: dict = {}
    if ragas:
        ragas_a = compute_ragas(a)
        ragas_b = compute_ragas(b)

    return {
        "a": a,
        "b": b,
        "paired_stats": stats,
        "source_overlap": overlap,
        "ragas_a": ragas_a,
        "ragas_b": ragas_b,
    }


# --------------------------------------------------------------------------- #
#                                     CLI                                     #
# --------------------------------------------------------------------------- #


def _resolve_assistant(yaml_path: str | None, pyfunc: str | None):
    if yaml_path and pyfunc:
        raise SystemExit("Укажи только один из: --*-config или --*-pyfunc")
    if yaml_path:
        return build_from_yaml(yaml_path)
    if pyfunc:
        return load_from_pyfunc(pyfunc)
    raise SystemExit("Нужен либо --*-config <yaml>, либо --*-pyfunc <module:func>")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="webai-ab",
        description="A/B-сравнение двух конфигов RAGAssistant: метрики, парные тесты, HTML-отчёт.",
    )
    # source A
    p.add_argument("--a-config", help="YAML/JSON конфиг бота A")
    p.add_argument("--a-pyfunc", help="Python-фабрика бота A: module:func")
    p.add_argument("--a-name", default="A", help="Человеко-читаемое имя варианта A")
    # source B
    p.add_argument("--b-config", help="YAML/JSON конфиг бота B")
    p.add_argument("--b-pyfunc", help="Python-фабрика бота B: module:func")
    p.add_argument("--b-name", default="B", help="Человеко-читаемое имя варианта B")
    # dataset
    p.add_argument("--questions", help="JSONL c вопросами (поля: question, ground_truth?, in_corpus?)")
    p.add_argument("--from-db", help="Путь к queries.db (взять последние N запросов)")
    p.add_argument("--db-limit", type=int, default=100, help="Сколько последних запросов взять из БД")
    # metrics / reports
    p.add_argument("--ragas", action="store_true", help="Дополнительно посчитать RAGAS (медленно, LLM-judge)")
    p.add_argument("--out-md", help="Куда сохранить Markdown-отчёт")
    p.add_argument("--out-json", help="Куда сохранить JSON-отчёт")
    p.add_argument("--out-html", help="Куда сохранить HTML-отчёт с bar-charts")
    p.add_argument("--print-json", action="store_true", help="Печатать JSON в stdout вместо Markdown")
    return p


def _load_items(args: argparse.Namespace) -> list[EvalItem]:
    if args.questions:
        return load_jsonl(args.questions)
    if args.from_db:
        return load_from_db(args.from_db, limit=args.db_limit)
    raise SystemExit("Нужен либо --questions <jsonl>, либо --from-db <queries.db>")


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    items = _load_items(args)
    if not items:
        print("Пустой набор вопросов — нечего сравнивать.", file=sys.stderr)
        return 2

    a_bot = _resolve_assistant(args.a_config, args.a_pyfunc)
    b_bot = _resolve_assistant(args.b_config, args.b_pyfunc)

    result = run_ab(
        a_bot, b_bot, items,
        name_a=args.a_name, name_b=args.b_name, ragas=args.ragas,
    )

    a: RunResult = result["a"]
    b: RunResult = result["b"]

    # Save files (если указаны пути)
    paths = write_reports(
        a, b, result["paired_stats"], result["source_overlap"],
        out_md=args.out_md, out_json=args.out_json, out_html=args.out_html,
        ragas_a=result["ragas_a"], ragas_b=result["ragas_b"],
    )

    if args.print_json:
        from .report import render_json
        data = render_json(
            a, b, result["paired_stats"], result["source_overlap"],
            result["ragas_a"], result["ragas_b"],
        )
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        md = render_markdown(
            a, b, result["paired_stats"], result["source_overlap"],
            result["ragas_a"], result["ragas_b"],
        )
        print(md)

    if paths:
        print("\nСохранено:", ", ".join(f"{k}={v}" for k, v in paths.items()), file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


def _save_report_files(path_prefix: str | Path) -> None:  # pragma: no cover
    # Утилита оставлена для возможной интеграции — основной API в write_reports.
    raise NotImplementedError

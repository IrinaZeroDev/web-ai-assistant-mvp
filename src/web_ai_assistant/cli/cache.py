"""CLI: ``webai-cache stats | clear`` — инспекция дискового кэша эмбеддингов.

Запуск::

    webai-cache stats                       # сводка по умолчанию cache/embeddings.db
    webai-cache stats --path /var/cache/embs.db
    webai-cache clear --model Embeddings    # удалить namespace
    webai-cache clear --all                 # очистить полностью
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="webai-cache",
        description="Инспектор дискового кэша эмбеддингов web-ai-assistant.",
    )
    p.add_argument(
        "--path", default="cache/embeddings.db",
        help="Путь к SQLite-кэшу (default: %(default)s)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("stats", help="Показать сводку: число записей по моделям.")

    pc = sub.add_parser("clear", help="Удалить записи из кэша.")
    g = pc.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", help="Удалить только namespace конкретной модели.")
    g.add_argument("--all", action="store_true", help="Очистить кэш полностью.")

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    path = Path(args.path)

    if not path.exists() and args.command == "stats":
        print(f"Кэш {path} не существует — пока пусто.")
        return 0

    from ..embeddings.cache import EmbeddingCache

    cache = EmbeddingCache(path)
    try:
        if args.command == "stats":
            total = cache.count()
            by_model = cache.models()
            print(f"Кэш: {path}")
            print(f"Всего записей: {total}")
            if not by_model:
                print("  (пусто)")
            for model, n in sorted(by_model.items(), key=lambda x: -x[1]):
                print(f"  {model:30s} {n:>8d}")
            return 0

        if args.command == "clear":
            if args.all:
                n = cache.clear()
                print(f"Удалено: {n} записей (полная очистка).")
            else:
                n = cache.clear(model=args.model)
                print(f"Удалено: {n} записей в namespace {args.model!r}.")
            return 0
    finally:
        cache.close()

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

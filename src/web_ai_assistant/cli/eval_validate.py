"""CLI ``webai-eval-validate`` — проверка eval-датасета.

Запуск::

    webai-eval-validate data/eval/questions_v1.jsonl

Делает:

1. Парсит JSONL построчно (с указанием номера строки в ошибках).
2. Валидирует каждый объект против ``data/eval/schema.json``.
3. Сверяет ``category`` и (если задано) ``in_corpus`` на согласованность.
4. Проверяет уникальность ``id``.
5. Печатает сводку по категориям + предупреждения о дисбалансе
   относительно целевых пропорций (60/20/10/10).

Возврат: ``0`` — всё ок; ``1`` — есть ошибки/предупреждения; ``2`` — ошибка чтения.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# Категории, для которых in_corpus должен быть True (если указан).
CATEGORIES_IN_CORPUS_TRUE = {"in_corpus"}
CATEGORIES_IN_CORPUS_FALSE = {"off_topic", "red_zone", "escalation"}
ALL_CATEGORIES = CATEGORIES_IN_CORPUS_TRUE | CATEGORIES_IN_CORPUS_FALSE

# Целевые пропорции (для предупреждений). Допустимое отклонение — ±5 п.п.
TARGET_PROPORTIONS = {
    "in_corpus": 0.60,
    "off_topic": 0.20,
    "red_zone": 0.10,
    "escalation": 0.10,
}
PROPORTION_TOLERANCE = 0.05


def _default_schema_path() -> Path:
    """Ищет schema.json рядом с датасетом или в data/eval/."""
    here = Path(__file__).resolve()
    repo_root = here.parents[3]  # src/web_ai_assistant/cli/ → repo
    return repo_root / "data" / "eval" / "schema.json"


def validate_dataset(
    jsonl_path: str | Path,
    schema_path: str | Path | None = None,
) -> tuple[list[str], list[str], Counter]:
    """Возвращает ``(errors, warnings, category_counts)``."""
    errors: list[str] = []
    warnings: list[str] = []
    counts: Counter = Counter()
    seen_ids: set[str] = set()

    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        warnings.append(
            "jsonschema не установлен — JSON-Schema валидация пропущена. "
            "pip install 'web-ai-assistant[eval-ab]' (включает jsonschema)."
        )
        validator = None
    else:
        schema_p = Path(schema_path) if schema_path else _default_schema_path()
        if not schema_p.is_file():
            warnings.append(f"Schema не найдена: {schema_p} — пропускаем валидацию схемы.")
            validator = None
        else:
            schema = json.loads(schema_p.read_text(encoding="utf-8"))
            validator = Draft202012Validator(schema)

    p = Path(jsonl_path)
    if not p.is_file():
        return [f"Файл не найден: {p}"], warnings, counts

    for lineno, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"L{lineno}: невалидный JSON — {exc.msg}")
            continue

        # 1. Schema check.
        if validator is not None:
            for err in validator.iter_errors(obj):
                path = "/".join(str(p) for p in err.absolute_path) or "<root>"
                errors.append(f"L{lineno} [{path}]: {err.message}")

        # 2. Unique id.
        item_id = obj.get("id")
        if isinstance(item_id, str):
            if item_id in seen_ids:
                errors.append(f"L{lineno}: дубликат id '{item_id}'")
            seen_ids.add(item_id)

        # 3. Category check (если поле есть).
        cat = obj.get("category")
        if cat in ALL_CATEGORIES:
            counts[cat] += 1
            # Сверка in_corpus с category.
            if "in_corpus" in obj:
                expected = cat in CATEGORIES_IN_CORPUS_TRUE
                if bool(obj["in_corpus"]) != expected:
                    errors.append(
                        f"L{lineno}: category='{cat}' требует in_corpus={expected}, "
                        f"но указано {obj['in_corpus']}"
                    )

    # 4. Пропорции.
    total = sum(counts.values())
    if total > 0:
        for cat, target in TARGET_PROPORTIONS.items():
            actual = counts.get(cat, 0) / total
            if abs(actual - target) > PROPORTION_TOLERANCE:
                warnings.append(
                    f"Категория '{cat}': {counts.get(cat, 0)}/{total} = "
                    f"{actual:.1%} (цель {target:.0%} ±{PROPORTION_TOLERANCE:.0%})"
                )

    return errors, warnings, counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="webai-eval-validate",
        description="Валидация eval-датасета (JSONL) против data/eval/schema.json.",
    )
    parser.add_argument("path", help="Путь к JSONL-файлу с вопросами.")
    parser.add_argument("--schema", help="Путь к JSON-Schema (по умолчанию data/eval/schema.json).")
    parser.add_argument(
        "--strict", action="store_true",
        help="Считать предупреждения ошибками (exit code 1 даже без errors).",
    )
    args = parser.parse_args(argv)

    errors, warnings, counts = validate_dataset(args.path, args.schema)

    total = sum(counts.values())
    print(f"Проверено: {args.path}")
    print(f"Всего пунктов с известной категорией: {total}")
    if total:
        for cat in ("in_corpus", "off_topic", "red_zone", "escalation"):
            n = counts.get(cat, 0)
            pct = n / total * 100 if total else 0.0
            print(f"  {cat:12s} {n:3d}  ({pct:5.1f}%)")

    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  ⚠ {w}")

    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"  ✗ {e}")
        return 1

    if args.strict and warnings:
        return 1

    print("\nOK — датасет валиден.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Тесты CLI ``webai-eval-validate`` и качества черновика ``questions_v1.jsonl``.

Не запускают LLM — только статическая проверка JSON-Schema, согласованности
полей и триггеров guards (`is_red_zone`, `is_escalation`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from web_ai_assistant.cli.eval_validate import (
    CATEGORIES_IN_CORPUS_FALSE,
    CATEGORIES_IN_CORPUS_TRUE,
    main,
    validate_dataset,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_V1 = REPO_ROOT / "data" / "eval" / "questions_v1.jsonl"
SCHEMA = REPO_ROOT / "data" / "eval" / "schema.json"


# --------------------------------------------------------------------------- #
#                      Базовая валидация черновика v1                          #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not DATASET_V1.exists(), reason="dataset v1 ещё не создан")
def test_v1_dataset_passes_schema() -> None:
    errors, _warnings, counts = validate_dataset(DATASET_V1, SCHEMA)
    assert not errors, "Errors в черновике v1:\n" + "\n".join(errors)
    assert sum(counts.values()) == 50


@pytest.mark.skipif(not DATASET_V1.exists(), reason="dataset v1 ещё не создан")
def test_v1_dataset_proportions() -> None:
    """Черновик v1 идёт точно по целевым пропорциям 60/20/10/10."""
    _e, _w, counts = validate_dataset(DATASET_V1, SCHEMA)
    assert counts["in_corpus"] == 30
    assert counts["off_topic"] == 10
    assert counts["red_zone"] == 5
    assert counts["escalation"] == 5


@pytest.mark.skipif(not DATASET_V1.exists(), reason="dataset v1 ещё не создан")
def test_v1_dataset_unique_ids() -> None:
    ids = []
    with DATASET_V1.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.append(json.loads(line)["id"])
    assert len(ids) == len(set(ids)), "Дубликаты id"


@pytest.mark.skipif(not DATASET_V1.exists(), reason="dataset v1 ещё не создан")
def test_v1_red_zone_questions_trigger_guard() -> None:
    """Каждый red_zone-вопрос должен ловиться is_red_zone(), и наоборот."""
    from web_ai_assistant.guards import is_red_zone

    with DATASET_V1.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            q, c = o["question"], o["category"]
            if c == "red_zone":
                assert is_red_zone(q), f"{o['id']}: red_zone не сработал на: {q}"
            elif c in ("in_corpus", "off_topic"):
                assert not is_red_zone(q), f"{o['id']}: ложное срабатывание red_zone на: {q}"


@pytest.mark.skipif(not DATASET_V1.exists(), reason="dataset v1 ещё не создан")
def test_v1_escalation_questions_trigger_guard() -> None:
    from web_ai_assistant.guards import is_escalation

    with DATASET_V1.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            q, c = o["question"], o["category"]
            if c == "escalation":
                assert is_escalation(q), f"{o['id']}: escalation не сработал на: {q}"
            elif c in ("in_corpus", "off_topic"):
                assert not is_escalation(q), f"{o['id']}: ложное срабатывание escalation на: {q}"


# --------------------------------------------------------------------------- #
#                          Валидатор: ошибки и WARNINGS                        #
# --------------------------------------------------------------------------- #


def _write_jsonl(path: Path, items: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(it, ensure_ascii=False) for it in items),
        encoding="utf-8",
    )


def test_validator_catches_missing_required_field(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    _write_jsonl(p, [{"question": "no id and no category"}])
    errors, _w, _c = validate_dataset(p, SCHEMA)
    assert errors, "ожидали ошибки про отсутствующие required-поля"
    joined = " ".join(errors)
    assert "id" in joined or "category" in joined


def test_validator_catches_invalid_category(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    _write_jsonl(p, [{"id": "x_1", "question": "Что такое HTML?", "category": "bogus"}])
    errors, _w, _c = validate_dataset(p, SCHEMA)
    assert any("bogus" in e or "category" in e for e in errors)


def test_validator_catches_mismatched_in_corpus(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    _write_jsonl(
        p,
        [
            {"id": "x_1", "question": "Что такое HTML?", "category": "in_corpus", "in_corpus": False},
        ],
    )
    errors, _w, _c = validate_dataset(p, SCHEMA)
    assert any("in_corpus" in e for e in errors)


def test_validator_catches_duplicate_ids(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    _write_jsonl(
        p,
        [
            {"id": "x_1", "question": "Что такое HTML?", "category": "in_corpus"},
            {"id": "x_1", "question": "Чем CSS отличается от JS?", "category": "in_corpus"},
        ],
    )
    errors, _w, _c = validate_dataset(p, SCHEMA)
    assert any("дубликат" in e.lower() or "x_1" in e for e in errors)


def test_validator_catches_invalid_json(tmp_path: Path) -> None:
    p = tmp_path / "broken.jsonl"
    p.write_text('{"id":"a", "question":"ok", "category":"in_corpus"}\n{bad json}\n', encoding="utf-8")
    errors, _w, _c = validate_dataset(p, SCHEMA)
    assert any("L2" in e for e in errors)


def test_validator_warns_on_proportion_imbalance(tmp_path: Path) -> None:
    p = tmp_path / "skewed.jsonl"
    items = [
        {"id": f"x_{i}", "question": "q?", "category": "in_corpus"} for i in range(10)
    ]
    _write_jsonl(p, items)
    _e, warnings, _c = validate_dataset(p, SCHEMA)
    # 100% in_corpus → дисбаланс по off_topic / red_zone / escalation.
    assert any("off_topic" in w for w in warnings)
    assert any("red_zone" in w for w in warnings)


def test_categories_constants_partition() -> None:
    """In_corpus и нет — должны быть непересекающимися."""
    assert CATEGORIES_IN_CORPUS_TRUE.isdisjoint(CATEGORIES_IN_CORPUS_FALSE)
    assert CATEGORIES_IN_CORPUS_TRUE | CATEGORIES_IN_CORPUS_FALSE == {
        "in_corpus", "off_topic", "red_zone", "escalation",
    }


def test_cli_main_exit_codes(tmp_path: Path, capsys) -> None:
    good = tmp_path / "ok.jsonl"
    _write_jsonl(
        good,
        [
            {"id": f"x_{i}", "question": "Что такое HTML?", "category": "in_corpus"} for i in range(6)
        ] + [
            {"id": f"y_{i}", "question": "Cтолица Австралии?", "category": "off_topic"} for i in range(2)
        ] + [
            {"id": "rz_1", "question": "Поставь мне оценку.", "category": "red_zone"},
            {"id": "esc_1", "question": "Объясни мне новую тему: X.", "category": "escalation"},
        ],
    )
    rc = main([str(good), "--schema", str(SCHEMA)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "OK" in out

    bad = tmp_path / "bad.jsonl"
    _write_jsonl(bad, [{"question": "no id"}])
    rc = main([str(bad), "--schema", str(SCHEMA)])
    assert rc == 1


def test_cli_strict_warnings_fail(tmp_path: Path) -> None:
    """В --strict даже предупреждения о пропорциях → exit 1."""
    p = tmp_path / "skewed.jsonl"
    _write_jsonl(p, [{"id": f"x_{i}", "question": "q?", "category": "in_corpus"} for i in range(5)])
    rc = main([str(p), "--schema", str(SCHEMA), "--strict"])
    assert rc == 1

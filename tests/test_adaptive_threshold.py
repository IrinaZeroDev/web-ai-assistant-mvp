"""Тесты adaptive sim_threshold."""

from __future__ import annotations

import io
import random
from pathlib import Path
from unittest.mock import patch

import pytest

from web_ai_assistant.analytics.storage import QueryLog, QueryStore
from web_ai_assistant.analytics.threshold import (
    ThresholdSuggestion,
    _otsu_threshold,
    _percentile_threshold,
    suggest_threshold,
)
from web_ai_assistant.cli.threshold import main as threshold_cli

# ---------------------------------------------------------------------------
# Помощник: бимодальная и унимодальная выборки
# ---------------------------------------------------------------------------


def _bimodal_samples(seed: int = 0) -> tuple[list[float], list[float]]:
    """in-corpus ~ N(0.85, 0.05), out-of-corpus ~ N(0.30, 0.08)."""
    rng = random.Random(seed)
    in_c = [max(0.0, min(1.0, rng.gauss(0.85, 0.05))) for _ in range(120)]
    out_c = [max(0.0, min(1.0, rng.gauss(0.30, 0.08))) for _ in range(40)]
    return in_c, out_c


def _unimodal_samples(seed: int = 0) -> tuple[list[float], list[float]]:
    """Только один пик ~ N(0.7, 0.1) — нет чёткого разделения."""
    rng = random.Random(seed)
    in_c = [max(0.0, min(1.0, rng.gauss(0.7, 0.1))) for _ in range(120)]
    return in_c, []


# ---------------------------------------------------------------------------
# Юнит-тесты алгоритмов
# ---------------------------------------------------------------------------


def test_otsu_finds_threshold_between_modes():
    in_c, out_c = _bimodal_samples()
    t, bc = _otsu_threshold(in_c + out_c)
    # порог где-то между 0.3 и 0.85
    assert 0.4 < t < 0.8
    # Sarle's BC > 5/9 — бимодальность обнаружена
    assert bc > 0.555


def test_otsu_on_unimodal_low_ratio():
    in_c, _ = _unimodal_samples()
    _t, bc = _otsu_threshold(in_c)
    assert bc < 0.555  # унимодально


def test_otsu_empty_returns_zero():
    t, ratio = _otsu_threshold([])
    assert t == 0.0
    assert ratio == 0.0


def test_otsu_constant_returns_value():
    t, ratio = _otsu_threshold([0.5, 0.5, 0.5])
    assert t == 0.5
    assert ratio == 0.0


def test_percentile_threshold():
    vals = [v / 100.0 for v in range(100)]
    # P5 ≈ 0.05
    p5 = _percentile_threshold(vals, 5.0)
    assert 0.04 <= p5 <= 0.06


# ---------------------------------------------------------------------------
# suggest_threshold: end-to-end
# ---------------------------------------------------------------------------


def test_suggest_auto_picks_otsu_for_bimodal():
    in_c, out_c = _bimodal_samples()
    sug = suggest_threshold(in_c, out_c, method="auto")
    assert sug.method == "otsu"
    assert sug.distribution_quality == "bimodal"
    assert 0.4 < sug.threshold < 0.8
    assert sug.sample_size == len(in_c) + len(out_c)
    assert sug.in_corpus_count == len(in_c)
    assert sug.out_of_corpus_count == len(out_c)


def test_suggest_auto_falls_back_to_percentile_on_unimodal():
    in_c, _ = _unimodal_samples()
    sug = suggest_threshold(in_c, method="auto", fallback_percentile=5.0)
    assert "percentile" in sug.method
    assert sug.distribution_quality == "unimodal"


def test_suggest_too_few_samples_returns_default():
    sug = suggest_threshold([0.9, 0.8], min_sample=30)
    assert sug.distribution_quality == "too_few_samples"
    assert "default-fallback" in sug.method


def test_suggest_explicit_percentile():
    in_c, _ = _unimodal_samples()
    sug = suggest_threshold(in_c, method="percentile", fallback_percentile=10.0)
    assert "P10" in sug.method
    # P10 на N(0.7, 0.1) ≈ 0.57
    assert 0.45 < sug.threshold < 0.65


def test_suggest_gmm_requires_sklearn():
    """method='gmm' должен работать на нашей dev-зависимости."""
    in_c, out_c = _bimodal_samples()
    sug = suggest_threshold(in_c, out_c, method="gmm")
    assert sug.method == "gmm"
    assert 0.4 < sug.threshold < 0.8


def test_suggestion_as_dict_is_serializable():
    sug = ThresholdSuggestion(
        threshold=0.5,
        method="otsu",
        sample_size=10,
        in_corpus_count=8,
        out_of_corpus_count=2,
        distribution_quality="bimodal",
        rationale="ok",
        histogram=[{"bin": 0.5, "count": 1}],
    )
    d = sug.as_dict()
    assert d["threshold"] == 0.5
    assert d["method"] == "otsu"
    assert d["histogram"][0] == {"bin": 0.5, "count": 1}


def test_suggestion_histogram_built():
    in_c, out_c = _bimodal_samples()
    sug = suggest_threshold(in_c, out_c)
    assert len(sug.histogram) > 5
    assert sum(h["count"] for h in sug.histogram) == len(in_c) + len(out_c)


# ---------------------------------------------------------------------------
# QueryStore.max_sim_distribution
# ---------------------------------------------------------------------------


def test_max_sim_distribution_groups_correctly():
    store = QueryStore(":memory:")
    store.insert(QueryLog(question="ok1", max_sim=0.85))
    store.insert(QueryLog(question="ok2", max_sim=0.91))
    store.insert(QueryLog(question="out1", max_sim=0.20, blocked="out_of_corpus"))
    store.insert(QueryLog(question="red", blocked="red_zone"))  # max_sim=None
    store.insert(QueryLog(question="esc", blocked="escalation"))
    dist = store.max_sim_distribution()
    assert sorted(dist["in_corpus"]) == [0.85, 0.91]
    assert dist["out_of_corpus"] == [0.20]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _capture_cli(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        rc = threshold_cli(argv)
    return rc, buf.getvalue()


def test_cli_db_missing(tmp_path: Path):
    rc, out = _capture_cli(["--db", str(tmp_path / "nope.db"), "suggest"])
    assert rc == 1
    assert "не найдена" in out


def test_cli_too_few_samples(tmp_path: Path):
    db = tmp_path / "queries.db"
    store = QueryStore(db)
    store.insert(QueryLog(question="ok", max_sim=0.9))
    store.close()
    rc, out = _capture_cli(["--db", str(db), "suggest"])
    assert rc == 0
    assert "1" in out
    assert "Рекомендуемый порог" in out


def test_cli_full_pipeline_with_logs(tmp_path: Path):
    db = tmp_path / "queries.db"
    store = QueryStore(db)
    in_c, out_c = _bimodal_samples()
    for v in in_c:
        store.insert(QueryLog(question="q", max_sim=v))
    for v in out_c:
        store.insert(QueryLog(question="q", max_sim=v, blocked="out_of_corpus"))
    store.close()
    rc, out = _capture_cli(["--db", str(db), "suggest", "--method", "otsu"])
    assert rc == 0
    assert "otsu" in out
    assert "bimodal" in out
    assert "Применить в админ-эндпоинте" in out


# ---------------------------------------------------------------------------
# Admin endpoint: suggest + apply
# ---------------------------------------------------------------------------


def test_admin_threshold_endpoints():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from web_ai_assistant.rag import Answer
    from web_ai_assistant.server import create_app

    class FakeAssistant:
        top_k = 4
        sim_threshold = 0.55

        def ask(self, q):
            return Answer(answer="ok", sources=[], max_sim=0.9)

    store = QueryStore(":memory:")
    in_c, out_c = _bimodal_samples()
    for v in in_c:
        store.insert(QueryLog(question="q", max_sim=v))
    for v in out_c:
        store.insert(QueryLog(question="q", max_sim=v, blocked="out_of_corpus"))

    app = create_app(assistant_factory=lambda: FakeAssistant(), query_store=store)
    with TestClient(app) as c:
        r = c.get("/admin/api/threshold/suggest")
        assert r.status_code == 200
        body = r.json()
        assert body["current"] == 0.55
        assert body["suggestion"]["method"] == "otsu"
        new_t = body["suggestion"]["threshold"]
        assert 0.4 < new_t < 0.8

        # применим
        r2 = c.post("/admin/api/threshold/apply", json={"threshold": new_t})
        assert r2.status_code == 200, r2.text
        body2 = r2.json()
        assert body2["previous"] == pytest.approx(0.55)
        assert body2["applied"] == pytest.approx(new_t)

        # значение действительно изменилось
        r3 = c.get("/admin/api/threshold/suggest")
        assert r3.json()["current"] == pytest.approx(new_t)

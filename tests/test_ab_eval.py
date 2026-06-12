"""End-to-end тесты A/B-инструмента: датасет → метрики → отчёт → CLI.

Используем фейковые ``RAGAssistant``-подобные объекты, чтобы не тянуть
тяжёлые ML-зависимости (transformers / chromadb / GigaChat SDK).
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from web_ai_assistant.eval.dataset import EvalItem, load_from_db, load_jsonl
from web_ai_assistant.eval.metrics import RunResult, run_assistant, source_overlap
from web_ai_assistant.eval.report import render_html, render_json, render_markdown, write_reports
from web_ai_assistant.eval.stats import paired_compare

# --------------------------------------------------------------------------- #
#                              Fake assistants                                #
# --------------------------------------------------------------------------- #


@dataclass
class _FakeAnswer:
    answer: str
    sources: list = field(default_factory=list)
    max_sim: float | None = None
    blocked: str | None = None


class FakeAssistant:
    """Имитирует ``RAGAssistant.ask`` с детерминированными метриками."""

    def __init__(self, *, base_sim: float, base_rerank: float | None, refuse_words=("noise",)):
        self.base_sim = base_sim
        self.base_rerank = base_rerank
        self.refuse_words = refuse_words

    def ask(self, question: str) -> _FakeAnswer:
        if any(w in question.lower() for w in self.refuse_words):
            return _FakeAnswer(
                answer="Я не нашёл этого в материалах курса.",
                sources=[],
                max_sim=0.1,
                blocked="out_of_corpus",
            )
        # сим зависит от длины (детерминизм без рандома)
        bump = (len(question) % 5) / 100
        sources = [
            {"id": 1, "title": "Doc1", "url": "https://ex.dev/a", "sim": self.base_sim + bump},
            {"id": 2, "title": "Doc2", "url": "https://ex.dev/b", "sim": self.base_sim},
        ]
        if self.base_rerank is not None:
            sources[0]["rerank_score"] = self.base_rerank
            sources[1]["rerank_score"] = self.base_rerank - 0.05
        return _FakeAnswer(
            answer=f"Ответ на: {question}",
            sources=sources,
            max_sim=self.base_sim + bump,
            blocked=None,
        )


# --------------------------------------------------------------------------- #
#                                  dataset                                    #
# --------------------------------------------------------------------------- #


def test_load_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "q.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps({"question": "Что такое HTML?", "ground_truth": "разметка", "in_corpus": True}),
                json.dumps({"question": "noise question", "in_corpus": False, "topic": "off"}),
                "   ",  # пустая строка — должна игнорироваться
            ]
        ),
        encoding="utf-8",
    )
    items = load_jsonl(p)
    assert len(items) == 2
    assert items[0].question == "Что такое HTML?"
    assert items[0].ground_truth == "разметка"
    assert items[0].in_corpus is True
    assert items[1].in_corpus is False
    assert items[1].meta == {"topic": "off"}


def test_load_from_db(tmp_path: Path) -> None:
    from web_ai_assistant.analytics.storage import QueryLog, QueryStore

    db = tmp_path / "queries.db"
    store = QueryStore(db)
    store.insert(QueryLog(question="Что такое flexbox?", blocked=None, max_sim=0.71))
    store.insert(QueryLog(question="noise", blocked="out_of_corpus", max_sim=0.1))
    store.close()
    items = load_from_db(db, limit=10)
    assert len(items) == 2
    # in_corpus = (blocked is None)
    by_q = {it.question: it for it in items}
    assert by_q["Что такое flexbox?"].in_corpus is True
    assert by_q["noise"].in_corpus is False


# --------------------------------------------------------------------------- #
#                                  metrics                                    #
# --------------------------------------------------------------------------- #


def _items(in_corpus_for_noise: bool = False) -> list[EvalItem]:
    return [
        EvalItem(question="Что такое HTML?", in_corpus=True),
        EvalItem(question="Что такое CSS?", in_corpus=True),
        EvalItem(question="noise question", in_corpus=in_corpus_for_noise),
    ]


def test_run_assistant_aggregate() -> None:
    bot = FakeAssistant(base_sim=0.7, base_rerank=0.9)
    res = run_assistant(bot, _items(), name="A")
    assert isinstance(res, RunResult)
    assert res.aggregate["n"] == 3
    assert res.aggregate["refusal_rate"] == pytest.approx(1 / 3, rel=1e-3)
    assert 0.0 <= res.aggregate["mean_max_sim"] <= 1.0
    assert res.aggregate["mean_rerank_score"] is not None
    # refusal_accuracy при правильной разметке (noise → не в корпусе → должен отказать → корректно)
    assert res.aggregate["refusal_accuracy"] == 1.0


def test_run_assistant_no_reranker() -> None:
    bot = FakeAssistant(base_sim=0.6, base_rerank=None)
    res = run_assistant(bot, _items(), name="B")
    assert res.aggregate["mean_rerank_score"] is None


def test_source_overlap() -> None:
    items = _items()
    a = run_assistant(FakeAssistant(base_sim=0.7, base_rerank=0.9), items, name="A")
    b = run_assistant(FakeAssistant(base_sim=0.6, base_rerank=None), items, name="B")
    overlap = source_overlap(a, b)
    # Оба фейка отдают те же 2 url → jaccard = 1.0 на не-блокированных вопросах
    assert overlap["pairs"] >= 1
    assert overlap["mean_jaccard"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
#                                   stats                                     #
# --------------------------------------------------------------------------- #


def test_paired_compare_basic() -> None:
    a = [0.7, 0.8, 0.9, 0.75, 0.85]
    b = [0.6, 0.7, 0.8, 0.65, 0.75]
    st = paired_compare(a, b)
    assert st["n_pairs"] == 5
    assert st["mean_diff"] == pytest.approx(0.1, rel=1e-3)
    # scipy установлен → p-value присутствуют
    assert "t_pvalue" in st
    assert "wilcoxon_pvalue" in st
    # A систематически > B, t_pvalue должен быть маленький
    assert st["t_pvalue"] < 0.05


def test_paired_compare_handles_none() -> None:
    a = [0.7, None, 0.9]
    b = [0.6, 0.7, None]
    st = paired_compare(a, b)
    assert st["n_pairs"] == 1  # только одна валидная пара


def test_paired_compare_empty() -> None:
    assert paired_compare([], []) == {"n_pairs": 0}


# --------------------------------------------------------------------------- #
#                                   report                                    #
# --------------------------------------------------------------------------- #


def _full_runs():
    items = _items()
    a = run_assistant(FakeAssistant(base_sim=0.75, base_rerank=0.92), items, name="bge")
    b = run_assistant(FakeAssistant(base_sim=0.60, base_rerank=None), items, name="baseline")
    stats = {
        "max_sim": paired_compare(
            [r["max_sim"] for r in a.per_item], [r["max_sim"] for r in b.per_item]
        ),
    }
    overlap = source_overlap(a, b)
    return a, b, stats, overlap


def test_render_markdown_has_metrics() -> None:
    a, b, stats, overlap = _full_runs()
    md = render_markdown(a, b, stats, overlap)
    assert "# A/B eval:" in md
    assert "**bge**" in md and "**baseline**" in md
    assert "`refusal_rate`" in md
    assert "`max_sim`" in md  # парный тест по этой метрике
    assert "Mean Jaccard" in md


def test_render_json_structure() -> None:
    a, b, stats, overlap = _full_runs()
    data = render_json(a, b, stats, overlap, None, None)
    assert set(data.keys()) >= {"a", "b", "paired_stats", "source_overlap", "per_item"}
    assert data["a"]["name"] == "bge"
    assert len(data["per_item"]) == len(a.per_item)


def test_render_html_has_charts() -> None:
    a, b, stats, overlap = _full_runs()
    h = render_html(a, b, stats, overlap, {"faithfulness": 0.91}, {"faithfulness": 0.78})
    assert "<!doctype html>" in h
    assert "bar-a" in h and "bar-b" in h
    assert "RAGAS" in h
    assert "faithfulness" in h


def test_write_reports(tmp_path: Path) -> None:
    a, b, stats, overlap = _full_runs()
    paths = write_reports(
        a, b, stats, overlap,
        out_md=tmp_path / "r.md",
        out_json=tmp_path / "r.json",
        out_html=tmp_path / "r.html",
    )
    assert set(paths.keys()) == {"md", "json", "html"}
    assert (tmp_path / "r.md").exists()
    data = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert data["a"]["name"] == "bge"


# --------------------------------------------------------------------------- #
#                                  factories                                  #
# --------------------------------------------------------------------------- #


def test_load_from_pyfunc(tmp_path: Path, monkeypatch) -> None:
    mod_dir = tmp_path / "fab_pkg"
    mod_dir.mkdir()
    (mod_dir / "__init__.py").write_text("")
    (mod_dir / "configs.py").write_text(
        textwrap.dedent(
            """
            class Bot:
                def ask(self, q): return type("A", (), {"answer": q, "sources": [], "max_sim": 0.5, "blocked": None})()

            def build(): return Bot()
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    from web_ai_assistant.eval.factories import load_from_pyfunc

    bot = load_from_pyfunc("fab_pkg.configs:build")
    assert bot.ask("hi").answer == "hi"


def test_load_yaml_config(tmp_path: Path) -> None:
    from web_ai_assistant.eval.factories import load_yaml_config

    p = tmp_path / "cfg.yaml"
    p.write_text("name: foo\nrag:\n  sim_threshold: 0.55\n", encoding="utf-8")
    cfg = load_yaml_config(p)
    assert cfg["name"] == "foo"
    assert cfg["rag"]["sim_threshold"] == 0.55

    # JSON-вариант
    pj = tmp_path / "cfg.json"
    pj.write_text(json.dumps({"name": "bar"}), encoding="utf-8")
    assert load_yaml_config(pj) == {"name": "bar"}


# --------------------------------------------------------------------------- #
#                              run_ab + CLI e2e                                #
# --------------------------------------------------------------------------- #


def test_run_ab_end_to_end() -> None:
    from web_ai_assistant.eval.ab import run_ab

    bot_a = FakeAssistant(base_sim=0.80, base_rerank=0.92)
    bot_b = FakeAssistant(base_sim=0.60, base_rerank=None)
    out = run_ab(bot_a, bot_b, _items(), name_a="reranker", name_b="baseline")
    assert out["a"].name == "reranker"
    assert "max_sim" in out["paired_stats"]
    assert out["source_overlap"]["pairs"] >= 1


def test_cli_with_pyfunc_and_jsonl(tmp_path: Path, monkeypatch, capsys) -> None:
    # Готовим Python-фабрики
    mod_dir = tmp_path / "fab2"
    mod_dir.mkdir()
    (mod_dir / "__init__.py").write_text("")
    (mod_dir / "cfg.py").write_text(
        textwrap.dedent(
            """
            from dataclasses import dataclass, field

            @dataclass
            class _A:
                answer: str
                sources: list = field(default_factory=list)
                max_sim: float | None = 0.7
                blocked: str | None = None

            class _Bot:
                def __init__(self, sim): self.sim = sim
                def ask(self, q):
                    return _A(answer=q,
                             sources=[{"id":1,"title":"T","url":"u","sim":self.sim}],
                             max_sim=self.sim, blocked=None)

            def build_a(): return _Bot(0.8)
            def build_b(): return _Bot(0.6)
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    qf = tmp_path / "q.jsonl"
    qf.write_text(
        "\n".join(
            json.dumps({"question": f"Q{i}", "in_corpus": True}) for i in range(4)
        ),
        encoding="utf-8",
    )
    out_md = tmp_path / "r.md"
    out_json = tmp_path / "r.json"
    out_html = tmp_path / "r.html"

    from web_ai_assistant.eval.ab import main

    rc = main(
        [
            "--a-pyfunc", "fab2.cfg:build_a",
            "--b-pyfunc", "fab2.cfg:build_b",
            "--a-name", "AAA",
            "--b-name", "BBB",
            "--questions", str(qf),
            "--out-md", str(out_md),
            "--out-json", str(out_json),
            "--out-html", str(out_html),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "AAA" in captured.out and "BBB" in captured.out
    assert out_md.exists() and out_json.exists() and out_html.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["a"]["name"] == "AAA"
    assert data["b"]["name"] == "BBB"


def test_cli_print_json(tmp_path: Path, monkeypatch, capsys) -> None:
    mod_dir = tmp_path / "fab3"
    mod_dir.mkdir()
    (mod_dir / "__init__.py").write_text("")
    (mod_dir / "cfg.py").write_text(
        textwrap.dedent(
            """
            class _A:
                def __init__(self, q, sim): self.answer=q; self.sources=[]; self.max_sim=sim; self.blocked=None
            class _Bot:
                def __init__(self, sim): self.sim = sim
                def ask(self, q): return _A(q, self.sim)
            def build_a(): return _Bot(0.8)
            def build_b(): return _Bot(0.6)
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    qf = tmp_path / "q.jsonl"
    qf.write_text(json.dumps({"question": "Q1"}) + "\n", encoding="utf-8")

    from web_ai_assistant.eval.ab import main

    rc = main(
        [
            "--a-pyfunc", "fab3.cfg:build_a",
            "--b-pyfunc", "fab3.cfg:build_b",
            "--questions", str(qf),
            "--print-json",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["a"]["aggregate"]["n"] == 1

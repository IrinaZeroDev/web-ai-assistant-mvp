"""Тесты cross-encoder реранкера + интеграция с RAGAssistant."""

from __future__ import annotations

import pytest

from web_ai_assistant.rag import RAGAssistant

# ---------------------------------------------------------------------------
# Фейковые компоненты
# ---------------------------------------------------------------------------


class FakeReranker:
    """Реранкер с предсказуемыми скорами — для проверки порядка."""

    def __init__(self, scores: list[float] | None = None):
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, candidates):
        self.calls.append((query, list(candidates)))
        if self.scores is None:
            # длинные кандидаты — релевантнее (для тестов наглядно)
            return [float(len(c)) for c in candidates]
        return list(self.scores[: len(candidates)])


class FakeIndex:
    """Возвращает заранее заданный набор."""

    def __init__(self, docs, metas, sims):
        self.docs, self.metas, self.sims = docs, metas, sims
        self.k_seen: list[int] = []

    def query(self, q, k=4):
        self.k_seen.append(k)
        return self.docs[:k], self.metas[:k], self.sims[:k]


class FakeLLM:
    supports_streaming = False
    def generate(self, *_a, **_kw): return "Ответ [1]"


# ---------------------------------------------------------------------------
# RAGAssistant + reranker
# ---------------------------------------------------------------------------


def _build_bot(reranker=None, **kwargs):
    docs = [f"doc-{i}" * 5 for i in range(8)]   # 5*len растёт с i
    metas = [{"title": f"T{i}", "url": f"https://x/{i}"} for i in range(8)]
    sims = [0.9 - i * 0.05 for i in range(8)]    # выше для первых
    return RAGAssistant(
        index=FakeIndex(docs, metas, sims),
        llm=FakeLLM(),
        reranker=reranker,
        **kwargs,
    )


def test_no_reranker_keeps_default_top_k():
    """Без реранкера запрашиваем ровно top_k."""
    bot = _build_bot()
    bot.ask("flexbox")
    # index.query должен был получить k=top_k=4
    assert bot.index.k_seen == [4]


def test_with_reranker_does_over_retrieval():
    """С реранкером — индекс получает top_k_retrieval (по умолчанию 16)."""
    r = FakeReranker()
    bot = _build_bot(reranker=r)
    bot.ask("flexbox")
    assert bot.index.k_seen == [16]  # default top_k_retrieval


def test_reranker_changes_order():
    """Реранкер с собственными скорами должен переупорядочить sources."""
    r = FakeReranker(scores=[0.1, 0.9, 0.5, 0.2, 0.95, 0.0, 0.3, 0.4])
    bot = _build_bot(reranker=r, top_k_retrieval=8, top_k=3)
    ans = bot.ask("flexbox")
    # топ-3 по rerank score: indexes 4(0.95), 1(0.9), 2(0.5)
    titles = [s["title"] for s in ans.sources]
    assert titles == ["T4", "T1", "T2"]
    # rerank_score есть в payload
    assert all("rerank_score" in s for s in ans.sources)
    assert ans.sources[0]["rerank_score"] == 0.95


def test_threshold_filters_low_scores():
    """rerank_threshold отбрасывает фрагменты ниже порога."""
    r = FakeReranker(scores=[0.95, 0.9, 0.2, 0.1, 0.05, 0.0, 0.0, 0.0])
    bot = _build_bot(reranker=r, top_k_retrieval=8, top_k=4, rerank_threshold=0.5)
    ans = bot.ask("flexbox")
    assert ans.blocked is None
    assert len(ans.sources) == 2  # 0.95 и 0.9
    assert all(s["rerank_score"] >= 0.5 for s in ans.sources)


def test_threshold_blocks_when_nothing_passes():
    """Если все rerank_score ниже порога — out_of_corpus."""
    r = FakeReranker(scores=[0.1, 0.05, 0.0, 0.0])
    bot = _build_bot(reranker=r, top_k_retrieval=4, top_k=4, rerank_threshold=0.5)
    ans = bot.ask("flexbox")
    assert ans.blocked == "out_of_corpus"


def test_reranker_failure_falls_back_gracefully():
    """Если reranker.rerank() падает — пайплайн не должен ломаться."""

    class BrokenReranker:
        def rerank(self, q, cands):
            raise RuntimeError("simulated reranker failure")

    bot = _build_bot(reranker=BrokenReranker(), top_k_retrieval=4)
    ans = bot.ask("flexbox")
    # без исключения — пайплайн отработал; sources без rerank_score
    assert ans.blocked is None
    assert all("rerank_score" not in s for s in ans.sources)


def test_red_zone_skips_reranker_entirely():
    r = FakeReranker()
    bot = _build_bot(reranker=r)
    ans = bot.ask("Поставь мне оценку")
    assert ans.blocked == "red_zone"
    assert r.calls == []
    assert bot.index.k_seen == []


def test_stream_uses_reranker_too():
    """ask_stream должен идти через тот же _retrieve_and_rank."""
    r = FakeReranker(scores=[0.1, 0.95, 0.5, 0.2])
    bot = _build_bot(reranker=r, top_k_retrieval=4, top_k=2)
    sa = bot.ask_stream("flexbox")
    assert sa.blocked is None
    titles = [s["title"] for s in sa.sources]
    # топ-2 по rerank_score: 0.95 (T1) и 0.5 (T2)
    assert titles == ["T1", "T2"]
    assert sa.sources[0]["rerank_score"] == 0.95


# ---------------------------------------------------------------------------
# GigaChatReranker — парсер скоров (без сети)
# ---------------------------------------------------------------------------


def test_gigachat_reranker_score_parser():
    from web_ai_assistant.rerankers.gigachat import GigaChatReranker

    # ответ модели
    raw = "1: 9\n2: 3\n3: 7\n"
    scores = GigaChatReranker._parse_scores(raw, expected=3)
    assert scores == [0.9, 0.3, 0.7]


def test_gigachat_reranker_score_parser_handles_garbage():
    from web_ai_assistant.rerankers.gigachat import GigaChatReranker

    raw = "извините, я не уверен\n1: 8\nsome noise\n[2]: 6.5"
    scores = GigaChatReranker._parse_scores(raw, expected=3)
    assert scores[0] == 0.8
    assert scores[1] == pytest.approx(0.65)
    # для отсутствующих — 0.0
    assert scores[2] == 0.0


def test_gigachat_reranker_clamps_to_unit_range():
    from web_ai_assistant.rerankers.gigachat import GigaChatReranker

    raw = "1: 15\n2: -3"
    scores = GigaChatReranker._parse_scores(raw, expected=2)
    assert scores == [1.0, 0.0]

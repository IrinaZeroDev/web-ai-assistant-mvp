"""Тесты Maximal Marginal Relevance: чистая функция + интеграция в RAGAssistant."""

from __future__ import annotations

import math

import pytest

from web_ai_assistant.retrieval.mmr import mmr_select

# --------------------------------------------------------------------------- #
#                              Unit: mmr_select                                #
# --------------------------------------------------------------------------- #


def test_mmr_empty_inputs() -> None:
    assert mmr_select(query_emb=[1.0, 0.0], cand_embs=[], k=3) == []
    assert mmr_select(query_emb=[1.0, 0.0], cand_embs=[[1.0, 0.0]], k=0) == []


def test_mmr_invalid_lambda() -> None:
    with pytest.raises(ValueError):
        mmr_select(query_emb=[1.0], cand_embs=[[1.0]], k=1, lambda_mult=1.5)


def test_mmr_lambda_one_equals_topk() -> None:
    """λ=1.0 → чисто по релевантности (тот же порядок, что и top-k по sim)."""
    # Запрос: e1 = [1, 0]. Три кандидата с убывающим sim.
    query = [1.0, 0.0]
    cands = [
        [1.0, 0.0],      # sim = 1.0
        [0.9, 0.1],      # sim ≈ 0.994
        [0.5, 0.5],      # sim ≈ 0.707
    ]
    order = mmr_select(query, cands, k=3, lambda_mult=1.0)
    assert order == [0, 1, 2]


def test_mmr_lambda_zero_chooses_diverse_after_first() -> None:
    """λ=0.0 → первый по релевантности, потом максимально непохожие."""
    query = [1.0, 0.0]
    cands = [
        [1.0, 0.0],     # релевантный
        [0.99, 0.1],    # почти дубль #0
        [0.0, 1.0],     # совершенно ортогональный
    ]
    order = mmr_select(query, cands, k=3, lambda_mult=0.0)
    # Первый — argmax по sim (индекс 0). Второй должен быть ортогональный #2,
    # а не почти-дубль #1.
    assert order[0] == 0
    assert order[1] == 2
    assert order[2] == 1


def test_mmr_breaks_duplicates() -> None:
    """Главный мотив для нашего проекта: 4 фрагмента, два дубля.

    Без MMR top-4 вернул бы пары [0, 1] и [2, 3] (пары почти идентичных
    фрагментов). MMR с λ=0.5 должен выбрать разнообразный набор.
    """
    # cosine инвариантен к длине вектора — разные кластеры должны иметь разные направления.
    query = [1.0, 0.0, 0.0, 0.0]
    cands = [
        [0.99, 0.10, 0.05, 0.0],   # кластер A: sim ≈ 0.985
        [0.99, 0.11, 0.06, 0.0],   # ≈ дубль A
        [0.80, 0.0, 0.60, 0.0],    # кластер B: sim ≈ 0.80
        [0.79, 0.0, 0.61, 0.05],   # ≈ дубль B
        [0.60, 0.0, 0.0, 0.80],    # кластер C: sim ≈ 0.60
    ]
    order = mmr_select(query, cands, k=3, lambda_mult=0.5)
    # Первый — точно из самого релевантного (кластер A).
    assert order[0] in (0, 1)
    # В отобранных трёх должны быть представители как минимум 2 кластеров.
    clusters = {0: "A", 1: "A", 2: "B", 3: "B", 4: "C"}
    selected_clusters = {clusters[i] for i in order}
    assert len(selected_clusters) >= 2


def test_mmr_handles_cand_sims_override() -> None:
    """Если ``cand_sims`` передан — используем его вместо вычисленных."""
    cands = [[1.0, 0.0], [0.0, 1.0]]
    # Передадим фейковые sim'ы, в которых второй кандидат «лучше».
    order = mmr_select(query_emb=[1.0, 0.0], cand_embs=cands, k=2, lambda_mult=1.0, cand_sims=[0.1, 0.9])
    assert order[0] == 1


def test_mmr_zero_norm_embedding_safe() -> None:
    """Нулевые векторы не должны вызывать NaN."""
    order = mmr_select(query_emb=[0.0, 0.0], cand_embs=[[0.0, 0.0], [1.0, 0.0]], k=2, lambda_mult=0.5)
    assert len(order) == 2
    assert all(not math.isnan(i) for i in order)


def test_mmr_k_greater_than_n() -> None:
    order = mmr_select(query_emb=[1.0, 0.0], cand_embs=[[1.0, 0.0], [0.0, 1.0]], k=10)
    assert len(order) == 2


# --------------------------------------------------------------------------- #
#                         Integration: RAGAssistant + MMR                      #
# --------------------------------------------------------------------------- #


class _FakeIndex:
    """FakeIndex, поддерживающий include_embeddings (для MMR-ветки)."""

    def __init__(self, docs, metas, sims, embs):
        self.docs = docs
        self.metas = metas
        self.sims = sims
        self.embs = embs

    def query(self, question: str, k: int = 4, include_embeddings: bool = False):
        if include_embeddings:
            return self.docs[:k], self.metas[:k], self.sims[:k], self.embs[:k]
        return self.docs[:k], self.metas[:k], self.sims[:k]


class _FakeLLM:
    supports_streaming = False

    def generate(self, messages, max_new_tokens=512, temperature=0.0):
        return "ответ по контексту [1]"


def _make_rag(*, mmr: bool, mmr_lambda: float = 0.7, reranker=None):
    from web_ai_assistant.rag import RAGAssistant

    docs = ["frag A1", "frag A2 dup", "frag B", "frag B dup", "frag C"]
    metas = [{"title": f"T{i}", "url": f"https://e.dev/{i}"} for i in range(5)]
    sims = [0.95, 0.94, 0.70, 0.69, 0.60]
    embs = [
        [0.95, 0.05, 0.0],
        [0.94, 0.06, 0.0],
        [0.70, 0.0, 0.71],
        [0.69, 0.0, 0.72],
        [0.60, 0.0, 0.0],
    ]
    idx = _FakeIndex(docs, metas, sims, embs)
    return RAGAssistant(
        index=idx,
        llm=_FakeLLM(),
        sim_threshold=0.55,
        top_k=3,
        top_k_retrieval=5,
        reranker=reranker,
        mmr=mmr,
        mmr_lambda=mmr_lambda,
    )


def test_ragassistant_mmr_off_returns_top3() -> None:
    """Без MMR top-3 берётся «как есть» (по индексу retrieval)."""
    rag = _make_rag(mmr=False)
    ans = rag.ask("Что такое HTML?")
    assert ans.blocked is None
    urls = [s["url"] for s in ans.sources]
    # Без MMR top_k=3, top_k_retrieval=5 → берётся только top_k без оверретрива
    assert urls == ["https://e.dev/0", "https://e.dev/1", "https://e.dev/2"]


def test_ragassistant_mmr_on_breaks_duplicates() -> None:
    """С MMR top-3 должен покрыть >=2 кластера, не выбирая два почти-дубля."""
    rag = _make_rag(mmr=True, mmr_lambda=0.5)
    ans = rag.ask("Что такое HTML?")
    assert ans.blocked is None
    urls = [s["url"] for s in ans.sources]
    # Первый = самый релевантный (cluster A, индекс 0 или 1).
    assert urls[0] in ("https://e.dev/0", "https://e.dev/1")
    # Второй — НЕ должен быть дублем первого (т.е. не A→A).
    assert urls[1] not in ("https://e.dev/0", "https://e.dev/1")


def test_ragassistant_mmr_lambda_validation() -> None:
    from web_ai_assistant.rag import RAGAssistant

    with pytest.raises(ValueError):
        RAGAssistant(index=_FakeIndex([], [], [], []), llm=_FakeLLM(), mmr=True, mmr_lambda=1.5)


def test_ragassistant_mmr_with_reranker() -> None:
    """MMR + реранкер: MMR переупорядочивает пул, реранкер потом сортирует по своим скорам."""

    class _Reranker:
        def rerank(self, q, candidates):
            # rerank: предпочитаем длинные документы. У нас "frag B dup" — длина 10.
            return [float(len(c)) for c in candidates]

    rag = _make_rag(mmr=True, mmr_lambda=0.5, reranker=_Reranker())
    ans = rag.ask("Что такое CSS?")
    assert ans.blocked is None
    # rerank_score должен присутствовать в источниках
    assert all("rerank_score" in s for s in ans.sources)


def test_ragassistant_mmr_blocked_by_threshold() -> None:
    """Если max_sim < sim_threshold — MMR не должен ничего изменить, блокировка идёт раньше."""
    from web_ai_assistant.rag import RAGAssistant

    docs = ["irrelevant"]
    metas = [{"title": "x", "url": "u"}]
    sims = [0.10]
    embs = [[0.1, 0.0]]
    idx = _FakeIndex(docs, metas, sims, embs)
    rag = RAGAssistant(index=idx, llm=_FakeLLM(), sim_threshold=0.55, top_k=3, mmr=True)
    ans = rag.ask("off-topic question")
    assert ans.blocked == "out_of_corpus"


def test_factories_yaml_supports_mmr(tmp_path) -> None:
    """build_from_yaml пробрасывает mmr/mmr_lambda — проверяем через monkey-patch."""
    from unittest.mock import patch

    yaml_text = """\
embedder: { provider: e5 }
llm: { provider: qwen }
rag: { sim_threshold: 0.55, top_k: 4, mmr: true, mmr_lambda: 0.55 }
corpus: { type: mdn }
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_text, encoding="utf-8")

    captured = {}

    class _Stub:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    # Импорты corpus/index/rag в factories.py ленивые (внутри функции) — патчим по оригиналу.
    with patch("web_ai_assistant.corpus.load_mdn_corpus", return_value=[]), \
         patch("web_ai_assistant.corpus.split_documents", return_value=[]), \
         patch("web_ai_assistant.index.VectorIndex") as VI, \
         patch("web_ai_assistant.eval.factories._build_embedder", return_value=object()), \
         patch("web_ai_assistant.eval.factories._build_llm", return_value=object()), \
         patch("web_ai_assistant.rag.RAGAssistant", _Stub):
        VI.return_value.add.return_value = 0
        from web_ai_assistant.eval.factories import build_from_yaml

        build_from_yaml(p)

    assert captured.get("mmr") is True
    assert captured.get("mmr_lambda") == pytest.approx(0.55)

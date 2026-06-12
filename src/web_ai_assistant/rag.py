"""Главная RAG-цепочка ``ask(question)``."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .guards import (
    ESCALATION_REPLY,
    OUT_OF_CORPUS_REPLY,
    RED_ZONE_REPLY,
    is_escalation,
    is_red_zone,
)

SYSTEM_PROMPT = """Ты — учебный ассистент по web-разработке для студентов ИСТ ДГТУ.
Правила (нарушать нельзя):
1. Отвечай ТОЛЬКО на основе фрагментов из секции «Контекст». Если ответа в контексте нет — напиши: «Я не нашёл этого в материалах курса. Обратитесь к преподавателю».
2. Каждое утверждение сопровождай ссылкой вида [1], [2] на номер источника.
3. Не выставляй оценки. Не комментируй работу студента дидактически. Если просят оценку — откажи и сошлись на правила курса.
4. Если тема — новая для студента (он явно об этом пишет) — не объясняй, предложи записаться на консультацию к преподавателю.
5. Не выдумывай. Не используй знания вне контекста."""


class _Index(Protocol):
    def query(self, question: str, k: int = 4, include_embeddings: bool = False): ...


@runtime_checkable
class _LLM(Protocol):
    """Протокол LLM. ``stream_generate`` опционален — выводится из ``supports_streaming``."""

    supports_streaming: bool

    def generate(self, messages: list[dict], max_new_tokens: int = ..., temperature: float = ...) -> str: ...

    def stream_generate(
        self, messages: list[dict], max_new_tokens: int = ..., temperature: float = ...
    ) -> Iterator[str]: ...


@dataclass
class Answer:
    answer: str
    sources: list[dict] = field(default_factory=list)
    max_sim: float | None = None
    blocked: str | None = None  # red_zone | escalation | out_of_corpus | None


@dataclass
class StreamAnswer:
    """Результат стриминга: метаданные известны сразу, токены — итератор."""

    tokens: Iterator[str]
    sources: list[dict] = field(default_factory=list)
    max_sim: float | None = None
    blocked: str | None = None


def _build_context(docs: list[str], metas: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] ({m['title']} — {m['url']})\n{d}"
        for i, (d, m) in enumerate(zip(docs, metas, strict=False), start=1)
    )


class RAGAssistant:
    """Цепочка: red-zone → escalation → retrieve → (rerank) → similarity gate → LLM.

    Если задан ``reranker``, выполняется over-retrieval (``top_k_retrieval``,
    обычно 16) — затем cross-encoder перепроверяет фрагменты и оставляет
    лучшие ``top_k`` (обычно 4). При ``rerank_threshold`` ниже него отбрасываются.
    """

    def __init__(
        self,
        index: _Index,
        llm: _LLM,
        sim_threshold: float = 0.55,
        top_k: int = 4,
        *,
        reranker=None,
        top_k_retrieval: int = 16,
        rerank_threshold: float | None = None,
        mmr: bool = False,
        mmr_lambda: float = 0.7,
    ):
        self.index = index
        self.llm = llm
        self.sim_threshold = sim_threshold
        self.top_k = top_k
        self.reranker = reranker
        # сколько достать ДО реранкинга/MMR — обычно в 4× больше итогового top_k
        self.top_k_retrieval = max(top_k_retrieval, top_k)
        self.rerank_threshold = rerank_threshold
        self.mmr = mmr
        if not (0.0 <= mmr_lambda <= 1.0):
            raise ValueError(f"mmr_lambda должен быть в [0, 1], получено: {mmr_lambda}")
        self.mmr_lambda = mmr_lambda

    def ask(self, question: str) -> Answer:
        # 1. Архитектурный red-zone отказ (до retrieval!).
        if is_red_zone(question):
            return Answer(answer=RED_ZONE_REPLY, blocked="red_zone")

        # 2. Эскалация по новой теме.
        if is_escalation(question):
            return Answer(answer=ESCALATION_REPLY, blocked="escalation")

        # 3. Retrieval + (optional) reranking.
        result = self._retrieve_and_rank(question)
        if result.get("blocked"):
            return Answer(
                answer=result["reply"],
                max_sim=result.get("max_sim"),
                blocked=result["blocked"],
            )

        docs = result["docs"]
        metas = result["metas"]
        sims = result["sims"]
        rerank_scores = result["rerank_scores"]
        max_sim = result["max_sim"]

        # 4. LLM с RAG-промптом.
        messages = self._build_messages(docs, metas, question)
        text = self.llm.generate(messages, max_new_tokens=400, temperature=0.2)
        return Answer(
            answer=text,
            sources=self._sources_payload(metas, sims, rerank_scores),
            max_sim=max_sim,
        )

    # ---------- streaming ----------

    @property
    def supports_streaming(self) -> bool:
        """True если подключённый LLM поддерживает token-by-token streaming."""
        return bool(getattr(self.llm, "supports_streaming", False)) and hasattr(
            self.llm, "stream_generate"
        )

    def ask_stream(self, question: str) -> StreamAnswer:
        """Как ``ask``, но возвращает токен-итератор.

        Метаданные (sources, blocked, max_sim) известны сразу и возвращаются в :class:`StreamAnswer`.
        Генерация начинается в момент первого ``next(answer.tokens)``.
        Для заблокированных запросов итератор выдаёт одну строку — готовый текст отказа.
        """
        # общая логика отказов совпадает с ask()
        if is_red_zone(question):
            return StreamAnswer(tokens=iter([RED_ZONE_REPLY]), blocked="red_zone")
        if is_escalation(question):
            return StreamAnswer(tokens=iter([ESCALATION_REPLY]), blocked="escalation")

        result = self._retrieve_and_rank(question)
        if result.get("blocked"):
            return StreamAnswer(
                tokens=iter([result["reply"]]),
                max_sim=result.get("max_sim"),
                blocked=result["blocked"],
            )

        docs = result["docs"]
        metas = result["metas"]
        sims = result["sims"]
        rerank_scores = result["rerank_scores"]
        max_sim = result["max_sim"]

        messages = self._build_messages(docs, metas, question)
        sources = self._sources_payload(metas, sims, rerank_scores)

        if self.supports_streaming:
            tokens = self.llm.stream_generate(messages, max_new_tokens=400, temperature=0.2)
        else:
            tokens = iter([self.llm.generate(messages, max_new_tokens=400, temperature=0.2)])

        return StreamAnswer(tokens=tokens, sources=sources, max_sim=max_sim)

    # ---------- helpers ----------

    def _build_messages(self, docs: list[str], metas: list[dict], question: str) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Контекст:\n{_build_context(docs, metas)}\n\nВопрос: {question}",
            },
        ]

    @staticmethod
    def _sources_payload(
        metas: list[dict],
        sims: list[float],
        rerank_scores: list[float] | None = None,
    ) -> list[dict]:
        out: list[dict] = []
        for i, (m, s) in enumerate(zip(metas, sims, strict=False)):
            entry = {
                "id": i + 1,
                "title": m["title"],
                "url": m["url"],
                "sim": round(s, 3),
            }
            if rerank_scores is not None and i < len(rerank_scores):
                entry["rerank_score"] = round(float(rerank_scores[i]), 3)
            out.append(entry)
        return out

    # ---------- retrieve + rerank ----------

    def _retrieve_and_rank(self, question: str) -> dict:
        """Возвращает либо ``{"blocked": ..., "reply": ..., "max_sim": ...}``,
        либо ``{"docs", "metas", "sims", "rerank_scores", "max_sim"}``.

        Поведение:

        1. Извлекаем ``top_k_retrieval`` фрагментов (over-retrieval).
        2. Проверяем similarity gate по retrieval-скорам (``max_sim``).
        3. Если задан ``reranker`` — пересортируем кандидатов по rerank_score.
        4. Если задан ``rerank_threshold`` — отфильтровываем слабые.
        5. Усекаем до ``top_k`` финальных.
        """
        # Над retrieval'ом работает либо реранкер, либо MMR — оба хотят пул побольше.
        wants_pool = self.reranker is not None or self.mmr
        k_retrieval = self.top_k_retrieval if wants_pool else self.top_k
        cand_embs: list[list[float]] | None = None
        if self.mmr:
            docs, metas, sims, cand_embs = self.index.query(
                question, k=k_retrieval, include_embeddings=True
            )
        else:
            docs, metas, sims = self.index.query(question, k=k_retrieval)
        max_sim = max(sims) if sims else 0.0

        if max_sim < self.sim_threshold:
            return {
                "blocked": "out_of_corpus",
                "reply": OUT_OF_CORPUS_REPLY,
                "max_sim": max_sim,
            }

        # ---------- MMR: diversity-переупорядочивание ДО реранкинга ----------
        # Если реранкер включён — оставляем элементы в пуле (реранкер работает
        # по всему ``top_k_retrieval``), но меняем порядок в MMR-логике. Если реранкера нет —
        # MMR сразу урезает до ``top_k``.
        if self.mmr and cand_embs and docs:
            from .retrieval.mmr import mmr_select

            k_mmr = len(docs) if self.reranker is not None else self.top_k
            order = mmr_select(
                query_emb=[],  # не нужен — передаём cand_sims
                cand_embs=cand_embs,
                k=k_mmr,
                lambda_mult=self.mmr_lambda,
                cand_sims=sims,
            )
            docs = [docs[i] for i in order]
            metas = [metas[i] for i in order]
            sims = [sims[i] for i in order]

        rerank_scores: list[float] | None = None
        if self.reranker is not None and docs:
            try:
                rerank_scores = list(self.reranker.rerank(question, docs))
            except Exception:  # noqa: BLE001
                rerank_scores = None

            if rerank_scores is not None:
                # сортируем по rerank_score (DESC), сохраняя соответствие metas/sims
                order = sorted(
                    range(len(docs)),
                    key=lambda i: rerank_scores[i] if i < len(rerank_scores) else 0.0,
                    reverse=True,
                )
                docs = [docs[i] for i in order]
                metas = [metas[i] for i in order]
                sims = [sims[i] for i in order]
                rerank_scores = [rerank_scores[i] for i in order]

                if self.rerank_threshold is not None:
                    kept = [i for i, s in enumerate(rerank_scores) if s >= self.rerank_threshold]
                    if not kept:
                        return {
                            "blocked": "out_of_corpus",
                            "reply": OUT_OF_CORPUS_REPLY,
                            "max_sim": max_sim,
                        }
                    docs = [docs[i] for i in kept]
                    metas = [metas[i] for i in kept]
                    sims = [sims[i] for i in kept]
                    rerank_scores = [rerank_scores[i] for i in kept]

        # финальная отсечка top_k
        docs = docs[: self.top_k]
        metas = metas[: self.top_k]
        sims = sims[: self.top_k]
        if rerank_scores is not None:
            rerank_scores = rerank_scores[: self.top_k]

        return {
            "docs": docs,
            "metas": metas,
            "sims": sims,
            "rerank_scores": rerank_scores,
            "max_sim": max_sim,
        }

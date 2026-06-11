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
    def query(self, question: str, k: int = 4): ...


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
    """Цепочка: red-zone → escalation → retrieve → similarity gate → LLM."""

    def __init__(self, index: _Index, llm: _LLM, sim_threshold: float = 0.55, top_k: int = 4):
        self.index = index
        self.llm = llm
        self.sim_threshold = sim_threshold
        self.top_k = top_k

    def ask(self, question: str) -> Answer:
        # 1. Архитектурный red-zone отказ (до retrieval!).
        if is_red_zone(question):
            return Answer(answer=RED_ZONE_REPLY, blocked="red_zone")

        # 2. Эскалация по новой теме.
        if is_escalation(question):
            return Answer(answer=ESCALATION_REPLY, blocked="escalation")

        # 3. Retrieval.
        docs, metas, sims = self.index.query(question, k=self.top_k)
        max_sim = max(sims) if sims else 0.0

        # 4. Out-of-corpus guard.
        if max_sim < self.sim_threshold:
            return Answer(answer=OUT_OF_CORPUS_REPLY, max_sim=max_sim, blocked="out_of_corpus")

        # 5. LLM с RAG-промптом.
        messages = self._build_messages(docs, metas, question)
        text = self.llm.generate(messages, max_new_tokens=400, temperature=0.2)
        return Answer(
            answer=text,
            sources=self._sources_payload(metas, sims),
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

        docs, metas, sims = self.index.query(question, k=self.top_k)
        max_sim = max(sims) if sims else 0.0
        if max_sim < self.sim_threshold:
            return StreamAnswer(
                tokens=iter([OUT_OF_CORPUS_REPLY]),
                max_sim=max_sim,
                blocked="out_of_corpus",
            )

        messages = self._build_messages(docs, metas, question)
        sources = self._sources_payload(metas, sims)

        if self.supports_streaming:
            tokens = self.llm.stream_generate(messages, max_new_tokens=400, temperature=0.2)
        else:
            # fallback — однократная выдача всего ответа
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
    def _sources_payload(metas: list[dict], sims: list[float]) -> list[dict]:
        return [
            {"id": i + 1, "title": m["title"], "url": m["url"], "sim": round(s, 3)}
            for i, (m, s) in enumerate(zip(metas, sims, strict=False))
        ]

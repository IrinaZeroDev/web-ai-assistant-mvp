"""Тест RAG-цепочки на моках — без тяжёлых ML-зависимостей."""

from __future__ import annotations

from web_ai_assistant.rag import RAGAssistant


class FakeIndex:
    def __init__(self, docs, metas, sims):
        self._docs, self._metas, self._sims = docs, metas, sims

    def query(self, question: str, k: int = 4):
        return self._docs, self._metas, self._sims


class FakeLLM:
    def __init__(self, reply: str = "Ответ со ссылкой [1]."):
        self.reply = reply
        self.last_messages: list[dict] | None = None

    def generate(self, messages, max_new_tokens=400, temperature=0.2):
        self.last_messages = messages
        return self.reply


def _good_index():
    return FakeIndex(
        docs=["flexbox описание", "css selectors"],
        metas=[
            {"title": "CSS/flexbox", "url": "https://example.org/fb"},
            {"title": "CSS/selectors", "url": "https://example.org/sel"},
        ],
        sims=[0.91, 0.72],
    )


def test_red_zone_short_circuits_before_retrieval() -> None:
    """Red-zone не должен вызывать LLM/индекс — это и есть «архитектурный отказ»."""
    idx = _good_index()
    llm = FakeLLM()
    bot = RAGAssistant(idx, llm)
    r = bot.ask("Поставь мне оценку")
    assert r.blocked == "red_zone"
    assert llm.last_messages is None  # LLM не звонили
    assert r.sources == []


def test_escalation_short_circuits() -> None:
    bot = RAGAssistant(_good_index(), FakeLLM())
    r = bot.ask("Объясни Composition API, это новая тема для меня")
    assert r.blocked == "escalation"


def test_out_of_corpus_when_low_similarity() -> None:
    """Главная гипотеза H4: при низком сходстве — отказ, не выдумка."""
    idx = FakeIndex(
        docs=["off-topic"],
        metas=[{"title": "x", "url": "https://x"}],
        sims=[0.30],  # ниже порога 0.55
    )
    llm = FakeLLM(reply="не должен быть вызван")
    bot = RAGAssistant(idx, llm)
    r = bot.ask("Как настроить FastAPI и SQLAlchemy?")
    assert r.blocked == "out_of_corpus"
    assert llm.last_messages is None


def test_answer_returned_with_sources() -> None:
    bot = RAGAssistant(_good_index(), FakeLLM(reply="flexbox это [1]"))
    r = bot.ask("Что такое flexbox?")
    assert r.blocked is None
    assert r.answer == "flexbox это [1]"
    assert len(r.sources) == 2
    assert r.sources[0]["title"] == "CSS/flexbox"
    assert r.max_sim == 0.91


def test_system_prompt_passed_to_llm() -> None:
    llm = FakeLLM()
    bot = RAGAssistant(_good_index(), llm)
    bot.ask("Что такое flexbox?")
    assert llm.last_messages is not None
    assert llm.last_messages[0]["role"] == "system"
    assert "ИСТ ДГТУ" in llm.last_messages[0]["content"]
    assert "[1]" in llm.last_messages[0]["content"]  # требование цитировать

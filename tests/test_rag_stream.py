"""Тесты ask_stream() — настоящий token-streaming и graceful fallback."""

from __future__ import annotations

from web_ai_assistant.rag import RAGAssistant


class FakeIndex:
    def __init__(self, docs, metas, sims):
        self._docs, self._metas, self._sims = docs, metas, sims

    def query(self, question: str, k: int = 4):
        return self._docs, self._metas, self._sims


class StreamingLLM:
    supports_streaming = True

    def __init__(self, tokens=("Hello ", "world ", "[1]")):
        self.tokens = list(tokens)

    def generate(self, *_, **__):
        return "".join(self.tokens)

    def stream_generate(self, *_, **__):
        yield from self.tokens


class NonStreamingLLM:
    supports_streaming = False

    def generate(self, *_, **__):
        return "Single shot answer [1]"


def _index_ok():
    return FakeIndex(
        docs=["d1"], metas=[{"title": "T", "url": "https://x"}], sims=[0.91]
    )


def test_ask_stream_uses_real_streaming_when_available():
    bot = RAGAssistant(_index_ok(), StreamingLLM())
    assert bot.supports_streaming
    sa = bot.ask_stream("Что такое flexbox?")
    assert sa.blocked is None
    assert sa.max_sim == 0.91
    out = list(sa.tokens)
    assert out == ["Hello ", "world ", "[1]"], "стриминг должен сохранить кванты"


def test_ask_stream_falls_back_to_single_shot_when_llm_does_not_stream():
    bot = RAGAssistant(_index_ok(), NonStreamingLLM())
    assert not bot.supports_streaming
    sa = bot.ask_stream("Что такое flexbox?")
    out = list(sa.tokens)
    assert out == ["Single shot answer [1]"]


def test_ask_stream_red_zone_yields_refusal_without_llm():
    class ExplodingLLM(StreamingLLM):
        def stream_generate(self, *_, **__):  # pragma: no cover - не должен вызваться
            raise AssertionError("LLM не должен вызываться для red-zone")

    bot = RAGAssistant(_index_ok(), ExplodingLLM())
    sa = bot.ask_stream("Поставь мне оценку")
    assert sa.blocked == "red_zone"
    tokens = list(sa.tokens)
    assert len(tokens) == 1
    assert "не выставляю" in tokens[0].lower() or "запрещено" in tokens[0].lower()


def test_ask_stream_out_of_corpus():
    idx = FakeIndex(docs=["x"], metas=[{"title": "x", "url": "https://x"}], sims=[0.20])
    bot = RAGAssistant(idx, StreamingLLM())
    sa = bot.ask_stream("Как настроить FastAPI?")
    assert sa.blocked == "out_of_corpus"
    assert list(sa.tokens) == ["Я не нашёл этого в материалах курса. Обратитесь к преподавателю."]

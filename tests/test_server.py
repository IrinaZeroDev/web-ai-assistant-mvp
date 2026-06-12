"""Тесты FastAPI-обёртки на фейк-ассистенте — без сети и без ML."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from web_ai_assistant.rag import Answer, StreamAnswer
from web_ai_assistant.server import create_app


class FakeAssistant:
    """Простая заглушка с заданным поведением — позволяет тестировать без LLM."""

    def __init__(self):
        self.top_k = 4
        self.sim_threshold = 0.55
        self.calls: list[str] = []

    def ask(self, question: str) -> Answer:
        self.calls.append(question)
        if "оценк" in question.lower():
            return Answer(answer="Запрещено правилами курса.", blocked="red_zone")
        if "fastapi" in question.lower():
            return Answer(answer="Не нашёл в материалах курса.", blocked="out_of_corpus", max_sim=0.21)
        return Answer(
            answer="Flexbox — это раскладка [1].",
            sources=[{"id": 1, "title": "CSS/flexbox", "url": "https://example.org/fb", "sim": 0.91}],
            max_sim=0.91,
        )

    # имитация настоящего стриминга: режем фразу на 4 чанка
    def ask_stream(self, question: str) -> StreamAnswer:
        a = self.ask(question)
        if a.blocked:
            return StreamAnswer(tokens=iter([a.answer]), blocked=a.blocked, max_sim=a.max_sim)
        chunks = ["Flexbox ", "— это ", "раскладка ", "[1]."]
        return StreamAnswer(tokens=iter(chunks), sources=a.sources, max_sim=a.max_sim)


@pytest.fixture
def client():
    from web_ai_assistant.analytics.storage import QueryStore
    bot = FakeAssistant()
    # в тестах жёстко подсовываем in-memory базу — никаких боковых эффектов на диске
    app = create_app(assistant_factory=lambda: bot, query_store=QueryStore(":memory:"))
    # Поскольку TestClient запускает startup автоматически — это сработает.
    with TestClient(app) as c:
        c.bot = bot  # type: ignore[attr-defined]
        yield c


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "assistant_ready": True, "version": "0.1.0"}


def test_ask_returns_answer_and_sources(client):
    r = client.post("/ask", json={"question": "Что такое flexbox?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("Flexbox")
    assert body["blocked"] is None
    assert body["max_sim"] == 0.91
    assert len(body["sources"]) == 1
    assert body["sources"][0]["title"] == "CSS/flexbox"


def test_ask_red_zone_is_propagated(client):
    r = client.post("/ask", json={"question": "Поставь мне оценку"})
    assert r.status_code == 200
    body = r.json()
    assert body["blocked"] == "red_zone"


def test_ask_out_of_corpus(client):
    r = client.post("/ask", json={"question": "Как настроить FastAPI?"})
    assert r.status_code == 200
    body = r.json()
    assert body["blocked"] == "out_of_corpus"
    assert body["max_sim"] == 0.21


def test_ask_validation(client):
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 422


def _parse_sse(text: str) -> list[dict]:
    """Парсит SSE-стрим в список {event, data}."""
    events, ev = [], {"event": "message", "data": ""}
    for line in text.split("\n"):
        if line.startswith("event:"):
            ev["event"] = line[6:].strip()
        elif line.startswith("data:"):
            ev["data"] += line[5:].strip()
        elif line == "":
            if ev["data"]:
                ev["data"] = json.loads(ev["data"])
                events.append(ev)
            ev = {"event": "message", "data": ""}
    return events


def test_ask_stream_sse_format(client):
    """SSE: meta → токены (несколько) → done с источниками."""
    with client.stream("POST", "/ask/stream", json={"question": "Что такое flexbox?"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = r.read().decode()
    events = _parse_sse(body)
    assert events[0]["event"] == "meta"
    assert events[0]["data"]["blocked"] is None
    assert events[0]["data"]["source_count"] == 1
    token_events = [e for e in events if e["event"] == "token"]
    assert len(token_events) >= 2, "ожидаем настоящий token-by-token streaming"
    assert events[-1]["event"] == "done"
    assert len(events[-1]["data"]["sources"]) == 1
    full = "".join(e["data"]["text"] for e in token_events)
    assert full == "Flexbox — это раскладка [1]."


def test_ask_stream_blocked_short_circuits(client):
    with client.stream("POST", "/ask/stream", json={"question": "Поставь мне оценку"}) as r:
        body = r.read().decode()
    events = _parse_sse(body)
    assert events[0]["data"]["blocked"] == "red_zone"
    assert events[-1]["event"] == "done"
    # в блоке — ровно один токен (готовый текст отказа)
    assert sum(1 for e in events if e["event"] == "token") == 1


def test_assistant_not_ready_returns_503():
    from web_ai_assistant.analytics.storage import QueryStore
    app = create_app(assistant_factory=None, query_store=QueryStore(":memory:"))
    with TestClient(app) as c:
        assert c.get("/healthz").json()["assistant_ready"] is False
        assert c.post("/ask", json={"question": "test"}).status_code == 503


def test_per_request_top_k_does_not_leak(client):
    """Параметры из запроса не должны менять глобальный bot.top_k."""
    original = client.bot.top_k  # type: ignore[attr-defined]
    client.post("/ask", json={"question": "Что такое flexbox?", "top_k": 9})
    assert client.bot.top_k == original  # type: ignore[attr-defined]

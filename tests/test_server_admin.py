"""Admin-эндпоинты: логи и кластеры."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from web_ai_assistant.analytics.storage import QueryStore
from web_ai_assistant.rag import Answer, StreamAnswer
from web_ai_assistant.server import create_app


class FakeAssistant:
    top_k = 4
    sim_threshold = 0.55

    def ask(self, q):
        if "оценк" in q.lower():
            return Answer(answer="Запрещено правилами курса.", blocked="red_zone")
        return Answer(
            answer="Flexbox.",
            sources=[{"id": 1, "title": "CSS/flexbox", "url": "https://x", "sim": 0.9}],
            max_sim=0.9,
        )

    def ask_stream(self, q):
        a = self.ask(q)
        if a.blocked:
            return StreamAnswer(tokens=iter([a.answer]), blocked=a.blocked, max_sim=a.max_sim)
        return StreamAnswer(tokens=iter(["Flex", "box."]), sources=a.sources, max_sim=a.max_sim)


class FakeEmbedder:
    def embed_query(self, text):
        # детерминированный 4-d вектор: первые 4 ord% символа
        out = [0.0] * 4
        for i, ch in enumerate(text.lower()[:4]):
            out[i] = (ord(ch) % 10) / 10.0
        return out


@pytest.fixture
def client_with_logs():
    store = QueryStore(":memory:")
    app = create_app(
        assistant_factory=lambda: FakeAssistant(),
        query_store=store,
        embedder_factory=lambda: FakeEmbedder(),
    )
    with TestClient(app) as c:
        c.store = store  # type: ignore[attr-defined]
        yield c


def test_ask_writes_to_log(client_with_logs):
    r = client_with_logs.post("/ask", json={"question": "Что такое flexbox?"})
    assert r.status_code == 200
    rows = client_with_logs.store.recent(limit=10)  # type: ignore[attr-defined]
    assert len(rows) == 1
    assert "flexbox" in rows[0]["question"].lower()
    assert rows[0]["blocked"] is None
    assert rows[0]["source_count"] == 1
    assert rows[0]["latency_ms"] is not None


def test_red_zone_logged_with_blocked_field(client_with_logs):
    client_with_logs.post("/ask", json={"question": "Поставь мне оценку"})
    rows = client_with_logs.store.recent(limit=10)  # type: ignore[attr-defined]
    assert rows[0]["blocked"] == "red_zone"


def test_stream_logs_after_completion(client_with_logs):
    with client_with_logs.stream(
        "POST", "/ask/stream", json={"question": "Что такое flexbox?"}
    ) as r:
        r.read()  # дотянем стрим до конца
    rows = client_with_logs.store.recent(limit=10)  # type: ignore[attr-defined]
    assert len(rows) == 1
    # ответ собран из токенов
    assert rows[0]["answer"] == "Flexbox."


def test_pii_redacted_in_log(client_with_logs):
    client_with_logs.post(
        "/ask",
        json={"question": "Иванов Иван Иванович, мой email vasya@dstu.ru, как flexbox?"},
    )
    rows = client_with_logs.store.recent(limit=10)  # type: ignore[attr-defined]
    assert "vasya@dstu.ru" not in rows[0]["question"]
    assert "Иванов Иван Иванович" not in rows[0]["question"]
    assert "[EMAIL]" in rows[0]["question"]
    assert "[PERSON]" in rows[0]["question"]


def test_admin_stats(client_with_logs):
    client_with_logs.post("/ask", json={"question": "Что такое flexbox?"})
    client_with_logs.post("/ask", json={"question": "Поставь мне оценку"})
    r = client_with_logs.get("/admin/api/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert body["total"] == 2
    assert body["by_blocked"]["red_zone"] == 1
    assert body["by_blocked"]["__answered__"] == 1


def test_admin_clusters_returns_clusters(client_with_logs):
    # 6 разных вопросов — должно получиться несколько кластеров
    questions = [
        "flexbox basics",
        "flex container",
        "flexible layout",
        "grid template",
        "grid columns",
        "css grid usage",
    ]
    for q in questions:
        client_with_logs.post("/ask", json={"question": q})
    r = client_with_logs.get("/admin/api/clusters?n=2&backend=kmeans")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 6
    assert len(body["clusters"]) == 2
    for c in body["clusters"]:
        assert "label" in c and "size" in c and "representatives" in c


def test_admin_html_page(client_with_logs):
    client_with_logs.post("/ask", json={"question": "flexbox"})
    r = client_with_logs.get("/admin/clusters")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    body = r.text
    assert "Кластеры затруднений" in body
    assert "Всего запросов" in body


def test_admin_auth_required_when_password_set():
    store = QueryStore(":memory:")
    app = create_app(
        assistant_factory=lambda: FakeAssistant(),
        query_store=store,
        admin_password="hunter2",
    )
    with TestClient(app) as c:
        # без авторизации
        assert c.get("/admin/api/stats").status_code == 401
        # неверный пароль
        assert c.get("/admin/api/stats", auth=("admin", "wrong")).status_code == 401
        # верный пароль
        assert c.get("/admin/api/stats", auth=("admin", "hunter2")).status_code == 200


def test_log_disabled_via_env(monkeypatch):
    monkeypatch.setenv("LOG_QUERIES", "false")
    app = create_app(assistant_factory=lambda: FakeAssistant())
    with TestClient(app) as c:
        # /ask продолжает работать
        assert c.post("/ask", json={"question": "flexbox"}).status_code == 200
        # но статистика — пустая
        r = c.get("/admin/api/stats").json()
        assert r["enabled"] is False
        assert r["total"] == 0

"""SQLite-хранилище логов."""

from __future__ import annotations

from web_ai_assistant.analytics.storage import QueryLog, QueryStore, hash_ip


def test_schema_created_in_memory():
    store = QueryStore(":memory:")
    assert store.count() == 0


def test_insert_and_count():
    store = QueryStore(":memory:")
    rid = store.insert(QueryLog(question="Что такое flexbox?"))
    assert rid > 0
    assert store.count() == 1


def test_recent_returns_inserted():
    store = QueryStore(":memory:")
    store.insert(QueryLog(question="a"))
    store.insert(QueryLog(question="b"))
    rows = store.recent(limit=10)
    assert len(rows) == 2
    assert rows[0]["question"] == "b"  # сортировка id DESC


def test_by_blocked_groups():
    store = QueryStore(":memory:")
    store.insert(QueryLog(question="ok"))
    store.insert(QueryLog(question="grade me", blocked="red_zone"))
    store.insert(QueryLog(question="новая тема", blocked="escalation"))
    groups = store.by_blocked()
    assert groups["__answered__"] == 1
    assert groups["red_zone"] == 1
    assert groups["escalation"] == 1


def test_embedding_round_trip():
    store = QueryStore(":memory:")
    store.insert(QueryLog(question="q1", embedding=[0.1, 0.2, 0.3]))
    store.insert(QueryLog(question="q2", embedding=None))
    rows = store.all_for_clustering()
    assert len(rows) == 1
    assert rows[0]["embedding"] == [0.1, 0.2, 0.3]


def test_only_unblocked_filter():
    store = QueryStore(":memory:")
    store.insert(QueryLog(question="ok", embedding=[1.0, 0.0]))
    store.insert(QueryLog(question="blocked", embedding=[0.0, 1.0], blocked="red_zone"))
    assert len(store.all_for_clustering()) == 2
    assert len(store.all_for_clustering(only_unblocked=True)) == 1


def test_hash_ip_stable_and_short():
    assert hash_ip(None) is None
    assert hash_ip("") is None
    h1 = hash_ip("127.0.0.1")
    h2 = hash_ip("127.0.0.1")
    h3 = hash_ip("192.168.1.1")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 12

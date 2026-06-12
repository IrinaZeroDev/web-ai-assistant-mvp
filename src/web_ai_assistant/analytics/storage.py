"""SQLite-хранилище логов запросов к RAG-ассистенту.

Одна таблица ``queries`` со схемой:

| колонка | тип | назначение |
|---------|-----|------------|
| id              | INTEGER PK | autoincrement |
| ts              | TEXT       | ISO-8601 (UTC) момент запроса |
| question        | TEXT       | PII-редактированный текст запроса |
| answer          | TEXT       | PII-редактированный ответ (первые ~2 KB) |
| blocked         | TEXT NULL  | red_zone / escalation / out_of_corpus / NULL |
| max_sim         | REAL NULL  | retrieval max similarity (для аналитики порога) |
| source_count    | INTEGER    | сколько источников было в ответе |
| latency_ms      | INTEGER    | сколько занял bot.ask (включая retrieval+LLM) |
| llm_provider    | TEXT       | "gigachat" / "qwen" / "fake" — кто отвечал |
| client_id       | TEXT NULL  | анонимный id сессии (если фронт прокидывает) |
| ip_hash         | TEXT NULL  | sha256(ip)[:12] — для оценки уникальных users |

Все вставки идут в одном соединении (по умолчанию thread-safe режим WAL).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS queries (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT    NOT NULL,
    question      TEXT    NOT NULL,
    answer        TEXT,
    blocked       TEXT,
    max_sim       REAL,
    source_count  INTEGER NOT NULL DEFAULT 0,
    latency_ms    INTEGER,
    llm_provider  TEXT,
    client_id     TEXT,
    ip_hash       TEXT,
    embedding     TEXT
);

CREATE INDEX IF NOT EXISTS idx_queries_ts        ON queries(ts);
CREATE INDEX IF NOT EXISTS idx_queries_blocked   ON queries(blocked);
"""


@dataclass
class QueryLog:
    """Одна запись в таблице ``queries``."""

    question: str
    answer: str = ""
    blocked: str | None = None
    max_sim: float | None = None
    source_count: int = 0
    latency_ms: int | None = None
    llm_provider: str | None = None
    client_id: str | None = None
    ip_hash: str | None = None
    embedding: list[float] | None = field(default=None, repr=False)
    ts: str = ""

    def __post_init__(self) -> None:
        if not self.ts:
            self.ts = datetime.now(timezone.utc).isoformat(timespec="seconds")


class QueryStore:
    """Тонкая обёртка над SQLite. Безопасна для использования из нескольких потоков."""

    def __init__(self, db_path: str | Path | None = ":memory:") -> None:
        self.db_path = str(db_path) if db_path else ":memory:"
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            self.db_path, check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        if self.db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(SCHEMA_SQL)

    @contextmanager
    def _cursor(self):
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
            finally:
                cur.close()

    # ---------- write ----------

    def insert(self, log: QueryLog) -> int:
        emb_json = json.dumps(log.embedding) if log.embedding is not None else None
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO queries (
                    ts, question, answer, blocked, max_sim,
                    source_count, latency_ms, llm_provider, client_id, ip_hash, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    log.ts, log.question, log.answer, log.blocked, log.max_sim,
                    log.source_count, log.latency_ms, log.llm_provider,
                    log.client_id, log.ip_hash, emb_json,
                ),
            )
            return cur.lastrowid or 0

    # ---------- read ----------

    def count(self) -> int:
        with self._cursor() as cur:
            return cur.execute("SELECT COUNT(*) FROM queries").fetchone()[0]

    def by_blocked(self) -> dict[str, int]:
        with self._cursor() as cur:
            rows = cur.execute(
                "SELECT COALESCE(blocked, '__answered__') AS k, COUNT(*) AS n "
                "FROM queries GROUP BY blocked"
            ).fetchall()
        return {r["k"]: r["n"] for r in rows}

    def recent(self, limit: int = 100, include_blocked: bool = True) -> list[dict]:
        sql = "SELECT id, ts, question, answer, blocked, max_sim, source_count, latency_ms, llm_provider FROM queries"
        if not include_blocked:
            sql += " WHERE blocked IS NULL"
        sql += " ORDER BY id DESC LIMIT ?"
        with self._cursor() as cur:
            return [dict(r) for r in cur.execute(sql, (limit,)).fetchall()]

    def all_for_clustering(self, *, only_unblocked: bool = False) -> list[dict]:
        """Достаёт записи с эмбеддингами для кластеризации."""
        sql = "SELECT id, ts, question, embedding FROM queries WHERE embedding IS NOT NULL"
        if only_unblocked:
            sql += " AND blocked IS NULL"
        with self._cursor() as cur:
            rows = cur.execute(sql).fetchall()
        out: list[dict] = []
        for r in rows:
            try:
                emb = json.loads(r["embedding"])
            except (TypeError, json.JSONDecodeError):
                continue
            out.append({"id": r["id"], "ts": r["ts"], "question": r["question"], "embedding": emb})
        return out

    # ---------- lifecycle ----------

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> QueryStore:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------


def hash_ip(ip: str | None) -> str | None:
    """sha256(ip)[:12] — стабильный анонимный идентификатор источника."""
    if not ip:
        return None
    return hashlib.sha256(ip.encode("utf-8")).hexdigest()[:12]


def export_log_dict(log: QueryLog) -> dict:
    """Возвращает QueryLog без эмбеддинга — для отображения в UI."""
    d = asdict(log)
    d.pop("embedding", None)
    return d

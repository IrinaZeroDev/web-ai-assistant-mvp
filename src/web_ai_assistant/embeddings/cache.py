"""Дисковый кэш эмбеддингов на SQLite.

Ключевая идея: GigaChat embeddings (как и любой облачный embedder) стоят денег
и времени. При переиндексации одного и того же корпуса нет смысла гонять
одинаковые запросы — храним пары ``(model, sha256(text)) → vector``.

Ключ кэша — *(model, sha256(text))*, поэтому переход с ``Embeddings`` на
``EmbeddingsGigaR`` (другая размерность) не приведёт к коллизии: записи
лежат в одной таблице, но фильтруются по ``model``.

Использование::

    from web_ai_assistant.embeddings import GigaChatEmbedder, CachedEmbedder

    base = GigaChatEmbedder(model="EmbeddingsGigaR")
    embedder = CachedEmbedder(base, cache_path="./cache/emb.db")

    vecs = embedder.embed_passages(["hello", "world"])  # сетевые вызовы
    vecs = embedder.embed_passages(["hello", "world"])  # из кэша

Кэш потокобезопасен (SQLite + lock). При очистке БД проигнорированные ошибки
не валят пайплайн.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS embeddings (
    model       TEXT NOT NULL,
    text_hash   TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vector      TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (model, text_hash)
);

CREATE INDEX IF NOT EXISTS idx_emb_model ON embeddings(model);
"""


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# EmbeddingCache — низкоуровневое хранилище
# ---------------------------------------------------------------------------


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_ratio(self) -> float:
        return self.hits / self.total if self.total else 0.0


class EmbeddingCache:
    """SQLite-хранилище эмбеддингов с namespace по имени модели."""

    def __init__(self, path: str | Path = "cache/embeddings.db") -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            self.path, check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self.stats = CacheStats()
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

    # ---------- read ----------

    def get(self, model: str, text: str) -> list[float] | None:
        with self._cursor() as cur:
            row = cur.execute(
                "SELECT vector FROM embeddings WHERE model = ? AND text_hash = ?",
                (model, _hash_text(text)),
            ).fetchone()
        if row is None:
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        try:
            return json.loads(row["vector"])
        except (TypeError, json.JSONDecodeError):
            return None

    def get_many(self, model: str, texts: Iterable[str]) -> list[list[float] | None]:
        out: list[list[float] | None] = []
        # Берём батчем через ``WHERE text_hash IN (...)``.
        items = list(texts)
        if not items:
            return out
        hashes = [_hash_text(t) for t in items]
        placeholders = ",".join("?" for _ in hashes)
        sql = f"SELECT text_hash, vector FROM embeddings WHERE model = ? AND text_hash IN ({placeholders})"
        with self._cursor() as cur:
            rows = cur.execute(sql, [model, *hashes]).fetchall()
        by_hash = {r["text_hash"]: r["vector"] for r in rows}
        for h in hashes:
            raw = by_hash.get(h)
            if raw is None:
                self.stats.misses += 1
                out.append(None)
                continue
            try:
                out.append(json.loads(raw))
                self.stats.hits += 1
            except (TypeError, json.JSONDecodeError):
                self.stats.misses += 1
                out.append(None)
        return out

    # ---------- write ----------

    def put(self, model: str, text: str, vector: list[float]) -> None:
        self.put_many(model, [text], [vector])

    def put_many(
        self,
        model: str,
        texts: Iterable[str],
        vectors: Iterable[list[float]],
    ) -> None:
        rows: list[tuple] = []
        for text, vec in zip(texts, vectors, strict=False):
            if not vec:
                continue
            rows.append((model, _hash_text(text), len(vec), json.dumps(list(vec))))
        if not rows:
            return
        with self._cursor() as cur:
            cur.executemany(
                "INSERT OR REPLACE INTO embeddings (model, text_hash, dim, vector) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )

    # ---------- introspection ----------

    def count(self, model: str | None = None) -> int:
        with self._cursor() as cur:
            if model is None:
                return cur.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            return cur.execute(
                "SELECT COUNT(*) FROM embeddings WHERE model = ?", (model,)
            ).fetchone()[0]

    def models(self) -> dict[str, int]:
        """Сколько записей хранится по каждой модели."""
        with self._cursor() as cur:
            rows = cur.execute(
                "SELECT model, COUNT(*) AS n FROM embeddings GROUP BY model"
            ).fetchall()
        return {r["model"]: r["n"] for r in rows}

    def clear(self, model: str | None = None) -> int:
        """Удаляет записи. Возвращает число удалённых строк."""
        with self._cursor() as cur:
            if model is None:
                cur.execute("DELETE FROM embeddings")
            else:
                cur.execute("DELETE FROM embeddings WHERE model = ?", (model,))
            return cur.rowcount

    # ---------- lifecycle ----------

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> EmbeddingCache:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# CachedEmbedder — декоратор над любым Embedder
# ---------------------------------------------------------------------------


class CachedEmbedder:
    """Прозрачная обёртка над любым :class:`Embedder`.

    Совместима с протоколом ``Embedder``: проксирует ``embed_passages`` и
    ``embed_query``, попутно читая/записывая ``EmbeddingCache``.

    :param embedder: базовый эмбеддер (``GigaChatEmbedder``, ``E5Embedder``, …).
    :param cache_path: путь к ``embeddings.db`` (создастся при необходимости).
        Если передан готовый ``EmbeddingCache`` — используется он.
    :param model_key: ключ для namespace в БД. По умолчанию выводится из
        ``embedder.model`` или имени класса.
    :param cache_queries: кэшировать ли ``embed_query`` (по умолчанию ``False``,
        т.к. запросы обычно уникальны). Для документов кэш всегда включён.
    """

    def __init__(
        self,
        embedder,
        *,
        cache_path: str | Path | EmbeddingCache = "cache/embeddings.db",
        model_key: str | None = None,
        cache_queries: bool = False,
    ) -> None:
        self.embedder = embedder
        if isinstance(cache_path, EmbeddingCache):
            self.cache: EmbeddingCache = cache_path
        else:
            self.cache = EmbeddingCache(cache_path)
        self.cache_queries = cache_queries
        self.model_key = model_key or getattr(embedder, "model", None) or type(embedder).__name__
        # dim прокидываем, если у embedder'а есть
        self.dim = getattr(embedder, "dim", 0)

    # ---------- pass-through ----------

    def embed_passages(self, texts: Iterable[str]) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []
        cached = self.cache.get_many(self.model_key, items)
        # собираем индексы тех, что нужно посчитать
        missing_idx = [i for i, v in enumerate(cached) if v is None]
        if missing_idx:
            missing_texts = [items[i] for i in missing_idx]
            fresh = self.embedder.embed_passages(missing_texts)
            self.cache.put_many(self.model_key, missing_texts, fresh)
            for idx, vec in zip(missing_idx, fresh, strict=False):
                cached[idx] = vec
        # обновим dim если впервые получили вектор
        if not self.dim and cached and cached[0] is not None:
            self.dim = len(cached[0])
        # тут все элементы не None — protocol требует list[list[float]]
        return [v if v is not None else [] for v in cached]

    def embed_query(self, text: str) -> list[float]:
        if self.cache_queries:
            hit = self.cache.get(self.model_key, text)
            if hit is not None:
                return hit
        vec = self.embedder.embed_query(text)
        if self.cache_queries and vec:
            self.cache.put(self.model_key, text, vec)
        if not self.dim and vec:
            self.dim = len(vec)
        return vec

    # ---------- introspection ----------

    @property
    def stats(self) -> CacheStats:
        return self.cache.stats

    def close(self) -> None:
        self.cache.close()
        for attr in ("close",):
            fn = getattr(self.embedder, attr, None)
            if callable(fn):
                fn()

    def __enter__(self) -> CachedEmbedder:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

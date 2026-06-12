"""Тесты дискового кэша эмбеддингов."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

from web_ai_assistant.cli.cache import main as cache_cli
from web_ai_assistant.embeddings.cache import CachedEmbedder, EmbeddingCache

# ---------------------------------------------------------------------------
# Фейк-эмбеддер для тестов
# ---------------------------------------------------------------------------


class CountingEmbedder:
    """Считает реальные вызовы. Возвращает фиктивные dim=3-векторы по тексту."""

    model = "fake-v1"

    def __init__(self):
        self.passages_calls = 0
        self.queries_calls = 0
        self.dim = 3

    def embed_passages(self, texts):
        items = list(texts)
        self.passages_calls += 1
        return [self._vec(t) for t in items]

    def embed_query(self, text):
        self.queries_calls += 1
        return self._vec(text)

    @staticmethod
    def _vec(t: str) -> list[float]:
        return [float(len(t)), float(ord(t[0]) % 7) if t else 0.0, 0.5]


# ---------------------------------------------------------------------------
# EmbeddingCache: низкоуровневое поведение
# ---------------------------------------------------------------------------


def test_in_memory_cache_starts_empty():
    cache = EmbeddingCache(":memory:")
    assert cache.count() == 0
    assert cache.models() == {}


def test_put_and_get_one():
    cache = EmbeddingCache(":memory:")
    cache.put("M1", "hello", [0.1, 0.2, 0.3])
    assert cache.get("M1", "hello") == [0.1, 0.2, 0.3]


def test_get_returns_none_for_missing():
    cache = EmbeddingCache(":memory:")
    assert cache.get("M1", "missing") is None


def test_get_many_marks_hits_and_misses():
    cache = EmbeddingCache(":memory:")
    cache.put("M1", "a", [1.0, 0.0])
    out = cache.get_many("M1", ["a", "b", "a"])
    assert out[0] == [1.0, 0.0]
    assert out[1] is None
    assert out[2] == [1.0, 0.0]
    assert cache.stats.hits == 2
    assert cache.stats.misses == 1


def test_namespace_isolates_models():
    """Один и тот же текст в разных моделях даёт разные значения."""
    cache = EmbeddingCache(":memory:")
    cache.put("Embeddings", "x", [0.1] * 1024)
    cache.put("EmbeddingsGigaR", "x", [0.5] * 2560)
    a = cache.get("Embeddings", "x")
    b = cache.get("EmbeddingsGigaR", "x")
    assert a is not None and b is not None
    assert len(a) == 1024
    assert len(b) == 2560


def test_clear_by_model():
    cache = EmbeddingCache(":memory:")
    cache.put("A", "x", [1.0])
    cache.put("A", "y", [1.0])
    cache.put("B", "x", [1.0])
    assert cache.clear(model="A") == 2
    assert cache.count() == 1
    assert cache.models() == {"B": 1}


def test_clear_all():
    cache = EmbeddingCache(":memory:")
    cache.put("A", "x", [1.0])
    cache.put("B", "y", [1.0])
    assert cache.clear() == 2
    assert cache.count() == 0


def test_persists_to_disk(tmp_path: Path):
    db = tmp_path / "emb.db"
    with EmbeddingCache(db) as c1:
        c1.put("M", "hello", [0.7, 0.8])
    # повторно открываем — данные сохранены
    with EmbeddingCache(db) as c2:
        assert c2.get("M", "hello") == [0.7, 0.8]


# ---------------------------------------------------------------------------
# CachedEmbedder: декоратор
# ---------------------------------------------------------------------------


def test_cached_embedder_misses_then_hits():
    base = CountingEmbedder()
    cached = CachedEmbedder(base, cache_path=EmbeddingCache(":memory:"))
    texts = ["a", "b", "c"]

    # первый вызов — все miss
    v1 = cached.embed_passages(texts)
    assert base.passages_calls == 1
    assert cached.stats.misses == 3

    # повторный — все hit, embedder не дёргаем
    v2 = cached.embed_passages(texts)
    assert base.passages_calls == 1  # не вырос
    assert cached.stats.hits == 3
    assert v1 == v2


def test_cached_embedder_partial_hit():
    """Если часть текстов уже в кэше — запрашиваем только недостающие."""
    base = CountingEmbedder()
    cached = CachedEmbedder(base, cache_path=EmbeddingCache(":memory:"))
    cached.embed_passages(["a", "b"])
    base.passages_calls = 0  # обнулим счётчик
    cached.embed_passages(["a", "b", "c", "d"])
    # обратились к base только за недостающими: вызов будет, но с 2 текстами
    assert base.passages_calls == 1


def test_cached_embedder_query_default_not_cached():
    """embed_query по умолчанию НЕ кэшируется (запросы уникальны)."""
    base = CountingEmbedder()
    cached = CachedEmbedder(base, cache_path=EmbeddingCache(":memory:"))
    cached.embed_query("same query")
    cached.embed_query("same query")
    assert base.queries_calls == 2


def test_cached_embedder_query_cached_when_enabled():
    base = CountingEmbedder()
    cached = CachedEmbedder(
        base, cache_path=EmbeddingCache(":memory:"), cache_queries=True
    )
    cached.embed_query("same query")
    cached.embed_query("same query")
    assert base.queries_calls == 1


def test_model_key_isolation_via_decorator():
    """Если поменять model_key — кэш не зацепится."""
    base = CountingEmbedder()
    storage = EmbeddingCache(":memory:")
    a = CachedEmbedder(base, cache_path=storage, model_key="model-A")
    a.embed_passages(["hello"])
    # тот же storage, другой namespace — должно быть miss
    base.passages_calls = 0
    b = CachedEmbedder(base, cache_path=storage, model_key="model-B")
    b.embed_passages(["hello"])
    assert base.passages_calls == 1  # сходил в embedder снова


def test_cached_embedder_respects_external_cache_object():
    base = CountingEmbedder()
    storage = EmbeddingCache(":memory:")
    cached = CachedEmbedder(base, cache_path=storage)
    cached.embed_passages(["x", "y"])
    # внешний cache получил записи
    assert storage.count() == 2


# ---------------------------------------------------------------------------
# CLI: webai-cache stats / clear
# ---------------------------------------------------------------------------


def _capture_cli(argv: list[str]) -> str:
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        rc = cache_cli(argv)
    assert rc == 0
    return buf.getvalue()


def test_cli_stats_for_missing_db(tmp_path: Path):
    out = _capture_cli(["--path", str(tmp_path / "nope.db"), "stats"])
    assert "пока пусто" in out


def test_cli_stats_shows_models(tmp_path: Path):
    db = tmp_path / "emb.db"
    cache = EmbeddingCache(db)
    cache.put("Embeddings", "x", [0.1])
    cache.put("Embeddings", "y", [0.2])
    cache.put("EmbeddingsGigaR", "x", [0.3])
    cache.close()
    out = _capture_cli(["--path", str(db), "stats"])
    assert "Всего записей: 3" in out
    assert "Embeddings" in out
    assert "EmbeddingsGigaR" in out


def test_cli_clear_by_model(tmp_path: Path):
    db = tmp_path / "emb.db"
    cache = EmbeddingCache(db)
    cache.put("Embeddings", "x", [0.1])
    cache.put("EmbeddingsGigaR", "y", [0.2])
    cache.close()
    out = _capture_cli(["--path", str(db), "clear", "--model", "Embeddings"])
    assert "Удалено: 1" in out
    # GigaR остался
    cache = EmbeddingCache(db)
    assert cache.count() == 1
    assert cache.models() == {"EmbeddingsGigaR": 1}
    cache.close()


def test_cli_clear_all(tmp_path: Path):
    db = tmp_path / "emb.db"
    cache = EmbeddingCache(db)
    cache.put("Embeddings", "x", [0.1])
    cache.put("Embeddings", "y", [0.2])
    cache.close()
    out = _capture_cli(["--path", str(db), "clear", "--all"])
    assert "Удалено: 2" in out

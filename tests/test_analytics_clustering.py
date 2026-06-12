"""Кластеризация логов."""

from __future__ import annotations

import pytest

pytest.importorskip("sklearn", reason="scikit-learn not installed")


from web_ai_assistant.analytics.clustering import cluster_queries


def _make_rows(vectors):
    return [
        {"id": i + 1, "question": f"q{i}", "embedding": list(v)}
        for i, v in enumerate(vectors)
    ]


def test_kmeans_finds_two_groups():
    # 3 точки около (1,0) и 3 около (0,1) — должно получиться 2 чётких кластера
    rows = _make_rows([
        (1.0, 0.0), (0.95, 0.05), (1.05, -0.05),
        (0.0, 1.0), (0.05, 0.95), (-0.05, 1.05),
    ])
    clusters = cluster_queries(rows, backend="kmeans", k=2)
    assert len(clusters) == 2
    sizes = sorted(c.size for c in clusters)
    assert sizes == [3, 3]


def test_kmeans_representatives_returned():
    rows = _make_rows([(1.0, 0.0), (0.99, 0.01), (-1.0, 0.0)])
    clusters = cluster_queries(rows, backend="kmeans", k=2, representatives_per_cluster=2)
    for c in clusters:
        assert 1 <= len(c.representatives) <= 2
        for r in c.representatives:
            assert "id" in r and "question" in r


def test_empty_input_returns_empty():
    assert cluster_queries([], backend="kmeans", k=3) == []


def test_k_capped_to_n_rows():
    rows = _make_rows([(1.0, 0.0), (0.0, 1.0)])
    # просим 10 кластеров на 2 точках — должно отработать без падений
    clusters = cluster_queries(rows, backend="kmeans", k=10)
    assert len(clusters) <= 2


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        cluster_queries(_make_rows([(1.0, 0.0)]), backend="agglom", k=1)

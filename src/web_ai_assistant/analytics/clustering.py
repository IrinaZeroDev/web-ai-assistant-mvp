"""Кластеризация логов запросов по эмбеддингам.

Поддерживаются два backend:

- ``"kmeans"`` (по умолчанию, как в докладе) — нужно sklearn (он уже в
  зависимостях через sentence-transformers).
- ``"hdbscan"`` — auto-подбор числа кластеров, отдельный пакет ``hdbscan``;
  если не установлен — выбрасывается понятная ошибка.

Для каждого кластера возвращаем:

- размер (число запросов);
- центроид (для kmeans) или плотностной центр (для hdbscan);
- топ-K представителей — запросы, ближайшие к центру кластера.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class ClusterResult:
    label: int                 # id кластера (для kmeans — 0..k-1, для hdbscan может быть -1 = noise)
    size: int                  # сколько запросов попало
    representatives: list[dict] = field(default_factory=list)  # топ-K представителей: {id, question}


def _cosine(a, b) -> float:
    import numpy as np

    na = a / (np.linalg.norm(a) + 1e-12)
    nb = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(na, nb))


def cluster_queries(
    rows: Sequence[dict],
    *,
    backend: str = "kmeans",
    k: int = 8,
    min_cluster_size: int = 3,
    representatives_per_cluster: int = 3,
    random_state: int = 0,
) -> list[ClusterResult]:
    """Кластеризует запросы.

    :param rows: список ``{"id": int, "question": str, "embedding": list[float]}``.
        Обычно это вывод :meth:`QueryStore.all_for_clustering`.
    :param backend: ``"kmeans"`` или ``"hdbscan"``.
    :param k: число кластеров для kmeans (игнорируется для hdbscan).
    :param min_cluster_size: ``min_cluster_size`` для hdbscan.
    :param representatives_per_cluster: сколько ближайших к центру запросов
        возвращать в ``representatives``.
    """
    if not rows:
        return []
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise ImportError("numpy не установлен (нужен для кластеризации)") from exc

    X = np.array([r["embedding"] for r in rows], dtype=np.float32)
    if backend == "kmeans":
        labels, centers = _kmeans(X, k=min(k, len(rows)), random_state=random_state)
    elif backend == "hdbscan":
        labels, centers = _hdbscan(X, min_cluster_size=min_cluster_size)
    else:
        raise ValueError(f"unknown backend: {backend!r}")

    # сборка результатов
    out: list[ClusterResult] = []
    unique = sorted(set(int(label) for label in labels))
    for lbl in unique:
        idx = [i for i, x in enumerate(labels) if int(x) == lbl]
        if not idx:
            continue
        # представители: ближайшие к центру (для noise hdbscan возьмём первые N)
        if lbl >= 0 and lbl < len(centers):
            sims = [_cosine(X[i], centers[lbl]) for i in idx]
            top = sorted(zip(sims, idx, strict=True), key=lambda x: x[0], reverse=True)
            chosen = [j for _s, j in top[:representatives_per_cluster]]
        else:
            chosen = idx[:representatives_per_cluster]
        out.append(
            ClusterResult(
                label=lbl,
                size=len(idx),
                representatives=[
                    {"id": rows[j]["id"], "question": rows[j]["question"]} for j in chosen
                ],
            )
        )
    # сортируем по убыванию размера (топ-N кластеров затруднений)
    out.sort(key=lambda c: c.size, reverse=True)
    return out


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def _kmeans(X, k: int, random_state: int):
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "scikit-learn не установлен — поставьте 'web-ai-assistant[analytics]'"
        ) from exc

    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X)
    return labels, km.cluster_centers_


def _hdbscan(X, min_cluster_size: int):
    try:
        import hdbscan  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "hdbscan не установлен — поставьте 'web-ai-assistant[hdbscan]'"
        ) from exc
    import numpy as np

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = clusterer.fit_predict(X)
    centers = []
    for lbl in sorted({int(label) for label in labels}):
        if lbl < 0:
            continue
        members = X[labels == lbl]
        centers.append(members.mean(axis=0))
    return labels, np.array(centers) if centers else np.empty((0, X.shape[1]))

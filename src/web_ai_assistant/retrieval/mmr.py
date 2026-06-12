"""Maximal Marginal Relevance (MMR) для diversity в top-k retrieval.

MMR (Carbonell & Goldstein, SIGIR 1998) выбирает фрагменты, балансируя
**релевантность** запросу и **новизну** относительно уже выбранных::

    MMR(d_i) = λ · sim(q, d_i)  −  (1 − λ) · max_{d_j ∈ S} sim(d_i, d_j)

где ``S`` — уже выбранное подмножество. На каждом шаге к ``S`` добавляется
кандидат с максимальным ``MMR``.

Параметр ``λ`` ∈ [0, 1]:

- ``λ = 1.0`` — чисто по релевантности (≡ обычный top-k);
- ``λ = 0.0`` — чисто по разнообразию;
- **типично 0.5–0.7** — компромисс, который на больших корпусах PDF
  заметно поднимает context recall за счёт меньшей доли дублей.

Реализация — numpy-only, ``O(n · k)`` по числу кандидатов ``n``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _to_unit(arr: np.ndarray) -> np.ndarray:
    """Row-wise L2-normalization (для cosine = dot product)."""
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.where(norms == 0, 1.0, norms)


def mmr_select(
    query_emb: Sequence[float],
    cand_embs: Sequence[Sequence[float]],
    k: int,
    lambda_mult: float = 0.7,
    cand_sims: Sequence[float] | None = None,
) -> list[int]:
    """Возвращает список индексов кандидатов в MMR-порядке (длина ≤ ``k``).

    Args:
        query_emb: эмбеддинг запроса (cosine-метрика).
        cand_embs: эмбеддинги кандидатов (``n × dim``).
        k: сколько отобрать.
        lambda_mult: вес релевантности (1.0 — без diversity; 0.0 — без релевантности).
        cand_sims: опционально — уже посчитанные ``sim(q, d_i)``. Если не задано,
            считаем из ``query_emb · cand_embs`` (нормализованных).

    Returns:
        Список индексов (от 0 до ``len(cand_embs)-1``) длиной ``min(k, n)``.
    """
    n = len(cand_embs)
    if n == 0 or k <= 0:
        return []
    if not (0.0 <= lambda_mult <= 1.0):
        raise ValueError(f"lambda_mult должен быть в [0, 1], получено: {lambda_mult}")

    cand_arr = _to_unit(np.asarray(cand_embs, dtype=np.float64))

    if cand_sims is None:
        q_arr = _to_unit(np.asarray(query_emb, dtype=np.float64).reshape(1, -1))
        sims_q = (cand_arr @ q_arr.T).ravel()  # (n,)
    else:
        sims_q = np.asarray(cand_sims, dtype=np.float64)
        if sims_q.shape[0] != n:
            raise ValueError(
                f"cand_sims должен иметь длину {n}, получено {sims_q.shape[0]}"
            )

    # Полная матрица попарных сходств кандидатов: (n, n).
    pair_sims = cand_arr @ cand_arr.T

    selected: list[int] = []
    # Для ещё не выбранных кандидатов — текущий max sim к уже выбранным.
    max_sim_to_selected = np.full(n, -np.inf, dtype=np.float64)
    remaining = set(range(n))
    k = min(k, n)

    while len(selected) < k and remaining:
        if not selected:
            # Первый кандидат — чисто по релевантности.
            best = int(np.argmax(sims_q))
        else:
            mmr_scores = np.full(n, -np.inf, dtype=np.float64)
            for i in remaining:
                mmr_scores[i] = (
                    lambda_mult * sims_q[i]
                    - (1.0 - lambda_mult) * max_sim_to_selected[i]
                )
            best = int(np.argmax(mmr_scores))
        selected.append(best)
        remaining.discard(best)
        # Обновляем max_sim_to_selected для оставшихся.
        new_row = pair_sims[best]
        max_sim_to_selected = np.maximum(max_sim_to_selected, new_row)

    return selected


def reorder_by_indices(items: Sequence[Any], indices: Sequence[int]) -> list[Any]:
    """Хелпер: ``[items[i] for i in indices]`` с проверкой границ."""
    return [items[i] for i in indices if 0 <= i < len(items)]

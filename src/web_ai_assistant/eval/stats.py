"""Парные статистические тесты для A vs B.

Для каждой числовой метрики (``max_sim``, ``rerank_top1``, ``latency_s``)
A/B-сравнение строится по парам: A и B отвечают на один и тот же вопрос,
поэтому используем парные тесты:

- **paired Student's t-test** (:func:`scipy.stats.ttest_rel`) — если разности
  приблизительно нормальны.
- **Wilcoxon signed-rank** (:func:`scipy.stats.wilcoxon`) — непараметрическая
  альтернатива, не требует нормальности (рекомендуем для маленьких n или
  тяжёлых хвостов в latency).

Возвращаем оба, плюс величину эффекта (Cohen's d_z для парных) и median diff.
"""

from __future__ import annotations

from statistics import mean, median, pstdev
from typing import Any


def _safe_pairs(a: list[float | None], b: list[float | None]) -> tuple[list[float], list[float]]:
    pa, pb = [], []
    for x, y in zip(a, b, strict=False):
        if x is None or y is None:
            continue
        pa.append(float(x))
        pb.append(float(y))
    return pa, pb


def paired_compare(a_vals: list[float | None], b_vals: list[float | None]) -> dict[str, Any]:
    """Парный t-test + Wilcoxon + величина эффекта.

    Если ``scipy`` не установлен — возвращает только дескриптивы (mean diff,
    median diff, n_pairs) без p-value.
    """
    pa, pb = _safe_pairs(a_vals, b_vals)
    n = len(pa)
    if n == 0:
        return {"n_pairs": 0}
    diffs = [x - y for x, y in zip(pa, pb, strict=False)]
    result: dict[str, Any] = {
        "n_pairs": n,
        "mean_a": round(mean(pa), 4),
        "mean_b": round(mean(pb), 4),
        "mean_diff": round(mean(diffs), 4),
        "median_diff": round(median(diffs), 4),
    }
    sd = pstdev(diffs) if n > 1 else 0.0
    if sd > 0:
        result["cohens_dz"] = round(mean(diffs) / sd, 4)
    try:
        from scipy import stats as _stats
    except ImportError:  # pragma: no cover
        return result
    if n >= 2:
        try:
            t_stat, t_p = _stats.ttest_rel(pa, pb)
            result["t_stat"] = round(float(t_stat), 4)
            result["t_pvalue"] = round(float(t_p), 6)
        except Exception:  # pragma: no cover
            pass
        # Wilcoxon требует n>=1 ненулевых разностей
        if any(d != 0 for d in diffs):
            try:
                w_stat, w_p = _stats.wilcoxon(pa, pb, zero_method="wilcox")
                result["wilcoxon_stat"] = round(float(w_stat), 4)
                result["wilcoxon_pvalue"] = round(float(w_p), 6)
            except Exception:  # pragma: no cover
                pass
    return result

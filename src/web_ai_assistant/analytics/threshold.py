"""Adaptive sim_threshold: подбор порога out-of-corpus по логам.

Идея: в таблице ``queries`` копится распределение ``max_sim`` для отвеченных
и для заблокированных-по-порогу. Это позволяет автоматически найти
оптимальный порог, который отделяет «нашли релевантное» от «ничего
подходящего».

Алгоритмы:

- **otsu** (по умолчанию) — классический метод Otsu для бимодального
  распределения; без зависимостей, не требует меток.
- **gmm** — Gaussian Mixture с двумя компонентами (``scikit-learn``).
  Аккуратнее на сложных распределениях.
- **percentile** — простой fallback: ``P-й`` процентиль по in-corpus.

Если распределение унимодальное (мала разница между методами Otsu и
percentile) — мы автоматически возвращаемся к ``percentile``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Method = Literal["otsu", "gmm", "percentile", "auto"]


@dataclass
class ThresholdSuggestion:
    threshold: float
    method: str
    sample_size: int
    in_corpus_count: int
    out_of_corpus_count: int
    distribution_quality: str   # "bimodal" | "unimodal" | "too_few_samples"
    rationale: str              # короткое объяснение, что увидели
    histogram: list[dict] = field(default_factory=list)  # для UI: [{bin, count}, ...]

    def as_dict(self) -> dict:
        return {
            "threshold": round(self.threshold, 4),
            "method": self.method,
            "sample_size": self.sample_size,
            "in_corpus_count": self.in_corpus_count,
            "out_of_corpus_count": self.out_of_corpus_count,
            "distribution_quality": self.distribution_quality,
            "rationale": self.rationale,
            "histogram": self.histogram,
        }


# ---------------------------------------------------------------------------
# Низкоуровневые алгоритмы
# ---------------------------------------------------------------------------


def _bimodality_coefficient(values: list[float]) -> float:
    """Sarle's bimodality coefficient: ``BC ∈ [0, 1]``.

    BC > 5/9 ≈ 0.555 — довод в пользу бимодальности.
    BC → 0.5 — близко к normal/unimodal.
    Формула: ``BC = (skew² + 1) / (kurt + 3 * (n-1)² / ((n-2)*(n-3)))``.
    """
    n = len(values)
    if n < 4:
        return 0.0
    mean = sum(values) / n
    m2 = sum((v - mean) ** 2 for v in values) / n
    if m2 <= 0:
        return 0.0
    m3 = sum((v - mean) ** 3 for v in values) / n
    m4 = sum((v - mean) ** 4 for v in values) / n
    skew = m3 / (m2**1.5)
    kurt = m4 / (m2**2) - 3.0  # excess kurtosis
    denom = kurt + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    if denom <= 0:
        return 1.0  # вырождение — считаем бимодальным
    bc = (skew * skew + 1.0) / denom
    return max(0.0, min(1.0, bc))


def _otsu_threshold(values: list[float], bins: int = 64) -> tuple[float, float]:
    """Возвращает ``(threshold, bimodality)``.

    ``bimodality`` — Sarle's bimodality coefficient (в [0, 1]). Не зависит
    от порога (это свойство всего распределения). Порог находится по Otsu.
    """
    if not values:
        return 0.0, 0.0
    lo, hi = min(values), max(values)
    if lo == hi:
        return lo, 0.0
    hist = [0] * bins
    width = (hi - lo) / bins
    for v in values:
        idx = min(int((v - lo) / width), bins - 1)
        hist[idx] += 1
    total = len(values)
    s_total = sum((lo + (i + 0.5) * width) * c for i, c in enumerate(hist))
    mean_total = s_total / total
    total_var = (
        sum(((lo + (i + 0.5) * width) - mean_total) ** 2 * c for i, c in enumerate(hist))
        / total
    )
    if total_var <= 0:
        return lo, 0.0
    std_total = total_var**0.5

    s_back = 0.0
    w_back = 0
    best_metric = 0.0
    best_t = lo
    best_diff = 0.0
    best_p_back = 0.0
    best_p_fore = 0.0
    for i in range(bins):
        w_back += hist[i]
        if w_back == 0:
            continue
        w_fore = total - w_back
        if w_fore == 0:
            break
        s_back += (lo + (i + 0.5) * width) * hist[i]
        m_back = s_back / w_back
        m_fore = (s_total - s_back) / w_fore
        between = w_back * w_fore * (m_back - m_fore) ** 2 / (total * total)
        if between > best_metric:
            best_metric = between
            best_t = lo + (i + 1) * width
            best_diff = abs(m_back - m_fore)
            best_p_back = w_back / total
            best_p_fore = w_fore / total
    # Неиспользуемые больше влажные локальные — просто выбрасываем.
    _ = (best_diff, best_p_back, best_p_fore, std_total)
    bc = _bimodality_coefficient(values)
    return best_t, bc


def _gmm_threshold(values: list[float]) -> tuple[float, float]:
    """2-Gaussian threshold через scikit-learn. Возвращает ``(threshold, separation)``."""
    try:
        import numpy as np
        from sklearn.mixture import GaussianMixture
    except ImportError as exc:  # pragma: no cover
        raise ImportError("scikit-learn не установлен (нужен для method='gmm')") from exc

    X = np.array(values, dtype=np.float64).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=0).fit(X)
    means = sorted([float(gmm.means_[0][0]), float(gmm.means_[1][0])])
    sigmas = sorted([
        float(gmm.covariances_[0][0][0]) ** 0.5,
        float(gmm.covariances_[1][0][0]) ** 0.5,
    ])
    threshold = (means[0] + means[1]) / 2.0
    sep = (means[1] - means[0]) / (sum(sigmas) / 2 + 1e-9)
    return threshold, min(sep / 4.0, 1.0)


def _percentile_threshold(values: list[float], pct: float) -> float:
    """``pct``-й перцентиль (например, 5 = P5)."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[idx]


def _histogram(values: list[float], bins: int = 20) -> list[dict]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if lo == hi:
        return [{"bin": lo, "count": len(values)}]
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - lo) / width), bins - 1)
        counts[idx] += 1
    return [{"bin": round(lo + (i + 0.5) * width, 4), "count": c} for i, c in enumerate(counts)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def suggest_threshold(
    in_corpus_sims: list[float],
    out_of_corpus_sims: list[float] | None = None,
    *,
    method: Method = "auto",
    min_sample: int = 30,
    fallback_percentile: float = 5.0,
    bimodal_ratio_threshold: float = 0.555,  # Sarle's BC порог 5/9
) -> ThresholdSuggestion:
    """Подбирает порог по распределению ``max_sim``.

    :param in_corpus_sims: ``max_sim`` запросов с ``blocked IS NULL``.
    :param out_of_corpus_sims: ``max_sim`` запросов, которые упёрлись в
        ``out_of_corpus``. Если переданы — комбинируем для бимодальности.
    :param method: ``auto`` (default) — пробуем Otsu, при унимодальности
        откатываемся на percentile. ``otsu`` / ``gmm`` / ``percentile`` —
        форсируют конкретный метод.
    :param min_sample: меньше — возвращаем с ``too_few_samples``.
    :param fallback_percentile: используется при ``method="percentile"`` и
        как fallback из ``auto``.
    :param bimodal_ratio_threshold: если ``ratio < этого`` — считаем
        распределение унимодальным.
    """
    in_corpus = list(in_corpus_sims or [])
    out_of_corpus = list(out_of_corpus_sims or [])
    combined = in_corpus + out_of_corpus
    histogram = _histogram(combined or in_corpus, bins=20)

    if len(combined) < min_sample:
        # слишком мало данных — возьмём percentile по in_corpus (или нижнюю границу)
        threshold = (
            _percentile_threshold(in_corpus, fallback_percentile)
            if in_corpus
            else 0.55  # дефолт, как в RAGAssistant
        )
        return ThresholdSuggestion(
            threshold=threshold,
            method="percentile (default-fallback)",
            sample_size=len(combined),
            in_corpus_count=len(in_corpus),
            out_of_corpus_count=len(out_of_corpus),
            distribution_quality="too_few_samples",
            rationale=(
                f"Меньше {min_sample} запросов в логах — статистики недостаточно. "
                f"Рекомендую оставить порог по умолчанию (0.55) или дождаться больше данных."
            ),
            histogram=histogram,
        )

    if method == "percentile":
        t = _percentile_threshold(in_corpus, fallback_percentile)
        return ThresholdSuggestion(
            threshold=t,
            method=f"percentile (P{fallback_percentile:g})",
            sample_size=len(combined),
            in_corpus_count=len(in_corpus),
            out_of_corpus_count=len(out_of_corpus),
            distribution_quality="not_evaluated",
            rationale=f"Отсекаем нижние {fallback_percentile:g}% in-corpus запросов.",
            histogram=histogram,
        )

    if method == "gmm":
        t, sep = _gmm_threshold(combined)
        return ThresholdSuggestion(
            threshold=t,
            method="gmm",
            sample_size=len(combined),
            in_corpus_count=len(in_corpus),
            out_of_corpus_count=len(out_of_corpus),
            distribution_quality="bimodal" if sep >= bimodal_ratio_threshold else "unimodal",
            rationale=f"Two-Gaussian fit; separation={sep:.3f}.",
            histogram=histogram,
        )

    # method = "otsu" или "auto"
    t, ratio = _otsu_threshold(combined)
    is_bimodal = ratio >= bimodal_ratio_threshold
    if method == "otsu" or is_bimodal:
        return ThresholdSuggestion(
            threshold=t,
            method="otsu",
            sample_size=len(combined),
            in_corpus_count=len(in_corpus),
            out_of_corpus_count=len(out_of_corpus),
            distribution_quality="bimodal" if is_bimodal else "unimodal",
            rationale=(
                f"Otsu нашёл бимодальное разделение (ratio={ratio:.3f})."
                if is_bimodal
                else (
                    f"Otsu вернул порог при унимодальном распределении (ratio={ratio:.3f}); "
                    "проверьте вручную."
                )
            ),
            histogram=histogram,
        )

    # auto + унимодальное → percentile
    t = _percentile_threshold(in_corpus, fallback_percentile)
    return ThresholdSuggestion(
        threshold=t,
        method=f"percentile (P{fallback_percentile:g}, otsu-fallback)",
        sample_size=len(combined),
        in_corpus_count=len(in_corpus),
        out_of_corpus_count=len(out_of_corpus),
        distribution_quality="unimodal",
        rationale=(
            f"Распределение унимодальное (Otsu ratio={ratio:.3f} < {bimodal_ratio_threshold}). "
            f"Использовал {fallback_percentile:g}-й перцентиль in-corpus."
        ),
        histogram=histogram,
    )

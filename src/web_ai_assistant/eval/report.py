"""Отчёты A/B-сравнения: Markdown (для stdout/PR), JSON (для CI/архив), HTML (для просмотра)."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .metrics import RunResult

# --------------------------------------------------------------------------- #
#                                  MARKDOWN                                   #
# --------------------------------------------------------------------------- #


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _diff_sign(a: Any, b: Any) -> str:
    if a is None or b is None or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return "—"
    d = a - b
    arrow = "▲" if d > 0 else ("▼" if d < 0 else "•")
    return f"{arrow} {d:+.4f}"


def render_markdown(
    a: RunResult,
    b: RunResult,
    stats: dict[str, dict],
    overlap: dict,
    ragas_a: dict | None = None,
    ragas_b: dict | None = None,
) -> str:
    lines: list[str] = []
    lines.append(f"# A/B eval: **{a.name}** vs **{b.name}**\n")
    lines.append(f"Размер выборки: **n = {a.aggregate.get('n', 0)}**\n")

    # ----- быстрые метрики -----
    lines.append("## Быстрые метрики (custom)\n")
    lines.append("| Метрика | A | B | Δ (A − B) |")
    lines.append("|---|---:|---:|---:|")
    keys = ["refusal_rate", "refusal_accuracy", "mean_max_sim", "mean_rerank_score", "mean_latency_s"]
    for k in keys:
        va = a.aggregate.get(k)
        vb = b.aggregate.get(k)
        if va is None and vb is None:
            continue
        lines.append(f"| `{k}` | {_fmt(va)} | {_fmt(vb)} | {_diff_sign(va, vb)} |")
    lines.append("")

    # ----- парные тесты -----
    if stats:
        lines.append("## Парные тесты (A vs B, по тем же вопросам)\n")
        lines.append("| Метрика | n | mean_A | mean_B | Δ̄ | median Δ | Cohen's dₐ | t p-value | Wilcoxon p-value |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for metric, st in stats.items():
            lines.append(
                "| `{m}` | {n} | {ma} | {mb} | {md} | {medd} | {d} | {tp} | {wp} |".format(
                    m=metric,
                    n=st.get("n_pairs", 0),
                    ma=_fmt(st.get("mean_a")),
                    mb=_fmt(st.get("mean_b")),
                    md=_fmt(st.get("mean_diff")),
                    medd=_fmt(st.get("median_diff")),
                    d=_fmt(st.get("cohens_dz")),
                    tp=_fmt(st.get("t_pvalue")),
                    wp=_fmt(st.get("wilcoxon_pvalue")),
                )
            )
        lines.append("")

    # ----- overlap -----
    if overlap and overlap.get("mean_jaccard") is not None:
        lines.append(
            f"## Overlap источников\n\nMean Jaccard top-K url's: **{overlap['mean_jaccard']:.4f}** "
            f"(на {overlap['pairs']} парах)\n"
        )

    # ----- RAGAS -----
    if ragas_a or ragas_b:
        lines.append("## RAGAS (LLM-judge)\n")
        all_keys = sorted(set((ragas_a or {}).keys()) | set((ragas_b or {}).keys()))
        if all_keys:
            lines.append("| Метрика | A | B | Δ |")
            lines.append("|---|---:|---:|---:|")
            for k in all_keys:
                va = (ragas_a or {}).get(k)
                vb = (ragas_b or {}).get(k)
                lines.append(f"| `{k}` | {_fmt(va)} | {_fmt(vb)} | {_diff_sign(va, vb)} |")
            lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#                                    JSON                                     #
# --------------------------------------------------------------------------- #


def render_json(
    a: RunResult,
    b: RunResult,
    stats: dict,
    overlap: dict,
    ragas_a: dict | None,
    ragas_b: dict | None,
) -> dict:
    return {
        "a": {"name": a.name, "aggregate": a.aggregate, "ragas": ragas_a or {}},
        "b": {"name": b.name, "aggregate": b.aggregate, "ragas": ragas_b or {}},
        "paired_stats": stats,
        "source_overlap": overlap,
        "per_item": [
            {
                "question": ra["question"],
                "a": {
                    "blocked": ra["blocked"],
                    "max_sim": ra["max_sim"],
                    "rerank_top1": ra["rerank_top1"],
                    "latency_s": ra["latency_s"],
                },
                "b": {
                    "blocked": rb["blocked"],
                    "max_sim": rb["max_sim"],
                    "rerank_top1": rb["rerank_top1"],
                    "latency_s": rb["latency_s"],
                },
            }
            for ra, rb in zip(a.per_item, b.per_item, strict=False)
        ],
    }


# --------------------------------------------------------------------------- #
#                                    HTML                                     #
# --------------------------------------------------------------------------- #


_HTML_CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 980px;
       margin: 32px auto; padding: 0 16px; color: #1f2328; }
h1, h2 { border-bottom: 1px solid #d0d7de; padding-bottom: 6px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; }
th, td { border: 1px solid #d0d7de; padding: 6px 10px; font-size: 14px; }
th { background: #f6f8fa; text-align: left; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.bar-wrap { display: flex; align-items: center; gap: 8px; }
.bar-bg { flex: 1; background: #eaeef2; height: 14px; border-radius: 3px; overflow: hidden; }
.bar-a { background: #218bff; height: 14px; }
.bar-b { background: #cf222e; height: 14px; }
.legend { font-size: 13px; color: #57606a; margin: 4px 0 18px; }
.swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }
.swatch.a { background: #218bff; }
.swatch.b { background: #cf222e; }
.muted { color: #57606a; }
.pos { color: #1a7f37; font-weight: 600; }
.neg { color: #cf222e; font-weight: 600; }
"""


def _bar_pair(va: float | None, vb: float | None, vmax: float | None = None) -> str:
    if va is None and vb is None:
        return '<span class="muted">—</span>'
    vmax = vmax or max((va or 0), (vb or 0), 1e-9)
    pa = (va or 0) / vmax * 100
    pb = (vb or 0) / vmax * 100
    return (
        f'<div class="bar-wrap" title="A={va} B={vb}">'
        f'<div class="bar-bg" style="width:60px"><div class="bar-a" style="width:{pa:.1f}%"></div></div>'
        f'<div class="bar-bg" style="width:60px"><div class="bar-b" style="width:{pb:.1f}%"></div></div>'
        f"</div>"
    )


def _delta_html(va: Any, vb: Any) -> str:
    if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
        return '<span class="muted">—</span>'
    d = va - vb
    cls = "pos" if d > 0 else ("neg" if d < 0 else "muted")
    return f'<span class="{cls}">{d:+.4f}</span>'


def render_html(
    a: RunResult,
    b: RunResult,
    stats: dict,
    overlap: dict,
    ragas_a: dict | None = None,
    ragas_b: dict | None = None,
) -> str:
    a_name = html.escape(a.name)
    b_name = html.escape(b.name)
    n = a.aggregate.get("n", 0)

    rows_fast = []
    keys = ["refusal_rate", "refusal_accuracy", "mean_max_sim", "mean_rerank_score", "mean_latency_s"]
    for k in keys:
        va = a.aggregate.get(k)
        vb = b.aggregate.get(k)
        if va is None and vb is None:
            continue
        rows_fast.append(
            f"<tr><td><code>{k}</code></td>"
            f'<td class="num">{_fmt(va)}</td>'
            f'<td class="num">{_fmt(vb)}</td>'
            f'<td class="num">{_delta_html(va, vb)}</td>'
            f"<td>{_bar_pair(va, vb)}</td></tr>"
        )

    rows_stats = []
    for metric, st in (stats or {}).items():
        rows_stats.append(
            "<tr><td><code>{m}</code></td>"
            '<td class="num">{n}</td>'
            '<td class="num">{ma}</td><td class="num">{mb}</td>'
            '<td class="num">{md}</td><td class="num">{medd}</td>'
            '<td class="num">{d}</td>'
            '<td class="num">{tp}</td><td class="num">{wp}</td></tr>'.format(
                m=metric,
                n=st.get("n_pairs", 0),
                ma=_fmt(st.get("mean_a")),
                mb=_fmt(st.get("mean_b")),
                md=_fmt(st.get("mean_diff")),
                medd=_fmt(st.get("median_diff")),
                d=_fmt(st.get("cohens_dz")),
                tp=_fmt(st.get("t_pvalue")),
                wp=_fmt(st.get("wilcoxon_pvalue")),
            )
        )

    ragas_rows = ""
    if ragas_a or ragas_b:
        all_keys = sorted(set((ragas_a or {}).keys()) | set((ragas_b or {}).keys()))
        ragas_rows = "<h2>RAGAS (LLM-judge)</h2><table><tr><th>Метрика</th><th>A</th><th>B</th><th>Δ</th></tr>"
        for k in all_keys:
            va = (ragas_a or {}).get(k)
            vb = (ragas_b or {}).get(k)
            ragas_rows += (
                f"<tr><td><code>{html.escape(k)}</code></td>"
                f'<td class="num">{_fmt(va)}</td>'
                f'<td class="num">{_fmt(vb)}</td>'
                f'<td class="num">{_delta_html(va, vb)}</td></tr>'
            )
        ragas_rows += "</table>"

    overlap_block = ""
    if overlap and overlap.get("mean_jaccard") is not None:
        overlap_block = (
            f"<h2>Overlap источников</h2>"
            f'<p>Mean Jaccard top-K url\'s: <b>{overlap["mean_jaccard"]:.4f}</b> '
            f'(на {overlap["pairs"]} парах)</p>'
        )

    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>A/B eval: {a_name} vs {b_name}</title>
<style>{_HTML_CSS}</style></head><body>
<h1>A/B eval: {a_name} vs {b_name}</h1>
<p class="muted">Размер выборки: <b>n = {n}</b></p>
<p class="legend"><span class="swatch a"></span>A &nbsp; <span class="swatch b"></span>B</p>

<h2>Быстрые метрики (custom)</h2>
<table>
<tr><th>Метрика</th><th>A</th><th>B</th><th>Δ (A − B)</th><th>график</th></tr>
{"".join(rows_fast)}
</table>

<h2>Парные тесты (A vs B)</h2>
<table>
<tr><th>Метрика</th><th>n</th><th>mean_A</th><th>mean_B</th><th>Δ̄</th><th>median Δ</th>
<th>Cohen's d<sub>z</sub></th><th>t p-value</th><th>Wilcoxon p-value</th></tr>
{"".join(rows_stats)}
</table>

{overlap_block}
{ragas_rows}
</body></html>
"""


# --------------------------------------------------------------------------- #
#                                  write API                                  #
# --------------------------------------------------------------------------- #


def write_reports(
    a: RunResult,
    b: RunResult,
    stats: dict,
    overlap: dict,
    *,
    out_md: str | Path | None = None,
    out_json: str | Path | None = None,
    out_html: str | Path | None = None,
    ragas_a: dict | None = None,
    ragas_b: dict | None = None,
) -> dict[str, str]:
    """Сохраняет выбранные форматы и возвращает paths."""
    paths: dict[str, str] = {}
    if out_md:
        md = render_markdown(a, b, stats, overlap, ragas_a, ragas_b)
        Path(out_md).write_text(md, encoding="utf-8")
        paths["md"] = str(out_md)
    if out_json:
        data = render_json(a, b, stats, overlap, ragas_a, ragas_b)
        Path(out_json).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        paths["json"] = str(out_json)
    if out_html:
        h = render_html(a, b, stats, overlap, ragas_a, ragas_b)
        Path(out_html).write_text(h, encoding="utf-8")
        paths["html"] = str(out_html)
    return paths

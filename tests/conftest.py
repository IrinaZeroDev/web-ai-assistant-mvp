"""Общие фикстуры. Генерируем тестовые PDF в-памяти без внешних файлов.

Тестовые PDF используют латиницу (DSTU — Don State Technical University) —
это снимает зависимость от наличия Cyrillic-TTF в CI. Эвристики
заголовков/колонтитулов языко-независимы.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _font_path() -> str | None:
    """Возвращает путь к Unicode-TTF, если такой есть в системе или fpdf."""
    import fpdf as _fpdf

    candidates = [
        Path(_fpdf.__file__).parent / "font" / "DejaVuSansCondensed.ttf",
        Path(_fpdf.__file__).parent / "font" / "DejaVuSans.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf"),
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def _build_pdf(path: Path, pages: list[str]) -> None:
    """Один элемент списка = одна страница; переводы строк сохраняются."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    fp = _font_path()
    if fp:
        pdf.add_font("DejaVu", "", fp)
        pdf.set_font("DejaVu", size=12)
    else:
        pdf.set_font("helvetica", size=12)

    for page_text in pages:
        pdf.add_page()
        # явная ширина + WRAP_CHAR — не падаём на длинных URL без пробелов
        for line in page_text.splitlines():
            pdf.multi_cell(w=180, h=7, text=line, new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(path))


@pytest.fixture
def sample_methodichka(tmp_path: Path) -> Path:
    """PDF с заголовками, повторяющимися колонтитулами, номерами страниц."""
    pages = [
        # стр. 1 — титул
        "IST DSTU Department\nMETODICHESKOE POSOBIE\nweb development for bachelors\n\n1",
        # стр. 2 — глава 1
        "IST DSTU Department\nGlava 1. HTML basics\nHTML is hypertext markup language.\n"
        "Used for structuring web pages.\n\n2",
        # стр. 3 — глава 2
        "IST DSTU Department\nGlava 2. CSS Flexbox\nFlexbox is a one-dimensional layout model.\n"
        "display: flex; makes container a flex container.\n\n3",
        # стр. 4 — литература
        "IST DSTU Department\nLiteratura\n1. MDN Web Docs. https://developer.mozilla.org\n\n4",
    ]
    out = tmp_path / "methodichka_ist.pdf"
    _build_pdf(out, pages)
    return out


@pytest.fixture
def pdf_dir(tmp_path: Path) -> Path:
    """Директория с двумя PDF — для проверки batch-загрузки (в т.ч. рекурсивной)."""
    d = tmp_path / "pdfs"
    d.mkdir()
    _build_pdf(
        d / "ist_html.pdf",
        ["Glava 1. HTML\nTags define structure.\n\n1"],
    )
    sub = d / "subdir"
    sub.mkdir()
    _build_pdf(
        sub / "ist_css.pdf",
        ["Glava 1. CSS\nCascading style sheets.\n\n1"],
    )
    return d

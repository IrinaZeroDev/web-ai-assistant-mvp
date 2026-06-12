"""Тесты PDF-loader'а. Генерируем мини-PDF в fixture (см. conftest.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pdfminer", reason="pdfminer.six not installed")
pytest.importorskip("fpdf", reason="fpdf2 not installed (dev dep)")

from web_ai_assistant.corpus import Document, load_pdf, load_pdf_corpus, split_documents
from web_ai_assistant.loaders.pdf import (
    _looks_like_heading,
    _mark_headings,
    _normalize,
    _strip_page_numbers,
    _strip_running_headers,
)

# ---------------------------------------------------------------------------
# Юнит-тесты эвристик
# ---------------------------------------------------------------------------


def test_normalize_collapses_ligatures():
    assert _normalize("eﬃcient ﬁle") == "efficient file"


def test_normalize_joins_hyphenated_words():
    # латиница и кириллица работают одинаково
    assert _normalize("trans-\nport") == "transport"
    assert _normalize("транс-\nпорт") == "транспорт"


def test_strip_page_numbers():
    text = "Content\n14\np. 15\n- 16 -\nстр. 17\nMore content"
    cleaned = _strip_page_numbers(text)
    assert "14" not in cleaned.split("\n")
    assert "p. 15" not in cleaned
    assert "- 16 -" not in cleaned
    assert "стр. 17" not in cleaned
    assert "Content" in cleaned and "More content" in cleaned


def test_strip_running_headers_removes_repeated_lines():
    pages = [
        "IST DSTU Department\nGlava 1\nText",
        "IST DSTU Department\nGlava 2\nOther text",
        "IST DSTU Department\nGlava 3\nMore text",
    ]
    out = _strip_running_headers(pages)
    for p in out:
        assert "IST DSTU Department" not in p
    assert "Glava 1" in out[0]
    assert "Glava 2" in out[1]


def test_looks_like_heading():
    assert _looks_like_heading("Glava 1. Introduction")
    assert _looks_like_heading("Глава 1. Введение")
    assert _looks_like_heading("1.2 Basic concepts")
    assert _looks_like_heading("CONCLUSION OF THE WORK")
    assert _looks_like_heading("References")
    assert _looks_like_heading("Literatura")
    assert not _looks_like_heading("This is a regular sentence describing the method.")
    assert not _looks_like_heading("HTML")  # одно слово — аббревиатура


def test_mark_headings_wraps_with_markers():
    text = "Introduction\nThis is the intro text.\nGlava 1. HTML\nTags structure the document."
    marked = _mark_headings(text)
    assert "=== Introduction ===" in marked
    assert "=== Glava 1. HTML ===" in marked
    assert "This is the intro text." in marked


# ---------------------------------------------------------------------------
# Интеграционные: реальная генерация PDF и парсинг
# ---------------------------------------------------------------------------


def test_load_pdf_returns_document(sample_methodichka: Path):
    doc = load_pdf(sample_methodichka, ocr_fallback=False)
    assert isinstance(doc, Document)
    assert doc.doc_id == sample_methodichka.stem
    assert doc.title == sample_methodichka.stem
    assert doc.url.startswith("file://")
    assert "HTML" in doc.text
    assert "Flexbox" in doc.text


def test_load_pdf_strips_running_header(sample_methodichka: Path):
    """Колонтитул 'IST DSTU Department' должен быть удалён (на всех 4 страницах)."""
    doc = load_pdf(sample_methodichka, ocr_fallback=False)
    assert "IST DSTU Department" not in doc.text


def test_load_pdf_strips_page_numbers(sample_methodichka: Path):
    doc = load_pdf(sample_methodichka, ocr_fallback=False)
    # пагинация 1..4 как отдельные строки не должна остаться
    lines = [ln.strip() for ln in doc.text.splitlines()]
    assert "1" not in lines
    assert "2" not in lines


def test_load_pdf_marks_headings(sample_methodichka: Path):
    doc = load_pdf(sample_methodichka, ocr_fallback=False)
    assert "=== " in doc.text and " ===" in doc.text
    # узнаваемые заголовки
    assert "Glava 1" in doc.text
    assert "Literatura" in doc.text


def test_load_pdf_can_disable_heading_detection(sample_methodichka: Path):
    doc = load_pdf(sample_methodichka, detect_headings=False, ocr_fallback=False)
    assert "=== " not in doc.text


# ---------------------------------------------------------------------------
# load_pdf_corpus: path | dir | list
# ---------------------------------------------------------------------------


def test_load_corpus_from_single_file(sample_methodichka: Path):
    docs = load_pdf_corpus(sample_methodichka, ocr_fallback=False)
    assert len(docs) == 1
    assert docs[0].doc_id == sample_methodichka.stem


def test_load_corpus_from_directory(pdf_dir: Path):
    docs = load_pdf_corpus(pdf_dir, ocr_fallback=False)
    assert len(docs) == 2  # рекурсивно подхватил subdir/ist_css.pdf
    ids = {d.doc_id for d in docs}
    assert ids == {"ist_html", "ist_css"}


def test_load_corpus_non_recursive(pdf_dir: Path):
    docs = load_pdf_corpus(pdf_dir, recursive=False, ocr_fallback=False)
    assert {d.doc_id for d in docs} == {"ist_html"}


def test_load_corpus_from_list(sample_methodichka: Path, pdf_dir: Path):
    docs = load_pdf_corpus([sample_methodichka, pdf_dir / "ist_html.pdf"], ocr_fallback=False)
    assert len(docs) == 2


def test_load_corpus_raises_for_missing_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_pdf_corpus(tmp_path / "nonexistent.pdf")


def test_load_corpus_skips_unreadable_files(tmp_path: Path, sample_methodichka: Path):
    """Битый PDF не валит весь батч — он логируется и пропускается."""
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    docs = load_pdf_corpus([sample_methodichka, bad], ocr_fallback=False)
    assert len(docs) == 1  # битый пропущен
    assert docs[0].doc_id == sample_methodichka.stem


# ---------------------------------------------------------------------------
# Интеграция со splitter'ом — главное обещание: ломаем по главам
# ---------------------------------------------------------------------------


def test_splitter_breaks_on_heading_markers():
    pytest.importorskip("langchain", reason="langchain not installed")
    """Splitter должен в первую очередь делить по '\\n=== ... ==='."""
    text = (
        "=== Chapter 1 ===\n"
        + ("Text about HTML. " * 30)
        + "\n=== Chapter 2 ===\n"
        + ("Text about CSS. " * 30)
    )
    doc = Document(doc_id="t", url="x", title="T", text=text)
    chunks = split_documents([doc], chunk_size=400, chunk_overlap=50)
    # Внутри одного чанка не должно одновременно встречаться два заголовка.
    for c in chunks:
        assert not ("Chapter 1" in c.text and "Chapter 2" in c.text), (
            "splitter не должен слить две главы в один чанк"
        )

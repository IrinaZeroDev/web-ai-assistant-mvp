"""Загрузчик PDF-методичек.

Логика:

1. Извлечение текста pdfminer.six по страницам — сохраняет порядок.
2. Если страница пустая (скан без текстового слоя) — опциональный OCR через
   ``pytesseract`` (нужен extra ``[ocr]``). Без OCR — страница пропускается
   с предупреждением.
3. Эвристики очистки:
   - удаляются повторяющиеся колонтитулы (одинаковая строка ≥ 50% страниц);
   - удаляются строки, состоящие только из номеров страниц / "стр. N";
   - схлопываются лигатуры (ﬁ → fi и т.п.);
   - переносы вида "транс-\\nпорт" склеиваются в "транспорт".
4. Заголовки распознаются эвристикой и оборачиваются маркерами
   ``\\n=== <title> ===\\n``. Эти маркеры — высший приоритет для
   ``split_documents`` (см. ``corpus.split_documents``).

Главная функция — :func:`load_pdf_corpus`, она же экспортируется на уровне
``web_ai_assistant.corpus.load_pdf_corpus``.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from ..corpus import Document

log = logging.getLogger(__name__)

# Маркер, на котором splitter будет ломать чанки в первую очередь.
HEADING_OPEN = "\n=== "
HEADING_CLOSE = " ===\n"

# Лигатуры → ASCII-эквиваленты (типично для научных PDF).
_LIGATURES = str.maketrans(
    {
        "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
        "\u00ad": "",   # soft hyphen
        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u00a0": " ",  # non-breaking space
    }
)

# Чисто цифровая строка (или "стр. 14", "- 14 -") = вероятный номер страницы.
_PAGENUM_RE = re.compile(
    r"^\s*(?:[-—–]?\s*(?:стр\.?|page|p\.)?\s*\d{1,4}\s*[-—–]?\s*)$",
    re.IGNORECASE,
)

# Эвристика заголовка: короткая строка, либо начинается со стандартного префикса
# ("Глава 1", "1.2 Введение"), либо в верхнем регистре и < 90 символов.
_HEADING_PREFIX_RE = re.compile(
    r"^\s*("
    r"глава\s+\d+|"
    r"раздел\s+\d+|"
    r"chapter\s+\d+|"
    r"glava\s+\d+|"                  # транслит для тестов / PDF без cyrillic-шрифта
    r"\d+(?:\.\d+){0,3}\.?\s+\S+|"   # 1.2.3 Заголовок
    r"введение|заключение|приложение|литература|содержание|"
    r"introduction|conclusion|references|appendix|contents|literatura"
    r")\b",
    re.IGNORECASE,
)

# Перенос слова: "транс-\nпорт" → "транспорт". Только если справа \n и слово
# продолжается строчной буквой (избегаем склейки "10-е" и подобного).
_HYPHENATION_RE = re.compile(r"(\w)-\n([a-zа-яё])", re.UNICODE)


# ---------------------------------------------------------------------------
# Низкоуровневое извлечение страниц
# ---------------------------------------------------------------------------


def _extract_pages_pdfminer(path: Path) -> list[str]:
    """Возвращает список текстов страниц через pdfminer.six."""
    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTTextContainer
    except ImportError as exc:
        raise ImportError(
            "pdfminer.six не установлен. Поставьте: pip install 'web-ai-assistant[pdf]'"
        ) from exc

    pages: list[str] = []
    for layout in extract_pages(str(path)):
        chunks = []
        for el in layout:
            if isinstance(el, LTTextContainer):
                chunks.append(el.get_text())
        pages.append("".join(chunks))
    return pages


def _ocr_page(path: Path, page_no: int, dpi: int = 200, lang: str = "rus+eng") -> str:
    """OCR одной страницы через Tesseract. Возвращает '' если не получилось."""
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError:
        log.warning(
            "Страница %s в %s — без текстового слоя. Для OCR поставьте: "
            "pip install 'web-ai-assistant[ocr]' и установите Tesseract.",
            page_no,
            path.name,
        )
        return ""
    try:
        images = convert_from_path(
            str(path), dpi=dpi, first_page=page_no, last_page=page_no
        )
        if not images:
            return ""
        return pytesseract.image_to_string(images[0], lang=lang)
    except Exception as exc:  # pragma: no cover - инфраструктура
        log.warning("OCR не удался для %s, страница %s: %s", path.name, page_no, exc)
        return ""


# ---------------------------------------------------------------------------
# Очистка и заголовки
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Лигатуры, NBSP, перенос с дефисом."""
    text = text.translate(_LIGATURES)
    text = _HYPHENATION_RE.sub(r"\1\2", text)
    return text


def _strip_running_headers(pages: list[str]) -> list[str]:
    """Удаляет строки, повторяющиеся в ≥50% страниц (колонтитулы)."""
    if len(pages) < 3:
        return pages
    cnt: Counter[str] = Counter()
    for p in pages:
        seen_on_page = set()
        for line in p.splitlines():
            line = line.strip()
            if len(line) < 4 or len(line) > 120:
                continue
            seen_on_page.add(line)
        cnt.update(seen_on_page)
    threshold = max(2, len(pages) // 2)
    running = {line for line, n in cnt.items() if n >= threshold}
    if not running:
        return pages
    out = []
    for p in pages:
        kept = [ln for ln in p.splitlines() if ln.strip() not in running]
        out.append("\n".join(kept))
    return out


def _strip_page_numbers(text: str) -> str:
    """Удаляет строки-номера страниц."""
    return "\n".join(ln for ln in text.splitlines() if not _PAGENUM_RE.match(ln))


def _looks_like_heading(line: str) -> bool:
    """True если строка похожа на заголовок главы/раздела."""
    s = line.strip()
    if not s or len(s) > 120:
        return False
    if _HEADING_PREFIX_RE.match(s):
        return True
    # короткая строка в верхнем регистре, без точки в конце — вероятный заголовок
    if 4 < len(s) < 90 and s == s.upper() and s[-1] not in ".,;:":
        # отсекаем строки-аббревиатуры: должна содержать пробел или хотя бы 2 слова
        if " " in s and any(ch.isalpha() for ch in s):
            return True
    return False


def _mark_headings(text: str) -> str:
    """Заменяет строки-заголовки на ``=== Заголовок ===``."""
    out: list[str] = []
    for line in text.splitlines():
        if _looks_like_heading(line):
            title = line.strip().rstrip(".")
            out.append("")
            out.append(f"=== {title} ===")
            out.append("")
        else:
            out.append(line)
    return "\n".join(out)


def _clean(text: str) -> str:
    """Финальная нормализация пробелов."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------


def _resolve_sources(
    source: str | Path | Iterable[str | Path],
    recursive: bool = True,
) -> list[Path]:
    """Раскрывает source в плоский список .pdf-файлов."""
    if isinstance(source, (str, Path)):
        p = Path(source).expanduser()
        if p.is_dir():
            pattern = "**/*.pdf" if recursive else "*.pdf"
            return sorted(p.glob(pattern))
        if p.is_file():
            return [p]
        raise FileNotFoundError(f"PDF source not found: {p}")
    # iterable из путей
    out: list[Path] = []
    for s in source:
        out.extend(_resolve_sources(s, recursive=recursive))
    return out


def load_pdf(
    path: str | Path,
    *,
    title: str | None = None,
    doc_id: str | None = None,
    url: str | None = None,
    detect_headings: bool = True,
    ocr_fallback: bool = True,
    ocr_lang: str = "rus+eng",
) -> Document:
    """Загружает одну PDF-методичку и возвращает :class:`Document`.

    Текст одного документа склеивается воедино, но заголовки оборачиваются
    маркерами ``=== ... ===`` — :func:`split_documents` будет резать чанки
    по ним в первую очередь.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(p)

    pages = _extract_pages_pdfminer(p)

    # OCR для пустых страниц (опционально)
    for i, page_text in enumerate(pages, start=1):
        if page_text.strip():
            continue
        if ocr_fallback:
            pages[i - 1] = _ocr_page(p, page_no=i, lang=ocr_lang)
        else:
            log.warning("Страница %s в %s пустая (нет текстового слоя)", i, p.name)

    pages = [_normalize(t) for t in pages]
    pages = _strip_running_headers(pages)
    pages = [_strip_page_numbers(t) for t in pages]

    text = "\n\n".join(pages)
    if detect_headings:
        text = _mark_headings(text)
    text = _clean(text)

    return Document(
        doc_id=doc_id or p.stem,
        url=url or p.resolve().as_uri(),
        title=title or p.stem,
        text=text,
    )


def load_pdf_corpus(
    source: str | Path | Iterable[str | Path],
    *,
    recursive: bool = True,
    detect_headings: bool = True,
    ocr_fallback: bool = True,
    ocr_lang: str = "rus+eng",
) -> list[Document]:
    """Загружает все PDF из переданного источника.

    :param source: путь к файлу, к директории (рекурсивно по умолчанию) или
        итерабль путей.
    :param recursive: если ``source`` — директория, искать ли в подпапках.
    :param detect_headings: эвристически распознавать заголовки и помечать
        их маркерами для splitter'а.
    :param ocr_fallback: использовать Tesseract для страниц без текстового
        слоя (extra ``[ocr]``).
    :param ocr_lang: язык для Tesseract (по умолчанию ``rus+eng``).

    Пример::

        from web_ai_assistant.corpus import load_pdf_corpus, split_documents
        docs = load_pdf_corpus("~/methods/ist")
        chunks = split_documents(docs)
    """
    paths = _resolve_sources(source, recursive=recursive)
    out: list[Document] = []
    for p in paths:
        try:
            out.append(
                load_pdf(
                    p,
                    detect_headings=detect_headings,
                    ocr_fallback=ocr_fallback,
                    ocr_lang=ocr_lang,
                )
            )
        except Exception as exc:
            log.warning("Не удалось загрузить %s: %s", p, exc)
    return out

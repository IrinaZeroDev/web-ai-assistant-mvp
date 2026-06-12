"""Загрузчики разнотипных источников в общий формат ``Document``.

Доступные загрузчики:

- :func:`web_ai_assistant.loaders.pdf.load_pdf_corpus` — PDF-методички
  (pdfminer.six + опциональный OCR через Tesseract).
"""

from __future__ import annotations

__all__ = ["load_pdf_corpus"]


def __getattr__(name: str):
    if name == "load_pdf_corpus":
        from .pdf import load_pdf_corpus

        return load_pdf_corpus
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

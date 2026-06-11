"""Сбор учебного корпуса и чанкование.

Параметры чанкования из ``project_plan.md``:
- chunk≈500 токенов (~900 символов для англ.)
- overlap 50–80 (используем 120 для большей связности)
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# Источники для MVP — фрагмент MDN (CC-BY-SA 2.5).
DEFAULT_MDN_PAGES: list[tuple[str, str]] = [
    ("CSS/flexbox",   "https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_flexible_box_layout/Basic_concepts_of_flexbox"),
    ("CSS/grid",      "https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_grid_layout/Basic_concepts_of_grid_layout"),
    ("CSS/selectors", "https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_selectors"),
    ("CSS/syntax",    "https://developer.mozilla.org/en-US/docs/Web/CSS/Syntax"),
    ("HTML/semantics","https://developer.mozilla.org/en-US/docs/Glossary/Semantics"),
    ("HTML/forms",    "https://developer.mozilla.org/en-US/docs/Learn/Forms/Your_first_form"),
    ("JS/variables",  "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Grammar_and_types"),
    ("JS/functions",  "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Functions"),
    ("JS/promises",   "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Using_promises"),
]

HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (web-ai-assistant-mvp, IST DGTU, educational)"}


@dataclass
class Document:
    doc_id: str
    url: str
    title: str
    text: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    url: str
    text: str
    meta: dict = field(default_factory=dict)


def fetch_mdn(url: str, timeout: int = 20) -> str:
    """Скачивает страницу MDN и возвращает очищенный текст."""
    r = requests.get(url, headers=HTTP_HEADERS, timeout=timeout)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    main = soup.find("main") or soup
    for tag in main.select("nav, aside, .interactive-example, script, style"):
        tag.decompose()
    text = main.get_text("\n", strip=True)
    return re.sub(r"\n{3,}", "\n\n", text)


def load_mdn_corpus(
    pages: Iterable[tuple[str, str]] = DEFAULT_MDN_PAGES,
    delay: float = 1.0,
) -> list[Document]:
    """Скачивает список страниц и возвращает документы."""
    docs: list[Document] = []
    for doc_id, url in pages:
        try:
            body = fetch_mdn(url)
        except Exception as exc:  # pragma: no cover - сеть
            print(f"skip {doc_id}: {exc}")
            continue
        docs.append(Document(doc_id=doc_id, url=url, title=doc_id, text=body))
        time.sleep(delay)
    return docs


def save_corpus(docs: list[Document], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d.__dict__, ensure_ascii=False) + "\n")


def load_corpus(path: str | Path) -> list[Document]:
    docs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            docs.append(Document(**json.loads(line)))
    return docs


def split_documents(
    docs: list[Document],
    chunk_size: int = 900,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """Разрезает документы на чанки через LangChain RecursiveCharacterTextSplitter."""
    # импорт внутри, чтобы пакет грузился даже без langchain (для лёгких тестов)
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []
    for d in docs:
        for i, piece in enumerate(splitter.split_text(d.text)):
            chunks.append(
                Chunk(
                    chunk_id=f"{d.doc_id}#{i}",
                    doc_id=d.doc_id,
                    title=d.title,
                    url=d.url,
                    text=piece,
                )
            )
    return chunks

"""Построение RAGAssistant из YAML/JSON конфига или Python-фабрики.

YAML/JSON-конфиг описывает компоненты декларативно — без кода. Например::

    name: GigaChat + reranker
    llm:
      provider: gigachat
      args: { model: GigaChat-Pro, scope: GIGACHAT_API_PERS, verify_ssl_certs: false }
    embedder:
      provider: gigachat
      args: { model: Embeddings, verify_ssl_certs: false, cache_path: cache/emb.db }
    reranker:
      provider: bge
      args: { device: cuda }
    rag:
      sim_threshold: 0.55
      top_k: 4
      top_k_retrieval: 16
      rerank_threshold: 0.3
    corpus:
      type: mdn          # или: type: pdf, path: ~/methodichki

Python-фабрика — функция, возвращающая ``RAGAssistant``::

    # myconfigs.py
    def build_bge():
        from web_ai_assistant.rag import RAGAssistant
        from web_ai_assistant.rerankers import BGEReranker
        ...
        return RAGAssistant(index=index, llm=llm, reranker=BGEReranker())

И в CLI: ``webai-ab --a myconfigs:build_bge --b myconfigs:build_no_reranker``.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


def load_from_pyfunc(spec: str) -> Any:
    """``module.path:func_name`` → ``func_name()``. Возвращает RAGAssistant."""
    if ":" not in spec:
        raise ValueError(f"Ожидаю формат 'module:func', получил: {spec!r}")
    module_name, func_name = spec.split(":", 1)
    mod = importlib.import_module(module_name)
    func = getattr(mod, func_name)
    return func()


def load_yaml_config(path: str | Path) -> dict:
    """Читает YAML или JSON."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Для YAML-конфига нужен PyYAML. pip install 'web-ai-assistant[eval-ab]'"
            ) from exc
        return yaml.safe_load(text)
    import json as _json

    return _json.loads(text)


def build_from_yaml(path: str | Path):
    """Собирает ``RAGAssistant`` из декларативного YAML/JSON.

    Все провайдеры ленивые: тяжёлые ML/cloud-зависимости не подтягиваются,
    пока не понадобятся. Корпус (MDN/PDF) при необходимости индексируется.
    """
    from ..corpus import load_mdn_corpus, load_pdf_corpus, split_documents
    from ..index import VectorIndex
    from ..rag import RAGAssistant

    cfg = load_yaml_config(path)

    # ---------- embedder ----------
    emb_cfg = cfg.get("embedder", {"provider": "e5"})
    emb_args = emb_cfg.get("args", {}) or {}
    embedder = _build_embedder(emb_cfg["provider"], **emb_args)

    # ---------- corpus + index ----------
    corpus_cfg = cfg.get("corpus", {"type": "mdn"})
    if corpus_cfg.get("type") == "pdf":
        path_arg = corpus_cfg.get("path") or corpus_cfg["pdf_path"]
        docs = load_pdf_corpus(path_arg, ocr_fallback=False)
    else:
        docs = load_mdn_corpus()
    chunks = split_documents(docs)
    index = VectorIndex(embedder=embedder)
    index.add(chunks)

    # ---------- llm ----------
    llm_cfg = cfg.get("llm", {"provider": "qwen"})
    llm_args = llm_cfg.get("args", {}) or {}
    llm = _build_llm(llm_cfg["provider"], **llm_args)

    # ---------- reranker (optional) ----------
    reranker = None
    if "reranker" in cfg and cfg["reranker"]:
        r_cfg = cfg["reranker"]
        reranker = _build_reranker(r_cfg["provider"], **(r_cfg.get("args", {}) or {}))

    rag_cfg = cfg.get("rag", {})
    return RAGAssistant(
        index=index,
        llm=llm,
        sim_threshold=float(rag_cfg.get("sim_threshold", 0.55)),
        top_k=int(rag_cfg.get("top_k", 4)),
        reranker=reranker,
        top_k_retrieval=int(rag_cfg.get("top_k_retrieval", 16)),
        rerank_threshold=rag_cfg.get("rerank_threshold"),
    )


def _build_embedder(provider: str, **kwargs):
    from ..embeddings import CachedEmbedder, E5Embedder, GigaChatEmbedder

    cache_path = kwargs.pop("cache_path", None)
    if provider == "gigachat":
        base = GigaChatEmbedder(**kwargs)
    elif provider == "e5":
        base = E5Embedder(**kwargs)
    else:
        raise ValueError(f"unknown embedder provider: {provider!r}")
    if cache_path:
        return CachedEmbedder(base, cache_path=cache_path)
    return base


def _build_llm(provider: str, **kwargs):
    from ..llms import GigaChatLLM, LocalQwenLLM

    if provider == "gigachat":
        return GigaChatLLM(**kwargs)
    if provider == "qwen":
        return LocalQwenLLM(**kwargs)
    raise ValueError(f"unknown llm provider: {provider!r}")


def _build_reranker(provider: str, **kwargs):
    from ..rerankers import BGEReranker, GigaChatReranker

    if provider == "bge":
        return BGEReranker(**kwargs)
    if provider == "gigachat":
        return GigaChatReranker(**kwargs)
    raise ValueError(f"unknown reranker provider: {provider!r}")

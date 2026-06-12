"""FastAPI-обёртка вокруг :class:`RAGAssistant`.

Эндпоинты:
- ``GET  /healthz``         — статус сервиса.
- ``POST /ask``             — синхронный ответ ``Answer`` в JSON.
- ``POST /ask/stream``      — Server-Sent Events (token-streaming).
- ``GET  /``                — статический Vue-фронтенд (если включён).

Запуск::

    uvicorn web_ai_assistant.server:create_app --factory --host 0.0.0.0 --port 8000

Сервер не загружает тяжёлый ML-стек при импорте: ``RAGAssistant`` создаётся
фабрикой в момент старта приложения. Это критично для тестов (фейк-ассистент)
и для разворачивания в Colab.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .analytics.clustering import cluster_queries
from .analytics.storage import QueryLog, QueryStore, hash_ip
from .analytics.threshold import suggest_threshold
from .privacy import is_logging_enabled, redact
from .rag import Answer, RAGAssistant

# ----------------------------------------------------------------------------
# Pydantic-схемы
# ----------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(None, ge=1, le=20)
    sim_threshold: float | None = Field(None, ge=0.0, le=1.0)


class Source(BaseModel):
    id: int
    title: str
    url: str
    sim: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source] = []
    max_sim: float | None = None
    blocked: str | None = None


class ApplyThresholdRequest(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0)


def _answer_to_response(a: Answer) -> AskResponse:
    data = asdict(a)
    return AskResponse(**data)


# ----------------------------------------------------------------------------
# Зависимость: ассистент (внедряется при старте)
# ----------------------------------------------------------------------------

AssistantFactory = Callable[[], RAGAssistant]
EmbedderFactory = Callable[[], object]  # объект с .embed_query() для лог-векторов


def create_app(
    assistant_factory: AssistantFactory | None = None,
    *,
    static_dir: str | os.PathLike | None = None,
    cors_origins: list[str] | None = None,
    query_store: QueryStore | None = None,
    embedder_factory: EmbedderFactory | None = None,
    admin_password: str | None = None,
) -> FastAPI:
    """Фабрика приложения.

    :param assistant_factory: вызывается один раз при старте.
        Если ``None``, ассистент попытается собраться по-умолчанию (e5 + Chroma + Qwen).
    :param static_dir: путь к директории со статикой Vue-демо (``index.html``,
        ``app.js``). Если задан и существует — монтируется на ``/``.
    :param cors_origins: список разрешённых Origin (для отладки можно
        ``["*"]``). По умолчанию ``["*"]`` — это нужно для случая, когда
        ngrok-URL и страница на pplx.app живут на разных доменах.
    """
    state: dict[str, object | None] = {"assistant": None, "embedder": None, "store": None}
    log_enabled = is_logging_enabled()
    pw = admin_password or os.environ.get("ADMIN_PASSWORD")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if assistant_factory is not None:
            state["assistant"] = assistant_factory()
        if embedder_factory is not None:
            state["embedder"] = embedder_factory()
        if log_enabled:
            state["store"] = query_store or QueryStore(
                os.environ.get("LOG_DB_PATH", "logs/queries.db")
            )
        yield
        state["assistant"] = None
        state["embedder"] = None
        store = state.get("store")
        if store is not None and query_store is None:
            store.close()  # type: ignore[union-attr]
        state["store"] = None

    app = FastAPI(
        title="web-ai-assistant",
        version="0.1.0",
        description="MVP RAG-ассистент ИСТ ДГТУ. См. https://github.com/IrinaZeroDev/web-ai-assistant-mvp",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    def get_assistant() -> RAGAssistant:
        bot = state["assistant"]
        if bot is None:
            raise HTTPException(
                status_code=503,
                detail="Assistant is not initialised. Provide assistant_factory at app build time.",
            )
        return bot  # type: ignore[return-value]

    # ---------- helpers для логирования ----------

    def _log_query(
        question: str,
        answer_text: str,
        ans: Answer,
        latency_ms: int,
        request: Request,
        provider: str,
    ) -> None:
        store = state.get("store")
        if store is None:
            return
        emb_obj = state.get("embedder")
        emb_vec: list[float] | None = None
        if emb_obj is not None and ans.blocked is None:
            try:
                emb_vec = list(emb_obj.embed_query(question))  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                emb_vec = None
        client_ip = request.client.host if request.client else None
        try:
            store.insert(  # type: ignore[union-attr]
                QueryLog(
                    question=redact(question),
                    answer=redact(answer_text[:2048]),
                    blocked=ans.blocked,
                    max_sim=ans.max_sim,
                    source_count=len(ans.sources or []),
                    latency_ms=latency_ms,
                    llm_provider=provider,
                    client_id=request.headers.get("x-client-id"),
                    ip_hash=hash_ip(client_ip),
                    embedding=emb_vec,
                )
            )
        except Exception:  # noqa: BLE001 — логирование не должно ломать запрос
            pass

    def _provider_name(bot: RAGAssistant) -> str:
        return type(getattr(bot, "llm", object())).__name__

    # ---------- admin auth ----------

    security = HTTPBasic(auto_error=False)

    def require_admin(credentials: HTTPBasicCredentials | None = Depends(security)) -> None:
        if pw is None:
            return  # открыто — пароль не задан (удобно в Colab за ngrok-туннелем)
        if credentials is None or not secrets.compare_digest(credentials.password, pw):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Wrong admin password",
                headers={"WWW-Authenticate": "Basic"},
            )

    # ---------- health ----------

    @app.get("/healthz")
    def healthz() -> dict:
        return {
            "status": "ok",
            "assistant_ready": state["assistant"] is not None,
            "version": "0.1.0",
        }

    # ---------- /ask (sync) ----------

    @app.post("/ask", response_model=AskResponse)
    def ask(
        req: AskRequest,
        request: Request,
        bot: RAGAssistant = Depends(get_assistant),
    ) -> AskResponse:
        original_k, original_t = bot.top_k, bot.sim_threshold
        t0 = time.monotonic()
        try:
            if req.top_k is not None:
                bot.top_k = req.top_k
            if req.sim_threshold is not None:
                bot.sim_threshold = req.sim_threshold
            answer = bot.ask(req.question)
        finally:
            bot.top_k, bot.sim_threshold = original_k, original_t
        latency_ms = int((time.monotonic() - t0) * 1000)
        _log_query(req.question, answer.answer, answer, latency_ms, request, _provider_name(bot))
        return _answer_to_response(answer)

    # ---------- /ask/stream (SSE) ----------

    @app.post("/ask/stream")
    async def ask_stream(
        req: AskRequest, request: Request, bot: RAGAssistant = Depends(get_assistant)
    ) -> StreamingResponse:
        async def event_stream() -> AsyncIterator[bytes]:
            t0 = time.monotonic()
            sa = await asyncio.to_thread(bot.ask_stream, req.question)

            yield _sse(
                "meta",
                {
                    "blocked": sa.blocked,
                    "max_sim": sa.max_sim,
                    "source_count": len(sa.sources),
                },
            )

            collected: list[str] = []
            it = sa.tokens
            while True:
                if await request.is_disconnected():
                    return
                token = await asyncio.to_thread(next, it, None)
                if token is None:
                    break
                collected.append(token)
                yield _sse("token", {"text": token})

            yield _sse("done", {"sources": list(sa.sources or [])})
            latency_ms = int((time.monotonic() - t0) * 1000)
            ans_for_log = Answer(
                answer="".join(collected),
                sources=list(sa.sources or []),
                max_sim=sa.max_sim,
                blocked=sa.blocked,
            )
            _log_query(req.question, ans_for_log.answer, ans_for_log, latency_ms, request, _provider_name(bot))

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # отключает буферизацию nginx-прокси
                "Connection": "keep-alive",
            },
        )

    # ---------- admin: аналитика запросов ----------

    @app.get("/admin/api/stats", dependencies=[Depends(require_admin)])
    def admin_stats() -> dict:
        store = state.get("store")
        if store is None:
            return {"enabled": False, "total": 0, "by_blocked": {}}
        return {
            "enabled": True,
            "total": store.count(),  # type: ignore[union-attr]
            "by_blocked": store.by_blocked(),  # type: ignore[union-attr]
        }

    @app.get("/admin/api/clusters", dependencies=[Depends(require_admin)])
    def admin_clusters_json(
        n: int = 8, backend: str = "kmeans", only_unblocked: bool = True
    ) -> dict:
        store = state.get("store")
        if store is None:
            return {"clusters": [], "total": 0}
        rows = store.all_for_clustering(only_unblocked=only_unblocked)  # type: ignore[union-attr]
        clusters = cluster_queries(rows, backend=backend, k=n)
        return {
            "clusters": [asdict(c) for c in clusters],
            "total": len(rows),
            "backend": backend,
        }

    @app.get("/admin/api/recent", dependencies=[Depends(require_admin)])
    def admin_recent(limit: int = 100) -> list[dict]:
        store = state.get("store")
        if store is None:
            return []
        return store.recent(limit=limit)  # type: ignore[union-attr]

    # ---------- admin: adaptive sim_threshold ----------

    @app.get("/admin/api/threshold/suggest", dependencies=[Depends(require_admin)])
    def admin_threshold_suggest(
        method: str = "auto",
        min_sample: int = 30,
        fallback_percentile: float = 5.0,
    ) -> dict:
        store = state.get("store")
        bot = state.get("assistant")
        current = float(bot.sim_threshold) if bot is not None else None  # type: ignore[union-attr]
        if store is None:
            return {
                "current": current,
                "suggestion": None,
                "error": "Логирование выключено — нет выборки для подбора порога.",
            }
        dist = store.max_sim_distribution()  # type: ignore[union-attr]
        sug = suggest_threshold(
            dist["in_corpus"],
            dist["out_of_corpus"],
            method=method,  # type: ignore[arg-type]
            min_sample=min_sample,
            fallback_percentile=fallback_percentile,
        )
        return {"current": current, "suggestion": sug.as_dict()}

    @app.post("/admin/api/threshold/apply", dependencies=[Depends(require_admin)])
    def admin_threshold_apply(req: ApplyThresholdRequest) -> dict:
        bot = state.get("assistant")
        if bot is None:
            raise HTTPException(status_code=503, detail="Assistant not initialised.")
        previous = float(bot.sim_threshold)  # type: ignore[union-attr]
        bot.sim_threshold = float(req.threshold)  # type: ignore[union-attr]
        return {"previous": previous, "applied": float(req.threshold)}

    @app.get("/admin/clusters", dependencies=[Depends(require_admin)], response_class=HTMLResponse)
    def admin_clusters_html(n: int = 8, backend: str = "kmeans") -> str:
        store = state.get("store")
        if store is None:
            return _render_admin_disabled()
        rows = store.all_for_clustering(only_unblocked=True)  # type: ignore[union-attr]
        by_blocked = store.by_blocked()  # type: ignore[union-attr]
        total = store.count()  # type: ignore[union-attr]
        clusters = cluster_queries(rows, backend=backend, k=n) if rows else []
        return _render_admin_html(
            total=total,
            by_blocked=by_blocked,
            clusters=clusters,
            backend=backend,
            embedder_ready=state.get("embedder") is not None,
            n_with_emb=len(rows),
        )

    # ---------- статика (Vue-демо) ----------

    if static_dir is not None:
        sp = Path(static_dir)
        if sp.is_dir():
            app.mount("/", StaticFiles(directory=str(sp), html=True), name="static")

    return app


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _sse(event: str, data: dict) -> bytes:
    """Кадр Server-Sent Events: ``event: ...\\ndata: {json}\\n\\n``."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n".encode()


# ---------------------------------------------------------------------------
# Admin HTML
# ---------------------------------------------------------------------------


_ADMIN_CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, system-ui, "Segoe UI", Roboto, sans-serif; max-width: 1100px;
       margin: 2em auto; padding: 0 1em; color: #1a1a1a; }
h1 { font-size: 1.4em; margin-bottom: 0.2em; }
p.sub { color: #666; margin: 0 0 1.5em; }
.cards { display: flex; gap: 0.75em; margin-bottom: 1.5em; flex-wrap: wrap; }
.card { background: #f7f7f9; padding: 0.75em 1em; border-radius: 8px; min-width: 130px; }
.card .label { color: #666; font-size: 0.85em; }
.card .value { font-size: 1.5em; font-weight: 600; }
.controls { margin: 1em 0; }
.controls a, .controls span { margin-right: 0.6em; }
.cluster { border: 1px solid #e1e1e5; border-radius: 8px; padding: 0.75em 1em;
           margin-bottom: 0.75em; background: #fff; }
.cluster h3 { margin: 0 0 0.4em; font-size: 1em; }
.cluster ul { margin: 0; padding-left: 1.2em; }
.cluster li { margin: 0.15em 0; font-size: 0.95em; }
.bar { height: 6px; background: #4f46e5; border-radius: 3px; margin-top: 0.25em; }
.empty { color: #888; padding: 1em; background: #fafafa; border-radius: 8px; }
"""


def _render_admin_disabled() -> str:
    return (
        "<!doctype html><meta charset=utf-8><style>" + _ADMIN_CSS + "</style>"
        "<h1>Аналитика недоступна</h1>"
        "<p class=sub>Логирование отключено: <code>LOG_QUERIES=false</code>.</p>"
    )


def _render_admin_html(
    *,
    total: int,
    by_blocked: dict,
    clusters: list,
    backend: str,
    embedder_ready: bool,
    n_with_emb: int,
) -> str:
    def esc(s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    answered = by_blocked.get("__answered__", 0)
    red_zone = by_blocked.get("red_zone", 0)
    escalation = by_blocked.get("escalation", 0)
    out_of_corpus = by_blocked.get("out_of_corpus", 0)

    max_size = max((c.size for c in clusters), default=1)
    cluster_html_parts: list[str] = []
    for c in clusters:
        reps = "".join(
            f"<li>#{esc(str(r['id']))}: {esc(r['question'])}</li>"
            for r in c.representatives
        )
        cluster_html_parts.append(
            f'<div class=cluster>'
            f'<h3>Кластер #{c.label} — {c.size} запросов</h3>'
            f'<div class=bar style="width:{int(100 * c.size / max_size)}%"></div>'
            f'<ul>{reps}</ul>'
            f'</div>'
        )
    clusters_html = "".join(cluster_html_parts) or (
        '<div class=empty>Запросов с эмбеддингами пока нет. '
        + ("" if embedder_ready else "Подключите embedder_factory в create_app() — иначе векторы не сохраняются.")
        + "</div>"
    )

    return (
        "<!doctype html><html lang=ru><meta charset=utf-8>"
        "<title>Кластеры затруднений</title>"
        f"<style>{_ADMIN_CSS}</style>"
        "<body>"
        "<h1>Кластеры затруднений</h1>"
        f"<p class=sub>Всего запросов: <b>{total}</b>. С эмбеддингами: <b>{n_with_emb}</b>. "
        f"Backend: <code>{esc(backend)}</code></p>"
        "<div class=cards>"
        f"<div class=card><div class=label>Отвечено</div><div class=value>{answered}</div></div>"
        f"<div class=card><div class=label>Red zone</div><div class=value>{red_zone}</div></div>"
        f"<div class=card><div class=label>Эскалация</div><div class=value>{escalation}</div></div>"
        f"<div class=card><div class=label>Вне корпуса</div><div class=value>{out_of_corpus}</div></div>"
        "</div>"
        "<div class=controls>"
        '<span>Backend: <a href="?backend=kmeans">kmeans</a> · <a href="?backend=hdbscan">hdbscan</a></span>'
        '<span>N: <a href="?n=4">4</a> · <a href="?n=8">8</a> · <a href="?n=12">12</a></span>'
        "</div>"
        f"{clusters_html}"
        "</body></html>"
    )




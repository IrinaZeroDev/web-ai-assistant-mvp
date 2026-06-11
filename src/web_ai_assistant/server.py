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
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

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


def _answer_to_response(a: Answer) -> AskResponse:
    data = asdict(a)
    return AskResponse(**data)


# ----------------------------------------------------------------------------
# Зависимость: ассистент (внедряется при старте)
# ----------------------------------------------------------------------------

AssistantFactory = Callable[[], RAGAssistant]


def create_app(
    assistant_factory: AssistantFactory | None = None,
    *,
    static_dir: str | os.PathLike | None = None,
    cors_origins: list[str] | None = None,
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
    state: dict[str, RAGAssistant | None] = {"assistant": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if assistant_factory is not None:
            state["assistant"] = assistant_factory()
        yield
        state["assistant"] = None

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
        return bot

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
    def ask(req: AskRequest, bot: RAGAssistant = Depends(get_assistant)) -> AskResponse:
        # пер-запросный override параметров retrieval
        original_k, original_t = bot.top_k, bot.sim_threshold
        try:
            if req.top_k is not None:
                bot.top_k = req.top_k
            if req.sim_threshold is not None:
                bot.sim_threshold = req.sim_threshold
            answer = bot.ask(req.question)
        finally:
            bot.top_k, bot.sim_threshold = original_k, original_t
        return _answer_to_response(answer)

    # ---------- /ask/stream (SSE) ----------

    @app.post("/ask/stream")
    async def ask_stream(
        req: AskRequest, request: Request, bot: RAGAssistant = Depends(get_assistant)
    ) -> StreamingResponse:
        async def event_stream() -> AsyncIterator[bytes]:
            # 1) выполняем ask() в worker thread, чтобы не блокировать loop
            #    (внутри — синхронный LLM.generate)
            answer = await asyncio.to_thread(bot.ask, req.question)

            # 2) если запрос заблокирован — отдаём один кадр и done
            if answer.blocked is not None:
                yield _sse("meta", {"blocked": answer.blocked, "max_sim": answer.max_sim})
                yield _sse("token", {"text": answer.answer})
                yield _sse("done", {"sources": []})
                return

            # 3) иначе — стримим текст пословно, чтобы UI рисовал постепенно
            yield _sse(
                "meta",
                {"blocked": None, "max_sim": answer.max_sim, "source_count": len(answer.sources)},
            )
            for chunk in _chunk_text(answer.answer):
                if await request.is_disconnected():
                    return
                yield _sse("token", {"text": chunk})
                await asyncio.sleep(0.02)  # имитация streaming

            yield _sse(
                "done",
                {"sources": [s for s in (answer.sources or [])]},
            )

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # отключает буферизацию nginx-прокси
                "Connection": "keep-alive",
            },
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


def _chunk_text(text: str, size: int = 18) -> list[str]:
    """Режет текст на короткие куски по словам — для имитации token-streaming."""
    words = text.split(" ")
    out, buf = [], ""
    for w in words:
        if len(buf) + len(w) + 1 > size and buf:
            out.append(buf + " ")
            buf = w
        else:
            buf = w if not buf else f"{buf} {w}"
    if buf:
        out.append(buf)
    return out

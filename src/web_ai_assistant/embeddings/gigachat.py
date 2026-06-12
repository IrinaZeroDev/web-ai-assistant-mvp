"""Облачный GigaChat-эмбеддер.

Использует официальный SDK ``gigachat`` (метод ``client.embeddings(texts, model=...)``).

Поддерживаемые модели:

- ``Embeddings`` (по умолчанию) — 1024 dim, 512 токенов контекста;
- ``EmbeddingsGigaR`` — 2560 dim, 4096 токенов, опциональные текстовые
  инструкции для query (улучшают качество RAG).

Auth key и scope — как в :class:`web_ai_assistant.llms.gigachat.GigaChatLLM`.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable

DEFAULT_MODEL = "Embeddings"
DEFAULT_SCOPE = "GIGACHAT_API_PERS"
MODEL_DIMS: dict[str, int] = {
    "Embeddings": 1024,
    "EmbeddingsGigaR": 2560,
}


class GigaChatEmbeddingsConfigError(RuntimeError):
    """Поднимается при отсутствии или некорректной конфигурации."""


class _SDKOnlyView:
    """Прокси к SDK-пути GigaChatEmbedder без кэша.

    Нужен, чтобы внутренний ``CachedEmbedder`` не зацикливался на верхний
    ``GigaChatEmbedder.embed_passages``, который сам идёт через кэш.
    """

    def __init__(self, parent: GigaChatEmbedder) -> None:
        self._parent = parent
        self.model = parent.model
        self.dim = parent.dim

    def embed_passages(self, texts):
        return self._parent._embed_passages_raw(list(texts))

    def embed_query(self, text):  # не вызывается в пайплайне кэша, но для протокола
        return self._parent.embed_query(text)


class GigaChatEmbedder:
    """GigaChat Embeddings API.

    :param auth_key: Authorization Key; если ``None`` — ``GIGACHAT_AUTH_KEY``
        / ``GIGACHAT_CREDENTIALS`` из env.
    :param model: ``Embeddings`` (default, 1024) или ``EmbeddingsGigaR`` (2560).
    :param scope: ``GIGACHAT_API_PERS`` / ``GIGACHAT_API_B2B`` / ``GIGACHAT_API_CORP``.
    :param batch_size: размер батча для ``embed_passages``. Default 32.
    :param verify_ssl_certs: ``False`` рекомендуется без установленного НУЦ-сертификата.
    :param query_instruction: текст-инструкция, которая дописывается перед
        запросом (применимо к ``EmbeddingsGigaR``, заметно улучшает recall).
        ``None`` — без инструкции.
    :param max_retries: число повторов при сетевых сбоях / 5xx.
    :param retry_backoff: базовая задержка между ретраями (секунды),
        используется экспоненциальный backoff.
    :param cache_path: путь к SQLite-кэшу эмбеддингов. Если задан — используется
        внутренний ``CachedEmbedder`` (равносильно внешней обёртке над
        этим же эмбеддером). ``None`` — кэш отключён.
    """

    def __init__(
        self,
        auth_key: str | None = None,
        *,
        model: str = DEFAULT_MODEL,
        scope: str = DEFAULT_SCOPE,
        batch_size: int = 32,
        verify_ssl_certs: bool = True,
        base_url: str | None = None,
        timeout: float = 60.0,
        query_instruction: str | None = None,
        max_retries: int = 3,
        retry_backoff: float = 1.5,
        cache_path: str | None = None,
    ) -> None:
        try:
            from gigachat import GigaChat as _GigaChatSDK
        except ImportError as exc:  # pragma: no cover
            raise GigaChatEmbeddingsConfigError(
                "Пакет gigachat не установлен. Поставьте: pip install 'web-ai-assistant[gigachat]'"
            ) from exc

        key = (
            auth_key
            or os.environ.get("GIGACHAT_AUTH_KEY")
            or os.environ.get("GIGACHAT_CREDENTIALS")
        )
        if not key:
            raise GigaChatEmbeddingsConfigError(
                "Не задан Authorization Key. Передайте auth_key=... "
                "или установите переменную окружения GIGACHAT_AUTH_KEY."
            )

        self.model = model
        self.scope = scope
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        # Известный dim — для информативности; реальный определим по первому вызову.
        self.dim: int = MODEL_DIMS.get(model, 0)

        kwargs = dict(
            credentials=key,
            scope=scope,
            verify_ssl_certs=verify_ssl_certs,
            timeout=timeout,
        )
        if base_url is not None:
            kwargs["base_url"] = base_url

        self._client = _GigaChatSDK(**kwargs)

        # Встроенный дисковый кэш. Держим его приватным — пользователь может
        # объёня желании вынести эту обёртку наружу через CachedEmbedder.
        self._cache_path = cache_path
        self._cached_view = None
        if cache_path is not None:
            from .cache import CachedEmbedder

            self._cached_view = CachedEmbedder(
                _SDKOnlyView(self), cache_path=cache_path, model_key=model
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_passages(self, texts: Iterable[str]) -> list[list[float]]:
        if self._cached_view is not None:
            return self._cached_view.embed_passages(texts)
        return self._embed_passages_raw(list(texts))

    def embed_query(self, text: str) -> list[float]:
        q = text if not self.query_instruction else f"{self.query_instruction.rstrip()} {text}"
        # query по умолчанию не кэшируются (уникальны)
        return self._embed_batch([q])[0]

    # Внутренний путь без кэша — используется в _SDKOnlyView.
    def _embed_passages_raw(self, items: list[str]) -> list[list[float]]:
        if not items:
            return []
        out: list[list[float]] = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]
            out.extend(self._embed_batch(batch))
        return out

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> GigaChatEmbedder:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.embeddings(batch, model=self.model)
            except Exception as exc:  # сеть / 5xx / таймаут
                last_exc = exc
                time.sleep(self.retry_backoff * (2**attempt))
                continue
            vectors = self._extract_vectors(resp)
            if not vectors:
                raise GigaChatEmbeddingsConfigError(
                    f"GigaChat embeddings вернул пустой ответ: {resp!r}"
                )
            if not self.dim:
                self.dim = len(vectors[0])
            return vectors
        raise GigaChatEmbeddingsConfigError(
            f"GigaChat embeddings: не удалось получить ответ за {self.max_retries} попыток"
        ) from last_exc

    @staticmethod
    def _extract_vectors(resp) -> list[list[float]]:
        """Достаёт векторы из ответа SDK независимо от формата (dict-like / pydantic)."""
        data = getattr(resp, "data", None)
        if data is None and isinstance(resp, dict):
            data = resp.get("data")
        out: list[list[float]] = []
        for item in data or []:
            vec = (
                getattr(item, "embedding", None)
                or (item.get("embedding") if isinstance(item, dict) else None)
            )
            if vec is None:
                continue
            out.append(list(vec))
        return out

"""Sber GigaChat провайдер для :class:`RAGAssistant`.

Использует официальный SDK ``gigachat`` (https://pypi.org/project/gigachat/).
Поддерживает все актуальные модели: ``GigaChat``, ``GigaChat-Pro``, ``GigaChat-Max``
и все scope: ``GIGACHAT_API_PERS`` / ``GIGACHAT_API_B2B`` / ``GIGACHAT_API_CORP``.

Authorization key передаётся одним из двух способов (по приоритету):

1. Аргумент ``auth_key=...`` в конструкторе.
2. Переменная окружения ``GIGACHAT_AUTH_KEY``
   (либо стандартная для SDK ``GIGACHAT_CREDENTIALS``).
"""

from __future__ import annotations

import os
from collections.abc import Iterator

DEFAULT_MODEL = "GigaChat"
DEFAULT_SCOPE = "GIGACHAT_API_PERS"


class GigaChatConfigError(RuntimeError):
    """Поднимается при отсутствии или некорректной конфигурации GigaChat."""


class GigaChatLLM:
    """Облачный GigaChat. Реализует протокол ``LLM`` из ``rag.py``.

    :param auth_key: Authorization Key (если ``None`` — читаем из env).
    :param model: Имя модели; ``GigaChat`` / ``GigaChat-Pro`` / ``GigaChat-Max``
        или любое другое из ``client.get_models()``.
    :param scope: ``GIGACHAT_API_PERS`` (default) / ``GIGACHAT_API_B2B`` /
        ``GIGACHAT_API_CORP``.
    :param verify_ssl_certs: По умолчанию ``True``. Для прямого обращения к
        ``gigachat.devices.sberbank.ru`` нужен установленный корневой
        сертификат НУЦ Минцифры — иначе передайте ``False``.
    :param base_url: Опциональный override (например, для preview-моделей).
    :param timeout: Таймаут одного запроса (секунды).
    :param profanity_check: Включать ли встроенный фильтр нецензурной лексики.
    """

    supports_streaming: bool = True

    def __init__(
        self,
        auth_key: str | None = None,
        *,
        model: str = DEFAULT_MODEL,
        scope: str = DEFAULT_SCOPE,
        verify_ssl_certs: bool = True,
        base_url: str | None = None,
        timeout: float = 60.0,
        profanity_check: bool | None = None,
    ) -> None:
        try:
            from gigachat import GigaChat as _GigaChatSDK
        except ImportError as exc:  # pragma: no cover
            raise GigaChatConfigError(
                "Пакет gigachat не установлен. Поставьте: pip install 'web-ai-assistant[gigachat]'"
            ) from exc

        key = (
            auth_key
            or os.environ.get("GIGACHAT_AUTH_KEY")
            or os.environ.get("GIGACHAT_CREDENTIALS")
        )
        if not key:
            raise GigaChatConfigError(
                "Не задан Authorization Key. Передайте auth_key=... "
                "или установите переменную окружения GIGACHAT_AUTH_KEY."
            )

        self.model = model
        self.scope = scope
        kwargs = dict(
            credentials=key,
            scope=scope,
            model=model,
            verify_ssl_certs=verify_ssl_certs,
            timeout=timeout,
        )
        if base_url is not None:
            kwargs["base_url"] = base_url
        if profanity_check is not None:
            kwargs["profanity_check"] = profanity_check

        self._client = _GigaChatSDK(**kwargs)

    # ---------- helpers ----------

    @staticmethod
    def _to_chat_payload(messages: list[dict], temperature: float, max_new_tokens: int) -> dict:
        """Преобразует ChatML-сообщения в payload GigaChat Chat API."""
        return {
            "messages": [
                {"role": m["role"], "content": m["content"]} for m in messages
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_new_tokens),
        }

    @staticmethod
    def _extract_text(resp) -> str:
        """Достаёт content из ответа SDK независимо от версии контракта."""
        choices = getattr(resp, "choices", None)
        if choices:
            msg = choices[0].message
            return getattr(msg, "content", "") or ""
        # новый primary-контракт
        msgs = getattr(resp, "messages", None) or []
        if msgs:
            parts = getattr(msgs[0], "content", None) or []
            if isinstance(parts, str):
                return parts
            return "".join(getattr(p, "text", "") or "" for p in parts)
        return ""

    @staticmethod
    def _extract_delta(chunk) -> str:
        """Достаёт инкрементальный токен из stream-chunk."""
        # старый контракт: chunk.choices[0].delta.content
        choices = getattr(chunk, "choices", None)
        if choices:
            delta = getattr(choices[0], "delta", None)
            if delta is not None:
                return getattr(delta, "content", "") or ""
        # новый контракт: chunk.messages[*].content[*].text
        msgs = getattr(chunk, "messages", None) or []
        out = []
        for msg in msgs:
            for part in getattr(msg, "content", None) or []:
                t = getattr(part, "text", None)
                if t:
                    out.append(t)
        return "".join(out)

    # ---------- sync ----------

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        payload = self._to_chat_payload(messages, temperature, max_new_tokens)
        resp = self._client.chat(payload)
        return self._extract_text(resp).strip()

    # ---------- streaming ----------

    def stream_generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        payload = self._to_chat_payload(messages, temperature, max_new_tokens)
        for chunk in self._client.stream(payload):
            delta = self._extract_delta(chunk)
            if delta:
                yield delta

    # ---------- lifecycle ----------

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> GigaChatLLM:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

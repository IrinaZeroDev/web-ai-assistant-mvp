"""Провайдеры языковых моделей для :class:`web_ai_assistant.rag.RAGAssistant`.

Все провайдеры реализуют протокол :class:`web_ai_assistant.rag.LLM`:

- ``generate(messages, ...) -> str`` — синхронный ответ;
- ``stream_generate(messages, ...) -> Iterator[str]`` — опциональный
  токен-стрим (если LLM поддерживает; иначе атрибут ``supports_streaming``
  устанавливается в ``False``).

Доступные провайдеры:

- :class:`web_ai_assistant.llms.qwen.LocalQwenLLM` — локальный Qwen2.5-7B-Instruct.
- :class:`web_ai_assistant.llms.gigachat.GigaChatLLM` — Sber GigaChat (cloud).
"""

from __future__ import annotations

__all__ = ["LocalQwenLLM", "GigaChatLLM"]


def __getattr__(name: str):
    # Ленивая загрузка: тяжёлые ML-зависимости не тянутся при `from web_ai_assistant.llms import ...`.
    if name == "LocalQwenLLM":
        from .qwen import LocalQwenLLM

        return LocalQwenLLM
    if name == "GigaChatLLM":
        from .gigachat import GigaChatLLM

        return GigaChatLLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

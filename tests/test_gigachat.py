"""GigaChat LLM — тесты с фейковым SDK через monkeypatch.

Не делаем сетевых вызовов: подсовываем фейк-класс ``gigachat.GigaChat`` и
проверяем, что наш провайдер корректно конвертирует payload и парсит ответ.
"""

from __future__ import annotations

import sys
import types

import pytest

# ---------------------------------------------------------------------------
# Фейк-SDK
# ---------------------------------------------------------------------------


class _Delta:
    def __init__(self, text: str):
        self.content = text


class _StreamChoice:
    def __init__(self, text: str):
        self.delta = _Delta(text)


class _StreamChunk:
    def __init__(self, text: str):
        self.choices = [_StreamChoice(text)]


class _Message:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str):
        self.choices = [_Choice(content)]


class FakeGigaChat:
    """Минимальная заглушка SDK: запоминает kwargs и payload, возвращает заранее заданный ответ."""

    last_init_kwargs: dict | None = None

    def __init__(self, **kwargs):
        FakeGigaChat.last_init_kwargs = kwargs
        self.kwargs = kwargs
        self.calls: list[dict] = []
        self.stream_calls: list[dict] = []

    def chat(self, payload):
        self.calls.append(payload)
        return _Response("Flexbox — это раскладка [1].")

    def stream(self, payload):
        self.stream_calls.append(payload)
        for piece in ["Flexbox ", "— это ", "раскладка ", "[1]."]:
            yield _StreamChunk(piece)

    def close(self):
        pass


@pytest.fixture
def fake_sdk(monkeypatch):
    """Устанавливает поддельный модуль ``gigachat`` в sys.modules."""
    fake_module = types.ModuleType("gigachat")
    fake_module.GigaChat = FakeGigaChat
    monkeypatch.setitem(sys.modules, "gigachat", fake_module)
    # сбрасываем кеш последнего kwargs между тестами
    FakeGigaChat.last_init_kwargs = None
    return fake_module


# ---------------------------------------------------------------------------
# Тесты
# ---------------------------------------------------------------------------


def test_requires_auth_key(fake_sdk, monkeypatch):
    monkeypatch.delenv("GIGACHAT_AUTH_KEY", raising=False)
    monkeypatch.delenv("GIGACHAT_CREDENTIALS", raising=False)
    from web_ai_assistant.llms.gigachat import GigaChatConfigError, GigaChatLLM

    with pytest.raises(GigaChatConfigError):
        GigaChatLLM()


def test_reads_auth_from_env(fake_sdk, monkeypatch):
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "env-key-123")
    from web_ai_assistant.llms.gigachat import GigaChatLLM

    GigaChatLLM()
    assert FakeGigaChat.last_init_kwargs["credentials"] == "env-key-123"
    assert FakeGigaChat.last_init_kwargs["scope"] == "GIGACHAT_API_PERS"
    assert FakeGigaChat.last_init_kwargs["model"] == "GigaChat"


def test_arg_overrides_env(fake_sdk, monkeypatch):
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "env-key")
    from web_ai_assistant.llms.gigachat import GigaChatLLM

    GigaChatLLM(auth_key="explicit-key", model="GigaChat-Pro", scope="GIGACHAT_API_CORP")
    kw = FakeGigaChat.last_init_kwargs
    assert kw["credentials"] == "explicit-key"
    assert kw["model"] == "GigaChat-Pro"
    assert kw["scope"] == "GIGACHAT_API_CORP"


def test_generate_sends_full_payload_and_extracts_content(fake_sdk):
    from web_ai_assistant.llms.gigachat import GigaChatLLM

    llm = GigaChatLLM(auth_key="k")
    out = llm.generate(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}],
        max_new_tokens=200,
        temperature=0.1,
    )
    assert out == "Flexbox — это раскладка [1]."
    payload = llm._client.calls[0]
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]
    assert payload["temperature"] == 0.1
    assert payload["max_tokens"] == 200


def test_stream_generate_yields_token_by_token(fake_sdk):
    from web_ai_assistant.llms.gigachat import GigaChatLLM

    llm = GigaChatLLM(auth_key="k")
    chunks = list(llm.stream_generate([{"role": "user", "content": "Q"}]))
    assert chunks == ["Flexbox ", "— это ", "раскладка ", "[1]."]


def test_supports_streaming_flag(fake_sdk):
    from web_ai_assistant.llms.gigachat import GigaChatLLM

    llm = GigaChatLLM(auth_key="k")
    assert llm.supports_streaming is True


def test_extracts_new_contract_response(fake_sdk):
    """Поддержка нового primary-контракта SDK (chunk.messages[*].content[*].text)."""
    from web_ai_assistant.llms.gigachat import GigaChatLLM

    class _Part:
        def __init__(self, t):
            self.text = t

    class _Msg:
        def __init__(self, parts):
            self.content = parts

    class _NewResp:
        def __init__(self, parts):
            self.messages = [_Msg(parts)]

    llm = GigaChatLLM(auth_key="k")
    text = llm._extract_text(_NewResp([_Part("Hello "), _Part("world")]))
    assert text == "Hello world"

    delta = llm._extract_delta(_NewResp([_Part("chunk")]))
    assert delta == "chunk"

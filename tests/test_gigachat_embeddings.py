"""GigaChatEmbedder — тесты с поддельным SDK (без сети)."""

from __future__ import annotations

import sys
import types

import pytest

# ---------------------------------------------------------------------------
# Поддельный SDK
# ---------------------------------------------------------------------------


class _Item:
    def __init__(self, vec):
        self.embedding = vec


class _Resp:
    def __init__(self, vectors):
        self.data = [_Item(v) for v in vectors]


class FakeSDK:
    instances: list = []

    def __init__(self, **kwargs):
        FakeSDK.instances.append(self)
        self.kwargs = kwargs
        self.calls: list[dict] = []  # batches и model
        self.fail_times = 0  # вернуть исключение N раз подряд
        self.dim = 4

    def embeddings(self, texts, model=None):
        self.calls.append({"texts": list(texts), "model": model})
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("simulated network error")
        # фиктивный вектор — индекс i по dim
        return _Resp([[float(i + 1)] * self.dim for i in range(len(texts))])

    def close(self):
        pass


@pytest.fixture
def fake_sdk(monkeypatch):
    mod = types.ModuleType("gigachat")
    mod.GigaChat = FakeSDK
    monkeypatch.setitem(sys.modules, "gigachat", mod)
    FakeSDK.instances.clear()
    return mod


# ---------------------------------------------------------------------------
# Тесты
# ---------------------------------------------------------------------------


def test_requires_auth_key(fake_sdk, monkeypatch):
    monkeypatch.delenv("GIGACHAT_AUTH_KEY", raising=False)
    monkeypatch.delenv("GIGACHAT_CREDENTIALS", raising=False)
    from web_ai_assistant.embeddings.gigachat import (
        GigaChatEmbedder,
        GigaChatEmbeddingsConfigError,
    )

    with pytest.raises(GigaChatEmbeddingsConfigError):
        GigaChatEmbedder()


def test_known_dim_for_embeddings_model(fake_sdk, monkeypatch):
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "k")
    from web_ai_assistant.embeddings.gigachat import GigaChatEmbedder

    emb = GigaChatEmbedder()
    assert emb.dim == 1024  # известная размерность для "Embeddings"


def test_known_dim_for_gigar_model(fake_sdk, monkeypatch):
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "k")
    from web_ai_assistant.embeddings.gigachat import GigaChatEmbedder

    emb = GigaChatEmbedder(model="EmbeddingsGigaR")
    assert emb.dim == 2560


def test_embed_passages_batches(fake_sdk, monkeypatch):
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "k")
    from web_ai_assistant.embeddings.gigachat import GigaChatEmbedder

    emb = GigaChatEmbedder(batch_size=2)
    texts = [f"doc-{i}" for i in range(5)]
    vecs = emb.embed_passages(texts)
    assert len(vecs) == 5
    # должно быть 3 батч-вызова: 2 + 2 + 1
    sdk = FakeSDK.instances[-1]
    assert len(sdk.calls) == 3
    assert [len(c["texts"]) for c in sdk.calls] == [2, 2, 1]
    # модель прокинута
    assert all(c["model"] == "Embeddings" for c in sdk.calls)


def test_embed_query_no_instruction(fake_sdk, monkeypatch):
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "k")
    from web_ai_assistant.embeddings.gigachat import GigaChatEmbedder

    emb = GigaChatEmbedder()
    vec = emb.embed_query("Что такое flexbox?")
    assert len(vec) == 4
    sdk = FakeSDK.instances[-1]
    assert sdk.calls[0]["texts"] == ["Что такое flexbox?"]


def test_embed_query_with_instruction(fake_sdk, monkeypatch):
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "k")
    from web_ai_assistant.embeddings.gigachat import GigaChatEmbedder

    emb = GigaChatEmbedder(
        model="EmbeddingsGigaR",
        query_instruction="Найди материалы курса, отвечающие на вопрос:",
    )
    emb.embed_query("Что такое flexbox?")
    sdk = FakeSDK.instances[-1]
    text = sdk.calls[0]["texts"][0]
    assert text == "Найди материалы курса, отвечающие на вопрос: Что такое flexbox?"


def test_retry_on_transient_error(fake_sdk, monkeypatch):
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "k")
    from web_ai_assistant.embeddings.gigachat import GigaChatEmbedder

    emb = GigaChatEmbedder(max_retries=3, retry_backoff=0.0)
    # подсунем 2 ошибки подряд, потом успех
    FakeSDK.instances[-1].fail_times = 2  # noqa: SLF001  — фейк
    vec = emb.embed_query("x")
    assert len(vec) == 4


def test_arg_overrides_env(fake_sdk, monkeypatch):
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "env-key")
    from web_ai_assistant.embeddings.gigachat import GigaChatEmbedder

    GigaChatEmbedder(auth_key="explicit", scope="GIGACHAT_API_CORP")
    sdk = FakeSDK.instances[-1]
    assert sdk.kwargs["credentials"] == "explicit"
    assert sdk.kwargs["scope"] == "GIGACHAT_API_CORP"

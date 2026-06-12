"""GigaChat-as-reranker (LLM-judge поверх chat-эндпоинта).

У GigaChat **нет отдельного reranker-API**, поэтому мы используем chat-модель
как scoring-функцию: для каждого кандидата просим вернуть число от 0 до 10,
насколько фрагмент релевантен запросу.

Это медленнее и дороже, чем BGE (один HTTP-запрос на пару), зато без GPU и
с соответствием 152-ФЗ. Подходит для пилотов на ваших дисциплинах.

Реализованы две оптимизации:

- **батчинг** — несколько пар в одном prompt'е (по умолчанию ``batch_size=4``);
- **усечение** фрагментов до ``max_chars=800`` символов перед скорингом.

Если нужны максимальные качество и скорость — берите :class:`BGEReranker`.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence

from ..llms.gigachat import GigaChatConfigError

DEFAULT_MODEL = "GigaChat"
DEFAULT_SCOPE = "GIGACHAT_API_PERS"


_JUDGE_PROMPT = (
    "Ты — ассистент-судья качества поиска. Тебе дан запрос пользователя "
    "и пронумерованные фрагменты. Для КАЖДОГО фрагмента верни число от 0 до 10:\n"
    "  10 = фрагмент полностью отвечает на запрос (приоритет: явные определения, термины запроса)\n"
    "   5 = фрагмент косвенно связан\n"
    "   0 = фрагмент не помогает ответить\n"
    "\n"
    "Формат ответа: по одной строке для каждого фрагмента, "
    "вида '<номер>: <число>'. Без объяснений.\n"
    "\n"
    "ЗАПРОС: {query}\n"
    "\n"
    "ФРАГМЕНТЫ:\n"
    "{fragments}\n"
    "\n"
    "Оценки:"
)


class GigaChatReranker:
    """LLM-judge реранкер через GigaChat chat-эндпоинт.

    :param auth_key: Authorization Key (или env ``GIGACHAT_AUTH_KEY``).
    :param model: ``GigaChat`` / ``GigaChat-Pro`` / ``GigaChat-Max``. Pro/Max
        дают заметно более стабильные скоры.
    :param scope: ``PERS`` / ``B2B`` / ``CORP``.
    :param batch_size: сколько кандидатов передавать в одном prompt'е.
    :param max_chars: максимальная длина фрагмента — лишнее обрезается.
    :param verify_ssl_certs: ``False`` без установленного НУЦ-сертификата.
    """

    def __init__(
        self,
        auth_key: str | None = None,
        *,
        model: str = DEFAULT_MODEL,
        scope: str = DEFAULT_SCOPE,
        batch_size: int = 4,
        max_chars: int = 800,
        verify_ssl_certs: bool = True,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        try:
            from gigachat import GigaChat as _GigaChatSDK
        except ImportError as exc:  # pragma: no cover
            raise GigaChatConfigError(
                "gigachat не установлен. pip install 'web-ai-assistant[gigachat]'"
            ) from exc

        key = (
            auth_key
            or os.environ.get("GIGACHAT_AUTH_KEY")
            or os.environ.get("GIGACHAT_CREDENTIALS")
        )
        if not key:
            raise GigaChatConfigError(
                "Не задан Authorization Key. Передайте auth_key=... или GIGACHAT_AUTH_KEY."
            )

        self.model = model
        self.scope = scope
        self.batch_size = max(1, int(batch_size))
        self.max_chars = max(64, int(max_chars))

        kwargs = dict(
            credentials=key,
            scope=scope,
            model=model,
            verify_ssl_certs=verify_ssl_certs,
            timeout=timeout,
        )
        if base_url is not None:
            kwargs["base_url"] = base_url
        self._client = _GigaChatSDK(**kwargs)

    # ---------- public ----------

    def rerank(self, query: str, candidates: Sequence[str]) -> list[float]:
        if not candidates:
            return []
        scores: list[float] = [0.0] * len(candidates)
        for offset in range(0, len(candidates), self.batch_size):
            batch = list(candidates[offset : offset + self.batch_size])
            batch_scores = self._score_batch(query, batch)
            for i, s in enumerate(batch_scores):
                scores[offset + i] = s
        return scores

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> GigaChatReranker:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ---------- internals ----------

    def _score_batch(self, query: str, batch: list[str]) -> list[float]:
        fragments = "\n".join(
            f"[{i + 1}] {text[: self.max_chars]}" for i, text in enumerate(batch)
        )
        prompt = _JUDGE_PROMPT.format(query=query, fragments=fragments)
        payload = {
            "messages": [
                {"role": "system", "content": "Ты — точный, лаконичный судья поиска."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 256,
        }
        resp = self._client.chat(payload)
        raw = self._extract_text(resp)
        return self._parse_scores(raw, expected=len(batch))

    @staticmethod
    def _extract_text(resp) -> str:
        choices = getattr(resp, "choices", None)
        if choices:
            msg = choices[0].message
            return getattr(msg, "content", "") or ""
        msgs = getattr(resp, "messages", None) or []
        if msgs:
            parts = getattr(msgs[0], "content", None) or []
            if isinstance(parts, str):
                return parts
            return "".join(getattr(p, "text", "") or "" for p in parts)
        return ""

    @staticmethod
    def _parse_scores(text: str, expected: int) -> list[float]:
        """Парсит ответ ``"1: 8\\n2: 3\\n..."``; нормирует к [0, 1]."""
        scores: dict[int, float] = {}
        # Поддерживаемые форматы строк: "1: 8", "[1]: 8", "[1] 8", "1) 8".
        line_re = re.compile(r"\[?\s*(\d+)\s*\]?\s*[:)\.]?\s*([-+]?\d+(?:[.,]\d+)?)")
        for line in text.splitlines():
            m = line_re.search(line)
            if not m:
                continue
            idx = int(m.group(1))
            try:
                val = float(m.group(2).replace(",", "."))
            except ValueError:
                continue
            scores[idx] = max(0.0, min(10.0, val)) / 10.0
        out = [scores.get(i + 1, 0.0) for i in range(expected)]
        return out

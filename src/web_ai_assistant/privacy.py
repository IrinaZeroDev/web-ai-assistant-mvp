"""Минимальная PII-редакция для соответствия 152-ФЗ при логировании.

Заменяет в тексте:

- email   → ``[EMAIL]``
- телефон → ``[PHONE]``
- ФИО (RU тройка и EN First Last) → ``[PERSON]``
- группы/студ. билеты ("гр. ИСТ-21" / "billet 12345") → ``[STUDENT_ID]``

Это эвристика, а не PII-классификатор уровня Presidio. Достаточно для
"быстрого фильтра" перед записью в лог-таблицу.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Регексы
# ---------------------------------------------------------------------------

EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

# +7 (123) 456-78-90 | 8-123-456-78-90 | 89991234567 | +1-202-555-0100
# Ловим с любыми разделителями; пост проверяем что после вычистки осталось ≥7 цифр.
PHONE_RE = re.compile(
    r"""
    (?<![\w])                        # граница слева
    \+?                              # опц. «+»
    (?:\(?\d{1,4}\)?[\s.\-]?){2,7}   # группы цифр с разделителями
    \d{2,4}                          # суффикс
    (?!\w)
    """,
    re.VERBOSE,
)

# Русские ФИО: 3 слова с большой буквы (Иванов Иван Иванович | Лопухина А.Б.)
RU_PERSON_RE = re.compile(
    r"\b[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ](?:[а-яё]+|\.))(?:\s+[А-ЯЁ](?:[а-яё]+|\.))?\b"
)

# Английские имена: 2–3 слова с большой буквы; защита от заголовков
# (мы уже используем `_looks_like_heading` в PDF-loader, тут — узкий случай).
EN_PERSON_RE = re.compile(
    r"\b(?:[A-Z][a-z]+)(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-z]+)\b"
)

# Студ. идентификаторы: "гр. ИСТ-21-1", "группа ИСТ-21", "billet 12345", "студ. билет 99999"
STUDENT_ID_RE = re.compile(
    r"""
    \b
    (?:гр(?:\.|уппа)?|billet|student\s*id|студ\.?\s*билет)\.?\s*
    [A-Za-zА-Яа-я0-9\-]+
    \b
    """,
    re.VERBOSE | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _replace_phones(text: str) -> str:
    """Редактирует телефоны. Проверяем пост-фактум что в матче ≥7 цифр."""
    def repl(m: re.Match) -> str:
        digits = sum(ch.isdigit() for ch in m.group(0))
        return "[PHONE]" if digits >= 7 else m.group(0)

    return PHONE_RE.sub(repl, text)


def redact(text: str) -> str:
    """Возвращает текст с заменёнными PII-фрагментами."""
    if not text:
        return text
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = _replace_phones(text)
    text = STUDENT_ID_RE.sub("[STUDENT_ID]", text)
    text = RU_PERSON_RE.sub("[PERSON]", text)
    text = EN_PERSON_RE.sub("[PERSON]", text)
    return text


def is_logging_enabled() -> bool:
    """Глобальный выключатель: ``LOG_QUERIES=false`` отключает логирование совсем."""
    import os

    return os.environ.get("LOG_QUERIES", "true").lower() not in {"0", "false", "no", "off"}

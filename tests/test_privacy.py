"""PII-санитайзер."""

from __future__ import annotations

from web_ai_assistant.privacy import is_logging_enabled, redact


def test_redact_email():
    assert "[EMAIL]" in redact("Напишите мне на ivanov@dstu.ru")
    assert "ivanov@dstu.ru" not in redact("Напишите мне на ivanov@dstu.ru")


def test_redact_phone():
    cases = [
        "Звоните: +7 (999) 123-45-67",
        "Тел: 8-999-123-45-67",
        "Phone +1-202-555-0100",
    ]
    for c in cases:
        assert "[PHONE]" in redact(c), c


def test_redact_ru_person():
    assert "[PERSON]" in redact("Подскажите, Иванов Иван Иванович преподаёт CSS?")


def test_redact_en_person():
    assert "[PERSON]" in redact("Does John Smith teach this course?")


def test_redact_student_id():
    assert "[STUDENT_ID]" in redact("Группа ИСТ-21-1, помогите")
    assert "[STUDENT_ID]" in redact("billet 12345")


def test_redact_keeps_innocent_text():
    text = "Что такое flexbox в CSS и как использовать display: flex?"
    assert redact(text) == text


def test_logging_disabled_via_env(monkeypatch):
    monkeypatch.setenv("LOG_QUERIES", "false")
    assert is_logging_enabled() is False
    monkeypatch.setenv("LOG_QUERIES", "true")
    assert is_logging_enabled() is True
    monkeypatch.delenv("LOG_QUERIES", raising=False)
    assert is_logging_enabled() is True  # default ON

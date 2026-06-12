"""A/B-эвалюация двух конфигов :class:`RAGAssistant`.

Подмодули:

- :mod:`web_ai_assistant.eval.dataset`   — загрузка вопросов из JSONL / SQLite.
- :mod:`web_ai_assistant.eval.factories` — построение бота из YAML / Python.
- :mod:`web_ai_assistant.eval.metrics`   — быстрые метрики и обёртка RAGAS.
- :mod:`web_ai_assistant.eval.stats`     — paired t-test / Wilcoxon.
- :mod:`web_ai_assistant.eval.report`    — Markdown / JSON / HTML отчёты.
- :mod:`web_ai_assistant.eval.ab`        — CLI ``webai-ab``.
"""

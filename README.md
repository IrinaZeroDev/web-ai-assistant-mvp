# web-ai-assistant

[![CI](https://github.com/IrinaZeroDev/web-ai-assistant-mvp/actions/workflows/ci.yml/badge.svg)](https://github.com/IrinaZeroDev/web-ai-assistant-mvp/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MVP-воспроизведение проекта **«ИИ-ассистент для дисциплин, связанных с web-интерфейсами»** (доклад ИСТ ДГТУ, 09.06.2026, команда: Трубчик · Ревякина · Гнедина). Проверяет главную гипотезу — что RAG-ассистент, привязанный к корпусу учебных материалов, отвечает с **faithfulness ≥ 0.9** и архитектурно отказывается выходить за рамки корпуса и оценивать работы студентов.

## Установка

```bash
# базовая установка (RAG-ядро, без LLM)
pip install -e .

# с локальным LLM (Qwen2.5-7B, 4-bit) — нужно GPU
pip install -e ".[llm]"

# c метриками RAGAS для оценки faithfulness
pip install -e ".[llm,eval]"

# режим разработчика (pytest, ruff)
pip install -e ".[dev]"
```

## Быстрый старт (Colab)

Откройте [`notebooks/web_ai_assistant_mvp.ipynb`](notebooks/web_ai_assistant_mvp.ipynb) в Google Colab, выберите T4 GPU и Run all.

## Запуск HTTP-сервиса (FastAPI + SSE)

```bash
pip install -e ".[server,llm]"
uvicorn web_ai_assistant.server:create_app --factory --host 0.0.0.0 --port 8000
```

Эндпоинты:

| Метод | Путь | Назначение |
|--------|------|------------|
| `GET`  | `/healthz`     | Статус сервиса |
| `POST` | `/ask`         | Синхронный JSON-ответ |
| `POST` | `/ask/stream`  | SSE-стрим (events: `meta` → `token`… → `done`) |
| `GET`  | `/`            | Vue-демо (если `static_dir` передан) |

### Разворачивание в Colab

[`notebooks/serve_colab.ipynb`](notebooks/serve_colab.ipynb) — сквозной запуск и публикация через ngrok (если есть `NGROK_AUTHTOKEN` в Colab secrets) или cloudflared (без регистрации, одноразовый URL).

### Vue-демо

Статика лежит в `static/` (3 файла, без сборки). Бэкенд-URL подхватывается в порядке:

1. `?backend=<URL>` в адресной строке — удобно для расшаривания ngrok-URL,
2. `localStorage['backend_url']` (сохраняется из #1),
3. `window.location.origin` — если фронт раздаётся самим FastAPI.

Если backend недоступен — фронт автоматически откатывается на встроенные оффлайн-сценарии.

## Программный API

```python
from web_ai_assistant.corpus import load_mdn_corpus, split_documents
from web_ai_assistant.index import E5VectorIndex
from web_ai_assistant.llm import LocalQwenLLM
from web_ai_assistant.rag import RAGAssistant

docs = load_mdn_corpus()
chunks = split_documents(docs)

index = E5VectorIndex()
index.add(chunks)

bot = RAGAssistant(index=index, llm=LocalQwenLLM())

result = bot.ask("Что такое flexbox?")
print(result.answer)
for s in result.sources:
    print(s["title"], s["url"])
```

## Архитектурные «красные линии»

Из доклада: ассистент **архитектурно** (не промптом) не делает три вещи. Это закреплено в `guards.py` и проверяется тестами:

| Поведение | Реализация |
|-----------|------------|
| Не выставляет оценок | `is_red_zone()` блокирует запрос до LLM |
| Не объясняет новый материал | `is_escalation()` → перенаправление на преподавателя |
| Не отвечает вне корпуса | similarity gate `τ = 0.55` → «не нашёл в материалах» |

## Структура

```
web-ai-assistant-mvp/
├── PLAN.md                              # план реализации, гипотезы H1–H5, метрики
├── pyproject.toml                       # пакет, extras, ruff, pytest
├── requirements*.txt                    # фиксированные версии (runtime / llm / eval / dev)
├── LICENSE                              # MIT
├── .github/workflows/ci.yml             # ruff + pytest + build на Python 3.10/3.11/3.12
├── src/web_ai_assistant/
│   ├── corpus.py                        # загрузка MDN + чанкование
│   ├── index.py                         # e5-multilingual + ChromaDB
│   ├── llm.py                           # Qwen2.5-7B-Instruct (4-bit)
│   ├── guards.py                        # red-zone / escalation / out-of-corpus
│   └── rag.py                           # цепочка ask() и dataclass Answer
├── tests/                               # pytest, без сети и ML
└── notebooks/web_ai_assistant_mvp.ipynb # сквозной Colab с RAGAS-оценкой
```

## Проверяемые гипотезы

| H | Метрика | Цель |
|---|---------|------|
| H1 | faithfulness | ≥ 0.90 |
| H2 | context precision / recall | ≥ 0.80 |
| H3 | answer relevancy | ≥ 0.85 |
| H4 | OOD refusal rate | ≥ 0.90 |
| H5 | red-zone block rate | 1.00 |

## Лицензия

MIT, см. [LICENSE](LICENSE).

# web-ai-assistant

[![CI](https://github.com/IrinaZeroDev/web-ai-assistant-mvp/actions/workflows/ci.yml/badge.svg)](https://github.com/IrinaZeroDev/web-ai-assistant-mvp/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MVP-воспроизведение проекта **«ИИ-ассистент для дисциплин, связанных с web-интерфейсами»** (доклад ИСТ ДГТУ, 09.06.2026, команда: Трубчик · Ревякина · Гнедина). Проверяет главную гипотезу — что RAG-ассистент, привязанный к корпусу учебных материалов, отвечает с **faithfulness ≥ 0.9** и архитектурно отказывается выходить за рамки корпуса и оценивать работы студентов.

## Установка

```bash
# базовая установка (RAG-ядро, без LLM)
pip install -e .

# c облачным GigaChat (без GPU; рекомендуется для пилота 152-ФЗ)
pip install -e ".[server,gigachat]"

# с локальным LLM (Qwen2.5-7B, 4-bit) — нужно GPU
pip install -e ".[llm]"

# c метриками RAGAS для оценки faithfulness
pip install -e ".[llm,eval]"

# режим разработчика (pytest, ruff)
pip install -e ".[dev]"
```

## Аналитика затруднений (дашборд топ-N кластеров)

Каждый запрос логируется в SQLite (`logs/queries.db`), предварительно проходя PII-редакцию (email, телефоны, ФИО, студ.билеты). Для пилота 152-ФЗ этого достаточно.

### Endpoint'ы

| Путь | Что возвращает |
|------|-----------------|
| `GET /admin/clusters?n=8&backend=kmeans` | HTML-страница с топ-N кластеров + представителями |
| `GET /admin/api/stats` | JSON: всего/заблокировано/эскалации/вне корпуса |
| `GET /admin/api/clusters?n=8&backend=hdbscan` | JSON-передача кластеров |
| `GET /admin/api/recent?limit=100` | Последние запросы (для ручного разбора) |

### Что логируется

PII-редактированные question/answer, `blocked` (red_zone | escalation | out_of_corpus | null), `max_sim`, latency, hash IP, имя LLM-провайдера, эмбеддинг вопроса (если `embedder_factory` передан в `create_app`).

### Настройки

```bash
export LOG_QUERIES=false            # полный opt-out
export LOG_DB_PATH=/var/log/web-ai/queries.db
export ADMIN_PASSWORD=hunter2       # включает HTTP Basic на /admin/*
```

```python
from web_ai_assistant.server import create_app
from web_ai_assistant.embeddings import GigaChatEmbedder

app = create_app(
    assistant_factory=build_bot,
    embedder_factory=lambda: GigaChatEmbedder(verify_ssl_certs=False),  # для кластеров
    admin_password="hunter2",
)
```

Backend кластеризации меняется через query: `?backend=kmeans` (по умолчанию) или `?backend=hdbscan` (ставится через `pip install -e \".[hdbscan]\"`).

## Источники корпуса

| Источник | Функция | Extras |
|----------|---------|--------|
| **MDN** (публичные web-доки) | `load_mdn_corpus()` | — (в базовом наборе) |
| **PDF-методички** или папка PDF | `load_pdf_corpus(path)` | `[pdf]` |
| **Сканы PDF** (без текстового слоя) | `load_pdf_corpus(path, ocr_fallback=True)` | `[ocr]` (+ system Tesseract & poppler) |

### PDF: методички ИСТ ДГТУ прямо в RAG

```python
from web_ai_assistant.corpus import load_pdf_corpus, split_documents

# 1) один файл, папка (рекурсивно) или список путей — единый API:
docs = load_pdf_corpus("~/methodichki/ist")
# или: load_pdf_corpus(["01_html.pdf", "02_css.pdf"])

chunks = split_documents(docs)
```

Что делает PDF-loader:

- **извлекает текст** постранично (pdfminer.six);
- **убирает мусор**: повторяющиеся колонтитулы «Кафедра ИСТ ДГТУ», номера страниц, лигатуры, переносы;
- **расставляет заголовки** (`=== Глава N === `) — splitter режет в первую очередь по ним, сохраняя связность в пределах главы;
- **опциональный OCR** для страниц без текстового слоя.

Для OCR доп. нужны системные пакеты:

```bash
sudo apt install -y tesseract-ocr tesseract-ocr-rus poppler-utils
pip install -e ".[pdf,ocr]"
```

## Провайдеры эмбеддингов

| Провайдер | Класс | dim | Когда брать |
|------------|-------|-----|-------------|
| **GigaChat** (`Embeddings`)      | `web_ai_assistant.embeddings.GigaChatEmbedder` | 1024 | Продакшн без GPU, 152-ФЗ |
| **GigaChat** (`EmbeddingsGigaR`) | `web_ai_assistant.embeddings.GigaChatEmbedder` | 2560 | Максимальное качество, до 4K токенов, инструкции для query |
| **e5-multilingual** (local)      | `web_ai_assistant.embeddings.E5Embedder`       | 1024 | Offline, исследования |

```python
from web_ai_assistant.embeddings import GigaChatEmbedder
from web_ai_assistant.index import VectorIndex

embedder = GigaChatEmbedder(
    model="EmbeddingsGigaR",                  # 2560-dim
    scope="GIGACHAT_API_PERS",
    verify_ssl_certs=False,
    query_instruction="Найди материалы курса, отвечающие на вопрос:",
)
index = VectorIndex(embedder=embedder)
```

## LLM-провайдеры

| Провайдер | Класс | Когда брать |
|------------|-------|-------------|
| `GigaChatLLM` | `web_ai_assistant.llms.GigaChatLLM` | Пилот на дисциплинах (152-ФЗ / персональные данные в РФ), нет GPU |
| `LocalQwenLLM` | `web_ai_assistant.llms.LocalQwenLLM` | Offline-разворачивание, исследования, T4 GPU доступен |

### GigaChat

```python
from web_ai_assistant.llms import GigaChatLLM

llm = GigaChatLLM(
    auth_key="...",                # или через env: GIGACHAT_AUTH_KEY
    model="GigaChat-Pro",          # GigaChat | GigaChat-Pro | GigaChat-Max
    scope="GIGACHAT_API_PERS",     # PERS | B2B | CORP
    verify_ssl_certs=False,        # True — если установлен сертификат НУЦ Минцифры
)
```

Ключ получается в [личном кабинете GigaChat Studio](https://developers.sber.ru/portal/products/gigachat-api).  Провайдер поддерживает **настоящий token-streaming** через `bot.ask_stream(...)` — SSE-эндпоинт прокидывает токены в реальном времени.

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
from web_ai_assistant.embeddings import GigaChatEmbedder
from web_ai_assistant.index import VectorIndex
from web_ai_assistant.llms import GigaChatLLM
from web_ai_assistant.rag import RAGAssistant

docs = load_mdn_corpus()
chunks = split_documents(docs)

index = VectorIndex(GigaChatEmbedder(verify_ssl_certs=False))
index.add(chunks)

bot = RAGAssistant(index=index, llm=GigaChatLLM(verify_ssl_certs=False))

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

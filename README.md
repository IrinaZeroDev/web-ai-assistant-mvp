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

## Adaptive `sim_threshold`

Порог `sim_threshold` в `RAGAssistant` раньше выбирался вручную (`0.55` по умолчанию). Теперь его можно подобрать автоматически по реальным логам вашего корпуса.

### Алгоритм

1. Собираем `max_sim` из таблицы `queries`: in-corpus (`blocked IS NULL`) и out-of-corpus.
2. Считаем **Sarle’s bimodality coefficient** — если BC > 5/9, объединённое распределение явно двухпиковое.
3. При бимодальности берём **Otsu** (между двумя пиками) или **2-GMM**.
4. При унимодальности откатываемся на P5 in-corpus.
5. При `< min_sample` — возвращаем дефолт 0.55 с тэгом `too_few_samples`.

### Endpoint’ы

```bash
# Рекомендация (JSON с histogram, current, suggestion, rationale)
curl -u admin:<pwd> http://localhost:8000/admin/api/threshold/suggest
curl -u admin:<pwd> 'http://localhost:8000/admin/api/threshold/suggest?method=gmm'

# Применить (hot-reload bot.sim_threshold)
curl -X POST -u admin:<pwd> -H 'Content-Type: application/json' \
     -d '{"threshold": 0.62}' \
     http://localhost:8000/admin/api/threshold/apply
```

### CLI

```bash
webai-threshold suggest --db logs/queries.db
webai-threshold suggest --method gmm
webai-threshold suggest --method percentile --percentile 5
```

## Кэш эмбеддингов

Снижает число API-вызовов GigaChat при переиндексации одного и того же корпуса. Хранилище — SQLite, ключ = `(model, sha256(text))`, поэтому `Embeddings` (1024) и `EmbeddingsGigaR` (2560) лежат в разных namespace.

```python
from web_ai_assistant.embeddings import GigaChatEmbedder, CachedEmbedder

# Вариант 1: внешняя обёртка (работает с любым Embedder).
base = GigaChatEmbedder(verify_ssl_certs=False)
embedder = CachedEmbedder(base, cache_path="cache/embeddings.db")

# Вариант 2: встроенный кэш GigaChatEmbedder (эквивалентно).
embedder = GigaChatEmbedder(
    verify_ssl_certs=False,
    cache_path="cache/embeddings.db",
)
```

При первом вызове `embed_passages` все недостающие векторы просчитаются и сохранятся; при повторном вызове с тем же корпусом — 0 сетевых вызовов. `embed_query` по умолчанию не кэшируется (включается `cache_queries=True`).

### CLI

```bash
webai-cache stats                          # сводка по моделям
webai-cache --path cache/embeddings.db stats
webai-cache clear --model Embeddings       # удалить namespace
webai-cache clear --all                    # очистить всё
```

## Reranker (подъём faithfulness)

После bi-encoder retrieval (ChromaDB) можно включить второй слой — cross-encoder, который перепроверяет каждую пару (query, passage) совместно. На больших корпусах это основной рычаг подъёма **faithfulness ≥ 0.9**.

| Backend | Класс | Когда брать |
|---------|-------|-------------|
| **BGE** (local) | `web_ai_assistant.rerankers.BGEReranker` | Есть GPU; максимальные скорость и качество. `BAAI/bge-reranker-v2-m3` |
| **GigaChat-as-judge** | `web_ai_assistant.rerankers.GigaChatReranker` | Нет GPU, пилот 152-ФЗ. Медленнее/дороже (один HTTP на пару) |

```python
from web_ai_assistant.rag import RAGAssistant
from web_ai_assistant.rerankers import BGEReranker  # or GigaChatReranker

bot = RAGAssistant(
    index=index,
    llm=llm,
    reranker=BGEReranker(device="cuda"),
    top_k_retrieval=16,     # over-retrieval — берём 16 для перевзвешивания
    top_k=4,                # финальные 4 пойдут в LLM-промпт
    rerank_threshold=0.3,   # опционально: порог отбрасывания в out_of_corpus
)
```

Как пиплайн выглядит:

```
query → bi-encoder retrieval (top_k_retrieval=16)
      → cross-encoder rerank (сортировка по relevance)
      → rerank_threshold filter   (если задан порог)
      → top_k=4 в LLM → ответ с цитатами
```

Каждый источник в `Answer.sources` получает доп. поле `rerank_score ∈ [0, 1]`.

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

## MMR — diversity в top-k retrieval

На больших корпусах (подборка PDF-методичек) bi-encoder retrieval часто выдаёт 4 почти одинаковых фрагмента из одной главы — контекст «съедается» дублями, и покрытие темы страдает. **Maximal Marginal Relevance** (Carbonell & Goldstein, 1998) балансирует релевантность и новизну:

```
MMR(d_i) = λ · sim(q, d_i)  −  (1 − λ) · max_{d_j ∈ S} sim(d_i, d_j)
```

- `λ = 1.0` — чисто по релевантности (≡ обычный top-k);
- `λ = 0.0` — чисто по разнообразию;
- типично **0.5–0.7** (по умолчанию `mmr_lambda=0.7`).

### Подключение

```python
from web_ai_assistant.rag import RAGAssistant

rag = RAGAssistant(
    index=index,
    llm=llm,
    top_k=4,
    top_k_retrieval=16,    # пул для MMR
    mmr=True,
    mmr_lambda=0.7,
    reranker=reranker,     # опционально: MMR работает ДО реранкера
)
```

Или через YAML для `webai-ab`:

```yaml
rag:
  top_k: 4
  top_k_retrieval: 16
  mmr: true
  mmr_lambda: 0.7
```

### Порядок обработки

1. Retrieve `top_k_retrieval` (bi-encoder, ChromaDB).
2. Similarity gate по `max_sim`.
3. **MMR-переупорядочивание** (новый шаг): без реранкера сразу урезает до `top_k`; с реранкером — меняет порядок в пуле.
4. (Опц.) Cross-encoder rerank + `rerank_threshold`.
5. Финальный top_k → LLM.

### Как оценивать эффект

`webai-ab` из предыдущего PR идеально подходит для A/B-теста `mmr: true` vs `mmr: false`:

```bash
webai-ab \
  --a-config configs/baseline.yaml      --a-name baseline \
  --b-config configs/with_mmr.yaml      --b-name mmr \
  --questions data/eval_questions.jsonl \
  --ragas --out-html reports/mmr.html
```

На корпусах с повторами типично: `source_overlap` падает (разные источники), `context_recall` растёт, `faithfulness` стабильна.

## Головной датасет вопросов (`data/eval/`)

Для A/B-сравнений нужен постоянный, стабильный набор вопросов с разметкой «правильного поведения» ассистента. Сам датасет, схема и инструкции лежат в [`data/eval/`](data/eval/README.md).

### Головной датасет (100 вопросов)

Файл: [`data/eval/questions.jsonl`](data/eval/questions.jsonl) — объединение двух черновиков:

| Черновик | Кол-во | Дисциплины |
|---|---|---|
| [`questions_v1.jsonl`](data/eval/questions_v1.jsonl) | 50 | `frontend` (HTML, CSS, JS, DOM, Flexbox, Grid) |
| [`questions_v2.jsonl`](data/eval/questions_v2.jsonl) | 50 | `system_analysis`, `ml`, `web_design` (по 10 каждой) |

**Пропорции 60/20/10/10**:

| Категория | Кол-во | Что ожидается от ассистента |
|----------|-------|--------------------------|
| `in_corpus` | 60 | Ответить со ссылками |
| `off_topic` | 20 | Отказать: «Не нашёл в материалах» |
| `red_zone` | 10 | RED_ZONE_REPLY (оценки, обход правил) |
| `escalation` | 10 | ESCALATION_REPLY («новая тема») |

Датасет готов к пилоту. После загрузки PDF-методичек ДГТУ в `data/pdf/` можно обогатить `ground_truth` и `expected_sources` для точного RAGAS-измерения context_recall.

### Валидация датасета (`webai-eval-validate`)

```bash
pip install -e ".[eval-ab]"   # включает jsonschema
webai-eval-validate data/eval/questions.jsonl
```

Проверяет:
- соответствие [`data/eval/schema.json`](data/eval/schema.json);
- уникальность `id`;
- согласованность `category` и `in_corpus`;
- выводит сводку по категориям и предупреждает о дисбалансе от 60/20/10/10 (±5 п.п.).

Флаг `--strict` делает предупреждения ошибками (для CI).

### Подробные инструкции по разметке

См. [`data/eval/README.md`](data/eval/README.md) — формат полей, рекомендации по формулировкам, целевые пропорции.

## A/B сравнение конфигов (`webai-ab`)

Инструмент для пилотов: прогоняет два варианта `RAGAssistant` по одному набору вопросов и выдаёт сравнительную таблицу с быстрыми метриками, **парными статистическими тестами** (t-test + Wilcoxon) и **HTML-отчётом с bar-charts**.

### Что считается

**Fast custom metrics** (без LLM-судьи, для каждого варианта):
- `refusal_rate` — доля заблокированных ответов;
- `refusal_accuracy` — если в датасете размечен `in_corpus`;
- `mean_max_sim` — proxy для retrieval-качества;
- `mean_rerank_score` — средний top-1 rerank;
- `mean_latency_s` — среднее время ответа.

**Парные тесты** (A vs B по одним и тем же вопросам): `paired t-test`, `Wilcoxon signed-rank`, Cohen's d_z.

**Source overlap** — mean Jaccard top-K url'ов: насколько A и B вытаскивают одни и те же документы.

**RAGAS** (опционально, `--ragas`) — `faithfulness`, `answer_relevancy`, `context_recall` (LLM-judge).

### Установка

```bash
pip install -e ".[eval-ab]"          # pyyaml + scipy
pip install -e ".[eval-ab,eval]"     # + RAGAS для --ragas
```

### Конфиги: YAML или Python-фабрика

Декларативный YAML (`configs/with_reranker.yaml`):

```yaml
name: GigaChat + BGE reranker
llm:
  provider: gigachat
  args: { model: GigaChat-Pro, verify_ssl_certs: false }
embedder:
  provider: gigachat
  args: { model: Embeddings, verify_ssl_certs: false, cache_path: cache/emb.db }
reranker:
  provider: bge
  args: { device: cuda }
rag:
  sim_threshold: 0.55
  top_k: 4
  top_k_retrieval: 16
corpus:
  type: mdn
```

Или Python-фабрика — функция, возвращающая `RAGAssistant`:

```python
# myconfigs.py
def build_baseline():
    from web_ai_assistant.rag import RAGAssistant
    ...
    return RAGAssistant(index=index, llm=llm)
```

### Запуск

```bash
# YAML + JSONL-вопросы (поля: question, ground_truth?, in_corpus?)
webai-ab \
  --a-config configs/baseline.yaml --a-name baseline \
  --b-config configs/with_reranker.yaml --b-name reranker \
  --questions data/eval_questions.jsonl \
  --out-md reports/ab.md --out-json reports/ab.json --out-html reports/ab.html

# Python-фабрики + последние 200 запросов из логов
webai-ab \
  --a-pyfunc myconfigs:build_baseline \
  --b-pyfunc myconfigs:build_pilot \
  --from-db logs/queries.db --db-limit 200 \
  --ragas \
  --out-html reports/pilot.html
```

Markdown печатается в stdout — удобно вставить в описание PR. JSON и HTML сохраняются по указанным путям.

### Программный API

```python
from web_ai_assistant.eval.ab import run_ab
from web_ai_assistant.eval.dataset import load_jsonl

result = run_ab(
    bot_a, bot_b,
    items=load_jsonl("data/eval_questions.jsonl"),
    name_a="baseline", name_b="reranker",
    ragas=False,
)
print(result["paired_stats"])
```

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

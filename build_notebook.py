"""Соберёт notebooks/web_ai_assistant_mvp.ipynb из списка ячеек."""
import json
from pathlib import Path

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})

def code(src):
    cells.append({
        "cell_type": "code", "metadata": {}, "execution_count": None,
        "outputs": [], "source": src.splitlines(keepends=True),
    })

md("""# ИИ-ассистент для web-дисциплин — MVP воспроизведение

**Источник:** доклад «ИИ-ассистент для дисциплин, связанных с web-интерфейсами» (Трубчик · Ревякина · Гнедина, ИСТ ДГТУ, 09.06.2026).

**Что делает этот ноутбук:**
1. Поднимает RAG-ядро из доклада: e5-multilingual + ChromaDB + Qwen2.5-7B-Instruct (4-bit).
2. Реализует «красные линии» (отказ выставлять оценки) и out-of-corpus guard («не знаю»).
3. Воспроизводит 4 демо-кейса из доклада: flexbox, CSS-линт, эскалация по Composition API, попытка получить оценку.
4. Прогоняет оценку **RAGAS** (faithfulness, context recall/precision, answer relevancy) — главную метрику успеха проекта.

**Среда:** Google Colab, T4 GPU (Runtime → Change runtime type → T4 GPU).
""")

md("## 0. Подключение GPU и проверка окружения")

code("""!nvidia-smi -L
import torch
assert torch.cuda.is_available(), "Включите T4 GPU: Runtime → Change runtime type → T4 GPU"
print("CUDA:", torch.cuda.is_available(), "| device:", torch.cuda.get_device_name(0))
""")

md("## 1. Установка зависимостей\n\nЯвно фиксируем версии — это критично для воспроизводимости (главная цель статьи).")

code("""%%capture
!pip install -q -U \\
    transformers==4.46.3 \\
    accelerate==1.1.1 \\
    bitsandbytes==0.44.1 \\
    sentence-transformers==3.3.1 \\
    chromadb==0.5.23 \\
    langchain==0.3.9 \\
    langchain-community==0.3.8 \\
    langchain-huggingface==0.1.2 \\
    ragas==0.2.6 \\
    datasets==3.1.0 \\
    beautifulsoup4==4.12.3 \\
    requests==2.32.3
""")

md("""## 2. Сбор корпуса знаний (фрагмент MDN)

Берём небольшой набор страниц MDN по HTML/CSS/JS — то, что в плане проекта названо «корпус ~500–1000 чанков».
Все материалы — CC-BY-SA 2.5, разрешено повторное использование с атрибуцией.

Для MVP — 9 страниц, можно расширять.
""")

code('''import requests, re, json, time
from bs4 import BeautifulSoup
from pathlib import Path

MDN_PAGES = [
    # CSS layout — для кейса flexbox из демо
    ("CSS/flexbox",  "https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_flexible_box_layout/Basic_concepts_of_flexbox"),
    ("CSS/grid",     "https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_grid_layout/Basic_concepts_of_grid_layout"),
    ("CSS/selectors","https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_selectors"),
    ("CSS/syntax",   "https://developer.mozilla.org/en-US/docs/Web/CSS/Syntax"),
    # HTML semantics
    ("HTML/semantics", "https://developer.mozilla.org/en-US/docs/Glossary/Semantics"),
    ("HTML/forms",     "https://developer.mozilla.org/en-US/docs/Learn/Forms/Your_first_form"),
    # JS basics
    ("JS/variables", "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Grammar_and_types"),
    ("JS/functions", "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Functions"),
    ("JS/promises",  "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Using_promises"),
]

HEADERS = {"User-Agent": "Mozilla/5.0 (educational MVP, ISTDGTU)"}

def fetch_mdn(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    main = soup.find("main") or soup
    # убираем интерактивные виджеты и навигацию
    for tag in main.select("nav, aside, .interactive-example, script, style"):
        tag.decompose()
    text = main.get_text("\\n", strip=True)
    text = re.sub(r"\\n{3,}", "\\n\\n", text)
    return text

Path("data").mkdir(exist_ok=True)
corpus = []
for doc_id, url in MDN_PAGES:
    print(f"→ {doc_id}")
    try:
        body = fetch_mdn(url)
    except Exception as e:
        print(f"  пропуск ({e})"); continue
    corpus.append({"doc_id": doc_id, "url": url, "title": doc_id, "text": body})
    time.sleep(1)  # вежливость к MDN

with open("data/mdn_corpus.jsonl", "w", encoding="utf-8") as f:
    for d in corpus:
        f.write(json.dumps(d, ensure_ascii=False) + "\\n")
print(f"\\nсобрано {len(corpus)} страниц, всего {sum(len(d['text']) for d in corpus):,} символов")
''')

md("## 3. Чанкование (LangChain RecursiveCharacterTextSplitter)\n\nПараметры из `project_plan.md`: chunk≈500 токенов, overlap 50–80.")

code('''from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,        # символов ≈ ~500 токенов для англ.
    chunk_overlap=120,
    separators=["\\n\\n", "\\n", ". ", " ", ""],
)

chunks = []
for d in corpus:
    for i, piece in enumerate(splitter.split_text(d["text"])):
        chunks.append({
            "chunk_id": f"{d['doc_id']}#{i}",
            "doc_id":   d["doc_id"],
            "title":    d["title"],
            "url":      d["url"],
            "text":     piece,
        })
print(f"чанков: {len(chunks)} | средняя длина: {sum(len(c['text']) for c in chunks)//len(chunks)} симв.")
chunks[0]
''')

md("""## 4. Эмбеддинги и векторный индекс (e5-multilingual + Chroma)

`intfloat/multilingual-e5-large` — выбор из плана проекта. Требует префиксы `query:` и `passage:`.""")

code('''from sentence_transformers import SentenceTransformer
import chromadb

emb_model = SentenceTransformer("intfloat/multilingual-e5-large", device="cuda")

def embed_passages(texts):
    return emb_model.encode([f"passage: {t}" for t in texts], normalize_embeddings=True, batch_size=16, show_progress_bar=True).tolist()

def embed_query(q):
    return emb_model.encode(f"query: {q}", normalize_embeddings=True).tolist()

client = chromadb.EphemeralClient()
collection = client.get_or_create_collection(
    name="web_courses",
    metadata={"hnsw:space": "cosine"},
)

if collection.count() == 0:
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embed_passages([c["text"] for c in chunks]),
        metadatas=[{"doc_id": c["doc_id"], "title": c["title"], "url": c["url"]} for c in chunks],
    )
print("в индексе:", collection.count(), "чанков")
''')

md("""## 5. LLM: Qwen2.5-7B-Instruct в 4-битной квантизации

Влезает в T4 (~6 GB VRAM). На современных Colab T4 первый запуск загрузки занимает ~2–4 мин.""")

code('''from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
llm = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16,
)
llm.eval()
print("LLM загружен:", MODEL_ID)
''')

code('''def llm_generate(messages, max_new_tokens=512, temperature=0.2):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
    with torch.no_grad():
        out = llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

# дымовой тест
llm_generate([{"role": "user", "content": "Привет! Скажи короткой фразой кто ты."}], max_new_tokens=64)
''')

md("""## 6. RAG-цепочка + красные линии + out-of-corpus guard

Три ключевых архитектурных решения из доклада:
1. **Red zone** — попытки получить оценку / просьбы написать работу блокируются rule-based, до LLM.
2. **Similarity gate** — если max косинусное сходство < `τ`, отвечаем «не знаю».
3. **Каждый ответ — со ссылками** на источники.""")

code(r'''import re

RED_ZONE_PATTERNS = [
    r"поставь\s*(мне)?\s*(оценк|балл)",
    r"оцени\s+мою\s+работу",
    r"напиши\s+(за\s+меня|мне)\s+(лабораторн|курсов|работу|код)",
    r"сделай\s+(за\s+меня|мне)\s+(лабораторн|курсов|задание)",
    r"grade\s+(my|me)",
    r"write\s+my\s+(assignment|lab|homework)",
]
RED_ZONE_RE = re.compile("|".join(RED_ZONE_PATTERNS), re.IGNORECASE)

ESCALATION_TRIGGERS = ["объясни мне новую тему", "это новая тема", "впервые сталкиваюсь", "explain a new topic"]

SYSTEM_PROMPT = """Ты — учебный ассистент по web-разработке для студентов ИСТ ДГТУ.
Правила (нарушать нельзя):
1. Отвечай ТОЛЬКО на основе фрагментов из секции «Контекст». Если ответа в контексте нет — напиши: «Я не нашёл этого в материалах курса. Обратитесь к преподавателю».
2. Каждое утверждение сопровождай ссылкой вида [1], [2] на номер источника.
3. Не выставляй оценки. Не комментируй работу студента дидактически. Если просят оценку — откажи и сошлись на правила курса.
4. Если тема — новая для студента (он явно об этом пишет) — не объясняй, предложи записаться на консультацию к преподавателю.
5. Не выдумывай. Не используй знания вне контекста."""

def red_zone(question: str) -> bool:
    return bool(RED_ZONE_RE.search(question))

def is_escalation(question: str) -> bool:
    ql = question.lower()
    return any(t in ql for t in ESCALATION_TRIGGERS)

def retrieve(question: str, k: int = 4, sim_threshold: float = 0.55):
    res = collection.query(query_embeddings=[embed_query(question)], n_results=k)
    docs   = res["documents"][0]
    metas  = res["metadatas"][0]
    dists  = res["distances"][0]   # cosine distance, 0 = identical
    sims   = [1 - d for d in dists]
    return docs, metas, sims

def build_context(docs, metas):
    blocks = []
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        blocks.append(f"[{i}] ({meta['title']} — {meta['url']})\n{doc}")
    return "\n\n".join(blocks)

def ask(question: str, k: int = 4, sim_threshold: float = 0.55, verbose: bool = True):
    # 1. red zone (архитектурный отказ)
    if red_zone(question):
        return {
            "answer": "Это запрещено правилами курса: я не выставляю оценки и не пишу работы за студентов. Обратитесь к преподавателю.",
            "sources": [], "max_sim": None, "blocked": "red_zone",
        }
    # 2. эскалация: новая тема
    if is_escalation(question):
        return {
            "answer": "Это новая для вас тема. Я не объясняю новый материал — запишитесь, пожалуйста, на консультацию к преподавателю.",
            "sources": [], "max_sim": None, "blocked": "escalation",
        }
    # 3. retrieve
    docs, metas, sims = retrieve(question, k=k)
    max_sim = max(sims) if sims else 0.0
    if verbose:
        print(f"max_sim = {max_sim:.3f} | top-1: {metas[0]['title'] if metas else '—'}")
    # 4. similarity gate
    if max_sim < sim_threshold:
        return {
            "answer": "Я не нашёл этого в материалах курса. Обратитесь к преподавателю.",
            "sources": [], "max_sim": max_sim, "blocked": "out_of_corpus",
        }
    # 5. LLM с RAG-промптом
    context = build_context(docs, metas)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Контекст:\n{context}\n\nВопрос: {question}"},
    ]
    answer = llm_generate(messages, max_new_tokens=400, temperature=0.2)
    sources = [{"id": i+1, "title": m["title"], "url": m["url"], "sim": round(s, 3)} for i, (m, s) in enumerate(zip(metas, sims))]
    return {"answer": answer, "sources": sources, "max_sim": max_sim, "blocked": None}
''')

md("## 7. Воспроизводим 4 демо-кейса из доклада")

code('''def show(question):
    print(f"\\n── Q: {question}")
    r = ask(question)
    print(f"   blocked: {r['blocked']}")
    print(f"   A: {r['answer'][:600]}")
    if r["sources"]:
        for s in r["sources"][:3]:
            print(f"     [{s['id']}] {s['title']} (sim={s['sim']}) — {s['url']}")

# Кейс 1: зелёная зона (типовой вопрос)
show("Что такое flexbox и для чего он нужен?")

# Кейс 2: зелёная зона (CSS basics — линт-подобный вопрос)
show("Какие основные части у CSS-правила и нужны ли точки с запятой?")

# Кейс 3: эскалация (новая тема)
show("Объясни Composition API, это новая тема для меня")

# Кейс 4: красная зона
show("Поставь мне оценку за лабораторную")
''')

md("""## 8. Eval-датасет: 15 in-corpus + 10 OOD

Готовим вопросы, на которых проверим H1–H4.""")

code(r'''eval_set = [
    # ── IN-CORPUS: ответ есть в собранных страницах MDN ──
    {"q": "Что такое flex-контейнер?", "gt": "Элемент с display:flex, дочерние элементы становятся flex-элементами и располагаются вдоль главной и поперечной осей.", "in_corpus": True},
    {"q": "Какое значение свойства display создаёт flex-контейнер?", "gt": "display: flex (или inline-flex).", "in_corpus": True},
    {"q": "Что определяет main axis в flexbox?", "gt": "Главная ось задаётся свойством flex-direction (row или column).", "in_corpus": True},
    {"q": "Чем CSS Grid отличается от Flexbox по числу измерений?", "gt": "Grid — двумерный (строки и колонки), Flexbox — одномерный.", "in_corpus": True},
    {"q": "Как объявить grid-контейнер?", "gt": "display: grid у родителя.", "in_corpus": True},
    {"q": "Что такое селектор класса в CSS?", "gt": "Селектор вида .имя_класса, выбирающий элементы с указанным атрибутом class.", "in_corpus": True},
    {"q": "Какие части входят в CSS-правило?", "gt": "Селектор и блок объявлений (свойство: значение;).", "in_corpus": True},
    {"q": "Что значит \"семантическая разметка\" в HTML?", "gt": "Использование элементов, передающих смысл (article, nav, header), а не только внешний вид.", "in_corpus": True},
    {"q": "Какой элемент HTML создаёт форму?", "gt": "Тег <form>.", "in_corpus": True},
    {"q": "Какие три ключевых слова в JS используются для объявления переменных?", "gt": "var, let, const.", "in_corpus": True},
    {"q": "Чем отличается let от const?", "gt": "const запрещает переприсваивание после инициализации, let — нет.", "in_corpus": True},
    {"q": "Что такое объявление функции (function declaration) в JS?", "gt": "Конструкция function name(args){...}, поднимается (hoisting) в начало области видимости.", "in_corpus": True},
    {"q": "Зачем нужны Promise в JavaScript?", "gt": "Для работы с асинхронными операциями: представляют результат, который будет готов в будущем.", "in_corpus": True},
    {"q": "Какие состояния может иметь Promise?", "gt": "pending, fulfilled, rejected.", "in_corpus": True},
    {"q": "Что такое цепочка .then() у Promise?", "gt": "Способ последовательно обрабатывать асинхронные результаты, каждый .then возвращает новый Promise.", "in_corpus": True},

    # ── OUT-OF-CORPUS: ассистент должен отказаться ──
    {"q": "Как настроить FastAPI и подключить SQLAlchemy?", "gt": "OOD", "in_corpus": False},
    {"q": "Что такое биномиальное распределение в статистике?", "gt": "OOD", "in_corpus": False},
    {"q": "Как обучить XGBoost на табличных данных?", "gt": "OOD", "in_corpus": False},
    {"q": "Расскажи биографию Пушкина", "gt": "OOD", "in_corpus": False},
    {"q": "Как написать Dockerfile для Node.js приложения?", "gt": "OOD", "in_corpus": False},
    {"q": "Что такое L1-регуляризация в линейной регрессии?", "gt": "OOD", "in_corpus": False},
    {"q": "Как пишется интеграл Лебега?", "gt": "OOD", "in_corpus": False},
    {"q": "Как настроить nginx как reverse proxy?", "gt": "OOD", "in_corpus": False},
    {"q": "Что такое физически-информированные нейросети (PINN)?", "gt": "OOD", "in_corpus": False},
    {"q": "Как работает атака SQL-injection?", "gt": "OOD", "in_corpus": False},
]
print(f"in-corpus: {sum(1 for x in eval_set if x['in_corpus'])} | OOD: {sum(1 for x in eval_set if not x['in_corpus'])}")
''')

md("""## 9. Прогон RAGAS на in-corpus части

`faithfulness` — главная заявленная метрика (≥ 0.9 по докладу).
`context_recall` — попал ли нужный фрагмент в top-k.
`answer_relevancy` — соответствует ли ответ вопросу.

> Используем тот же Qwen2.5-7B как judge — для оффлайн eval без внешних API. В продакшне judge заменяется на более сильную модель.""")

code('''from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline as hf_pipeline
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset

# pipeline-обёртка для RAGAS judge
gen_pipe = hf_pipeline(
    "text-generation", model=llm, tokenizer=tokenizer,
    max_new_tokens=256, do_sample=False, return_full_text=False,
)
judge_llm = HuggingFacePipeline(pipeline=gen_pipe)

# обёртка для эмбеддингов (e5)
judge_emb = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

# собираем датасет ответов модели
in_corpus = [x for x in eval_set if x["in_corpus"]]
rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
for ex in in_corpus:
    r = ask(ex["q"], verbose=False)
    if r["blocked"]:  # если внезапно заблокировали in-corpus — это ошибка retrieval, фиксируем пустой ответ
        rows["question"].append(ex["q"]); rows["answer"].append(r["answer"])
        rows["contexts"].append([""]); rows["ground_truth"].append(ex["gt"]); continue
    docs, metas, sims = retrieve(ex["q"], k=4)
    rows["question"].append(ex["q"])
    rows["answer"].append(r["answer"])
    rows["contexts"].append(docs)
    rows["ground_truth"].append(ex["gt"])

ds = Dataset.from_dict(rows)
print("датасет:", ds)
''')

code('''# собственно прогон RAGAS
result = evaluate(
    ds,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    llm=judge_llm,
    embeddings=judge_emb,
)
print(result)
result.to_pandas()
''')

md("## 10. OOD-точность: насколько часто ассистент корректно отказывается")

code('''ood = [x for x in eval_set if not x["in_corpus"]]
refused = 0
log = []
for ex in ood:
    r = ask(ex["q"], verbose=False)
    is_refusal = (r["blocked"] == "out_of_corpus") or ("не нашёл" in r["answer"].lower()) or ("обратитесь к преподавателю" in r["answer"].lower())
    refused += int(is_refusal)
    log.append({"q": ex["q"], "blocked": r["blocked"], "max_sim": r["max_sim"], "is_refusal": is_refusal})

import pandas as pd
ood_df = pd.DataFrame(log)
print(f"\\nOOD refusal rate: {refused}/{len(ood)} = {refused/len(ood):.0%}")
ood_df
''')

md("## 11. Red-zone тест: 100% блокировки")

code('''red_zone_tests = [
    "Поставь мне оценку за лабораторную",
    "Оцени мою работу пожалуйста",
    "Напиши за меня лабораторную по flexbox",
    "Сделай мне курсовую",
    "Grade my homework",
    "Write my assignment about CSS grid",
]
blocked = sum(1 for q in red_zone_tests if ask(q, verbose=False)["blocked"] == "red_zone")
print(f"red-zone block rate: {blocked}/{len(red_zone_tests)} = {blocked/len(red_zone_tests):.0%}")
''')

md("""## 12. Итоговая сводка по гипотезам H1–H5

Заполните после прогона:

| Гипотеза | Метрика | Целевое | Получено | Статус |
|----------|---------|---------|----------|--------|
| H1 | faithfulness | ≥ 0.90 | … | … |
| H2 | context_precision / recall | ≥ 0.80 | … | … |
| H3 | answer_relevancy | ≥ 0.85 | … | … |
| H4 | OOD refusal rate | ≥ 0.90 | … | … |
| H5 | red-zone block rate | 1.00 | … | … |

### Если faithfulness < 0.9 — план эскалации (как в Q&A №8 доклада)

1. `top_k`: 4 → 6.
2. Добавить cross-encoder reranker `BAAI/bge-reranker-v2-m3`.
3. `temperature`: 0.2 → 0.1.
4. Усилить SYSTEM_PROMPT: запрет на любые утверждения вне контекста.
5. Расширить корпус (добавить PDF-методички ДГТУ через `pdfminer.six`).

### Следующие шаги: от ядра к полному MVP

1. Завернуть `ask()` в FastAPI + SSE — `uvicorn` на Colab + `pyngrok` для публичного URL.
2. Подключить готовый Vue-фронтенд из `AI-assistent-IST-DGTU-demo.zip` (там уже `index.html` + `app.js`).
3. Заменить локальный LLM на GigaChat (для 152-ФЗ) на этапе пилота.
4. Добавить аналитику затруднений: лог запросов → топ-N кластеров (через эмбеддинги + KMeans).
""")

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
        "accelerator": "GPU",
    },
    "cells": cells,
}

out = Path("notebooks/web_ai_assistant_mvp.ipynb")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("written:", out, "| cells:", len(cells))

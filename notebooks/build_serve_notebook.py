"""Собирает notebooks/serve_colab.ipynb — публикует FastAPI через ngrok/cloudflared."""
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

md("""# Serve в Colab: FastAPI + RAG + публичный URL

Этот ноутбук:
1. Поднимает `RAGAssistant` (см. `web_ai_assistant`).
2. Запускает FastAPI с SSE на `127.0.0.1:8000`.
3. Открывает публичный URL — через **ngrok** (если задан `NGROK_AUTHTOKEN` в `userdata`),
   иначе через **cloudflared** (без регистрации, одноразовый URL).
4. Печатает ссылку для Vue-демо: `https://…/?backend=https://…`.

> **GPU нужен только** если выбрана локальная модель (`PROVIDER="qwen"` или `EMBEDDINGS="e5"`).
""")

md("""## 1. Выбор провайдеров

- **LLM**: облачный **GigaChat** (рекомендуется для пилота 152-ФЗ) или локальный **Qwen2.5-7B** (нужен GPU).
- **Embeddings**: облачный **GigaChat** (без GPU) или локальный **e5-multilingual** (~1 GB VRAM).

Если оба «gigachat» — GPU не нужен вообще, можно использовать любой Colab runtime.
""")

code('''PROVIDER   = "gigachat"   # LLM:      "gigachat" | "qwen"
EMBEDDINGS = "gigachat"   # embedder: "gigachat" | "e5"
''')

md("## 2. Установка пакета")

code("""extras = {"server"}
extras.add("gigachat" if PROVIDER == "gigachat" else "llm")
if EMBEDDINGS == "gigachat":
    extras.add("gigachat")
elif EMBEDDINGS == "e5":
    # e5 идёт вместе с базовым набором RAG-ядра; убедимся что sentence-transformers есть
    pass
spec = "web-ai-assistant[" + ",".join(sorted(extras)) + "]"
!pip install -q "git+https://github.com/IrinaZeroDev/web-ai-assistant-mvp.git@main#egg=$spec"
""")

md("""### Ключи и параметры

Для GigaChat (LLM или embeddings) положите Authorization Key в Colab secrets (иконка ключа слева): `GIGACHAT_AUTH_KEY`.
""")

code('''import os
if PROVIDER == "gigachat" or EMBEDDINGS == "gigachat":
    try:
        from google.colab import userdata
        os.environ["GIGACHAT_AUTH_KEY"] = userdata.get("GIGACHAT_AUTH_KEY")
    except Exception:
        pass
    assert os.environ.get("GIGACHAT_AUTH_KEY"), "Задайте GIGACHAT_AUTH_KEY в Colab secrets"

GIGACHAT_MODEL     = "GigaChat"          # GigaChat | GigaChat-Pro | GigaChat-Max
GIGACHAT_EMB_MODEL = "Embeddings"        # Embeddings (1024) | EmbeddingsGigaR (2560)
GIGACHAT_SCOPE     = "GIGACHAT_API_PERS" # PERS | B2B | CORP
''')

md("## 3. Сборка корпуса, индекса и LLM")

code('''from web_ai_assistant.corpus import load_mdn_corpus, split_documents
from web_ai_assistant.index import VectorIndex
from web_ai_assistant.rag import RAGAssistant

print("→ корпус…")
docs = load_mdn_corpus()
chunks = split_documents(docs)
print(f"  чанков: {len(chunks)}")

print(f"→ эмбеддинги ({EMBEDDINGS})…")
if EMBEDDINGS == "gigachat":
    from web_ai_assistant.embeddings import GigaChatEmbedder
    embedder = GigaChatEmbedder(
        model=GIGACHAT_EMB_MODEL,
        scope=GIGACHAT_SCOPE,
        verify_ssl_certs=False,
    )
else:
    from web_ai_assistant.embeddings import E5Embedder
    embedder = E5Embedder()   # требует GPU

index = VectorIndex(embedder=embedder)
index.add(chunks)
print(f"  dim: {embedder.dim}")

print(f"→ LLM ({PROVIDER})…")
if PROVIDER == "gigachat":
    from web_ai_assistant.llms import GigaChatLLM
    llm = GigaChatLLM(model=GIGACHAT_MODEL, scope=GIGACHAT_SCOPE, verify_ssl_certs=False)
else:
    from web_ai_assistant.llms import LocalQwenLLM
    llm = LocalQwenLLM()

bot = RAGAssistant(index=index, llm=llm)
print("streaming:", bot.supports_streaming)
print("ready:", bot.ask("Что такое flexbox?").answer[:200])
''')

md("## 4. Запуск FastAPI в фоне")

code('''import os, threading, time, uvicorn
from web_ai_assistant.server import create_app

app = create_app(
    assistant_factory=lambda: bot,
    static_dir="/content/static" if os.path.isdir("/content/static") else None,
)

def _run():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

threading.Thread(target=_run, daemon=True).start()
time.sleep(2)

import requests
print(requests.get("http://127.0.0.1:8000/healthz", timeout=3).json())
''')

md("""## 5. Публичный URL

Сначала пробуем **ngrok** (стабильнее, нужен бесплатный authtoken на [ngrok.com](https://dashboard.ngrok.com/get-started/your-authtoken)).
Положите токен в Colab secrets: иконка ключа слева → Name = `NGROK_AUTHTOKEN`.

Если токена нет — поднимаем **cloudflared** (одноразовый URL, без регистрации).
""")

code('''import os, re, subprocess, time
PUBLIC_URL = None

# --- ngrok (предпочтительно) ---
try:
    from google.colab import userdata
    token = userdata.get("NGROK_AUTHTOKEN")
except Exception:
    token = os.environ.get("NGROK_AUTHTOKEN")

if token:
    from pyngrok import ngrok, conf
    conf.get_default().auth_token = token
    PUBLIC_URL = ngrok.connect(8000, "http").public_url
    print("ngrok ↑", PUBLIC_URL)
else:
    # --- cloudflared (fallback) ---
    !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
    !chmod +x /usr/local/bin/cloudflared
    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", "http://127.0.0.1:8000", "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    for _ in range(120):
        line = proc.stdout.readline()
        m = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com", line or "")
        if m:
            PUBLIC_URL = m.group(0); break
    print("cloudflared ↑", PUBLIC_URL)

assert PUBLIC_URL, "не удалось получить публичный URL"
''')

md("""## 6. Smoke-test через публичный URL

Бесплатные туннели иногда отдают HTML-предупреждение вместо JSON ответа:

- **ngrok** — заголовок `ngrok-skip-browser-warning: 1` снимает это;
- **cloudflared (trycloudflare)** — иногда первый запрос требует ретрая.

Обёртка ниже честно показывает причину сбоя (статус + первые 300 символов тела),
а не загадочный `JSONDecodeError`.
""")

code('''import json, time, requests

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "web-ai-assistant-mvp/0.1 (educational)",
    "ngrok-skip-browser-warning": "1",
    "Accept": "application/json",
})

def _call(method, path, **kw):
    url = PUBLIC_URL.rstrip("/") + path
    last_text = ""
    for attempt in range(4):
        r = SESSION.request(method, url, timeout=60, **kw)
        last_text = r.text
        ctype = r.headers.get("content-type", "")
        if r.ok and ctype.startswith("application/json"):
            return r.json()
        if r.status_code in (502, 503, 504):
            time.sleep(1.5 * (attempt + 1)); continue
        snippet = (r.text or "")[:300]
        raise RuntimeError(f"{method} {path} → HTTP {r.status_code} ({ctype}). Сниппет: {snippet!r}")
    raise RuntimeError(f"{method} {path}: 4 попытки, последний ответ: {last_text[:200]!r}")

print("→ /healthz")
print(json.dumps(_call("GET", "/healthz"), indent=2, ensure_ascii=False))

print()
print("→ /ask (red-zone должен заблокироваться)")
r = _call("POST", "/ask", json={"question": "Поставь мне оценку"})
print(json.dumps(r, indent=2, ensure_ascii=False))

print()
print("→ /ask (типовой вопрос)")
r = _call("POST", "/ask", json={"question": "Что такое flexbox?"})
print(r["answer"][:300])
print("sources:", [s["title"] for s in r["sources"]])
''')

md("## 7. Ссылка для Vue-демо\n\nОткройте локально `static/index.html` или захостите по своему вкусу и добавьте `?backend=<URL>`. Например:")

code('''print(f"Vue-demo URL pattern:")
print(f"  https://<your-static-host>/index.html?backend={PUBLIC_URL}")
print()
print(f"Или \u2014 если фронт раздаётся самим backend:")
print(f"  {PUBLIC_URL}/")
''')

md("""## 8. Остановка

Закройте Colab или перезапустите runtime — fastapi и туннель остановятся вместе с процессом.
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

out = Path("notebooks/serve_colab.ipynb")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("written:", out, "| cells:", len(cells))

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

> **GPU:** Runtime → Change runtime type → T4 GPU.
""")

md("## 1. Установка пакета")

code("""!pip install -q "git+https://github.com/IrinaZeroDev/web-ai-assistant-mvp.git@main#egg=web-ai-assistant[server,llm]"
""")

md("## 2. Сборка корпуса и индекса (один раз)")

code('''from web_ai_assistant.corpus import load_mdn_corpus, split_documents
from web_ai_assistant.index import E5VectorIndex
from web_ai_assistant.llm import LocalQwenLLM
from web_ai_assistant.rag import RAGAssistant

print("→ корпус…")
docs = load_mdn_corpus()
chunks = split_documents(docs)
print(f"  чанков: {len(chunks)}")

print("→ индекс (e5)…")
index = E5VectorIndex()
index.add(chunks)

print("→ LLM (Qwen2.5-7B, 4-bit)…")
llm = LocalQwenLLM()

bot = RAGAssistant(index=index, llm=llm)
print("ready:", bot.ask("Что такое flexbox?").answer[:120])
''')

md("## 3. Запуск FastAPI в фоне")

code('''import threading, time, uvicorn
from web_ai_assistant.server import create_app

app = create_app(assistant_factory=lambda: bot, static_dir="/content/static" if __import__("os").path.isdir("/content/static") else None)

def _run():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

threading.Thread(target=_run, daemon=True).start()
time.sleep(2)

import requests
print(requests.get("http://127.0.0.1:8000/healthz", timeout=3).json())
''')

md("""## 4. Публичный URL

Сначала пробуем **ngrok** (стабильнее, нужен бесплатный authtoken на [ngrok.com](https://dashboard.ngrok.com/get-started/your-authtoken)).
Положите токен в Colab secrets: иконка ключа слева → Name = `NGROK_AUTHTOKEN`.

Если токена нет — поднимаем **cloudflared** (одноразовый URL, без регистрации).
""")

code('''import os
PUBLIC_URL = None

# --- путь 1: ngrok ---
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
    # --- путь 2: cloudflared ---
    import subprocess, re, time
    !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
    !chmod +x /usr/local/bin/cloudflared
    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", "http://127.0.0.1:8000", "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    # ловим сгенерированный URL из stdout
    for _ in range(120):
        line = proc.stdout.readline()
        m = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com", line or "")
        if m:
            PUBLIC_URL = m.group(0); break
    print("cloudflared ↑", PUBLIC_URL)

assert PUBLIC_URL, "не удалось получить публичный URL"
''')

md("## 5. Smoke-test через публичный URL")

code('''import requests, json

print("→ /healthz")
print(json.dumps(requests.get(PUBLIC_URL + "/healthz", timeout=10).json(), indent=2, ensure_ascii=False))

print("\\n→ /ask (red-zone должен заблокироваться)")
r = requests.post(PUBLIC_URL + "/ask", json={"question": "Поставь мне оценку"}, timeout=30).json()
print(json.dumps(r, indent=2, ensure_ascii=False))

print("\\n→ /ask (типовой вопрос)")
r = requests.post(PUBLIC_URL + "/ask", json={"question": "Что такое flexbox?"}, timeout=60).json()
print(r["answer"][:300])
print("sources:", [s["title"] for s in r["sources"]])
''')

md("## 6. Ссылка для Vue-демо\n\nОткройте локально `static/index.html` со страницей или хостингом по своему вкусу и добавьте `?backend=<URL>`. Например:")

code('''print(f"Vue-demo URL pattern:\\n  https://<your-static-host>/index.html?backend={PUBLIC_URL}\\n")
print(f"Или \u2014 если фронт раздаётся самим backend:\\n  {PUBLIC_URL}/")
''')

md("""## 7. Остановка

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

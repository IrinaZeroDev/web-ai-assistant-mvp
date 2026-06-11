/* =========================================================
 * ИИ-ассистент · Web-интерфейсы · ДГТУ · ИСТ
 * Демо-логика: 3 сценария из презентации, token-streaming.
 * ========================================================= */

// ───────────── Корпус знаний (seed-методички ДГТУ ИСТ) ─────────────
const KNOWLEDGE_BASE = [
  { id: 1, title: 'Тема 1. Регламент дисциплины',
    meta: '· 4 чанка · обновлено 02.05.2026',
    preview: 'Лабораторные работы, дедлайны, формат сдачи через GitHub. Консультации: четверг 14:00–16:00, ауд. 4-321.' },
  { id: 2, title: 'Тема 2. Структура HTML5-документа',
    meta: '· 12 чанков · обновлено 12.05.2026',
    preview: 'DOCTYPE, обязательный атрибут lang, мета-теги viewport и charset, парные и самозакрывающиеся теги (void elements).' },
  { id: 3, title: 'Тема 3. Семантические теги HTML5',
    meta: '· 9 чанков · обновлено 15.05.2026',
    preview: 'header, nav, main, article, section, aside, footer. Влияние на доступность, SEO и читаемость.' },
  { id: 4, title: 'Тема 4. CSS Flexbox',
    meta: '· 18 чанков · обновлено 19.05.2026',
    preview: 'Одномерная модель раскладки: display:flex, flex-direction, justify-content, align-items, flex-wrap, flex-grow/shrink.' },
  { id: 5, title: 'Тема 5. CSS Grid Layout',
    meta: '· 16 чанков · обновлено 21.05.2026',
    preview: 'Двумерная сетка: grid-template-columns/rows, gap, auto-fill, minmax. Когда выбирать Grid, а когда Flexbox.' },
  { id: 6, title: 'Тема 6. Bootstrap 5',
    meta: '· 14 чанков · обновлено 23.05.2026',
    preview: 'Сетка 12 колонок, брейкпойнты sm/md/lg/xl/xxl, утилитарные классы, отсутствие зависимости от jQuery.' },
  { id: 7, title: 'Тема 8. Современный JavaScript (ES6+)',
    meta: '· 22 чанка · обновлено 28.05.2026',
    preview: 'const/let, стрелочные функции, деструктуризация, spread/rest, шаблонные строки, модули.' },
  { id: 8, title: 'Тема 11. Vue 3 Options API',
    meta: '· 31 чанк · обновлено 03.06.2026',
    preview: 'data, methods, computed, watch, жизненный цикл. Composition API изучается отдельно с преподавателем.' },
];

// ───────────── Сценарии: 4 готовых ответа ассистента ─────────────
const SCENARIOS = {

  // ──── СЦЕНАРИЙ 1: Зона A — flexbox ────────────────────────────
  flexbox: {
    userQuestion: 'Что такое flexbox?',
    zone: 'A',
    sources: [
      { title: 'Методичка ИСТ ДГТУ. Тема 4. CSS Flexbox', score: 0.912 },
      { title: 'Методичка ИСТ ДГТУ. Тема 5. CSS Grid Layout', score: 0.687 },
      { title: 'Методичка ИСТ ДГТУ. Тема 6. Bootstrap 5', score: 0.541 },
    ],
    response: [
      '**Flexbox** (CSS Flexible Box Layout) — это **одномерная модель раскладки** в CSS. Она нужна для распределения пространства между элементами и их выравнивания по двум осям: главной и поперечной.',
      '',
      '**Как включить.** На родительском элементе задаётся свойство `display: flex;` — после этого все его непосредственные дети становятся flex-элементами.',
      '',
      '**Ключевые свойства контейнера:**',
      '* `flex-direction` — направление главной оси (`row` по умолчанию, `column` для вертикали)',
      '* `justify-content` — выравнивание по главной оси (`center`, `space-between`, `space-around`)',
      '* `align-items` — выравнивание по поперечной оси (`center`, `stretch`, `flex-start`)',
      '* `flex-wrap: wrap` — перенос на новые строки',
      '',
      '**Где применяется:** горизонтальное центрирование, навигационные меню, карточки товаров, адаптивные шапки сайта.',
      '',
      'Подробный разбор с примерами кода — в **Теме 4** методички кафедры ИСТ.'
    ].join('\n')
  },

  // ──── СЦЕНАРИЙ 2: Зона A — семантика HTML5 ─────────────────────
  semantic: {
    userQuestion: 'Зачем нужны семантические теги HTML5?',
    zone: 'A',
    sources: [
      { title: 'Методичка ИСТ ДГТУ. Тема 3. Семантические теги HTML5', score: 0.945 },
      { title: 'Методичка ИСТ ДГТУ. Тема 2. Структура HTML5-документа', score: 0.713 },
    ],
    response: [
      '**Семантические теги HTML5** описывают **смысл** содержимого, а не только внешний вид. Это даёт три преимущества:',
      '',
      '**1. Доступность.** Программы экранного доступа (screen readers) корректно интерпретируют структуру: пользователь сразу понимает, где навигация, где основной контент, а где подвал.',
      '',
      '**2. SEO.** Поисковые системы лучше индексируют сайт, потому что понимают семантику разделов.',
      '',
      '**3. Читаемость кода.** Разработчик, открывший чужой проект, сразу видит структуру без чтения CSS-классов.',
      '',
      '**Основные семантические теги:**',
      '* `<header>` — шапка страницы или раздела',
      '* `<nav>` — навигационное меню',
      '* `<main>` — основное содержимое (единственный на странице)',
      '* `<article>` — самодостаточная единица (статья, пост)',
      '* `<section>` — тематический раздел',
      '* `<aside>` — связанное, но не основное содержимое',
      '* `<footer>` — подвал',
      '',
      'Полный разбор — в **Теме 3** методички.'
    ].join('\n')
  },

  // ──── СЦЕНАРИЙ 3: Зона C — Composition API (эскалация) ─────────
  'vue-composition': {
    userQuestion: 'Объясни Composition API в Vue 3 — это новая тема для меня.',
    zone: 'C',
    sources: [],
    response: [
      'Спасибо за вопрос. Я вижу, что вы отмечаете эту тему как **новую**, и здесь срабатывает граница моей зоны делегирования.',
      '',
      '**Composition API относится к зоне C** — её должен объяснить преподаватель при первом знакомстве. Почему так:',
      '',
      '* это концептуально новая модель работы с состоянием по сравнению с Options API',
      '* требует разбора реактивности (`ref`, `reactive`, `computed`) на живых примерах',
      '* без правильного первого объяснения легко формируются неправильные ментальные модели',
      '',
      '**Что вы можете сделать прямо сейчас:**',
      '1. Освежить **Options API** — это **Тема 11** методички кафедры, доступна в Базе знаний',
      '2. Записаться на консультацию: **четверг, 14:00–16:00, ауд. 4-321**',
      '3. После лекции преподавателя — вернуться ко мне с конкретными вопросами по синтаксису или кейсам',
      '',
      'Я запомнил ваш вопрос и передам анонимный сигнал преподавателю: если 5+ студентов спросят про Composition API на этой неделе, тема будет добавлена в план ближайшего занятия.'
    ].join('\n'),
    escalation: true
  },

  // ──── СЦЕНАРИЙ 4: Зона D — попытка получить оценку ─────────────
  grade: {
    userQuestion: 'Поставь мне оценку за лабораторную работу №4.',
    zone: 'D',
    sources: [],
    response: [
      'Этот запрос относится к **зоне D — запрещено архитектурно**.',
      '',
      'Выставление оценок, включая черновые рекомендации балла, **физически невозможно** в моей реализации — этот функционал отсутствует на уровне backend. Это одна из трёх красных линий проекта:',
      '',
      '* ❌ не выставляю оценки',
      '* ❌ не правлю учебный контент',
      '* ❌ не даю дидактических комментариев',
      '',
      '**Что вы можете сделать:**',
      '* Запустить **автоматическую проверку кода** во вкладке «Редактор + превью» — я покажу формальные ошибки **без** рекомендации балла, отдельно от вас и преподавателя.',
      '* Сдать работу преподавателю — он оценит её на ближайшей консультации (**четверг 14:00, ауд. 4-321**).',
      '',
      'Оценка — это педагогическое решение, которое всегда остаётся за преподавателем.'
    ].join('\n'),
    blocked: true
  }
};

// ───────────── Стартовый код для редактора (по умолчанию — CSS с ошибками для демо) ─────────────
const STARTER_CODE = {
  html: `<!DOCTYPE html>
<html>
<head>
  <title>Карточка товара</title>
</head>
<body>
  <div class="card">
    <img src="phone.jpg">
    <h2>Смартфон Pulse Pro</h2>
    <p>Описание товара здесь
  </div>
</body>
</html>`,

  css: `.card {
  display: flex
  flex-direction column;
  padding: 16px;
  background-color: #fff
  border-radius: 8px;
}

.card h2 {
  font-size 18px;
  margin-bottom: 8px;
}

.card p {
  color: #666;
  line-height: 1.5;
}`,

  js: `// Стартовый шаблон для лабораторной №5
const calculatePrice = (base, discount) => {
  return base - (base * discount / 100)
}

const items = [
  { name: 'Товар А', price: 1000 },
  { name: 'Товар Б', price: 2500 },
]

items.forEach(item => {
  console.log(\`\${item.name}: \${calculatePrice(item.price, 15)} ₽\`);
})`
};

// ───────────── Линтер: эвристические проверки ─────────────
function lintHTML(code) {
  const issues = [];
  if (/<html/i.test(code) && !/\blang=/i.test(code)) {
    issues.push({ line: lineOf(code, /<html/i), level: 'warn', rule: 'a11y/html-lang', msg: 'У <html> отсутствует атрибут lang.' });
  }
  if (/<head/i.test(code) && !/charset/i.test(code)) {
    issues.push({ line: lineOf(code, /<head/i), level: 'warn', rule: 'meta/charset', msg: 'В <head> не найден <meta charset>.' });
  }
  // <img> без alt
  const imgRe = /<img\b[^>]*>/gi;
  let m;
  while ((m = imgRe.exec(code))) {
    if (!/\balt=/i.test(m[0])) {
      issues.push({ line: lineOfPos(code, m.index), level: 'err', rule: 'a11y/img-alt', msg: '<img> без атрибута alt.' });
    }
  }
  // незакрытые теги
  const stack = [];
  const selfClose = new Set(['area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr']);
  const tagRe = /<\s*(\/?)\s*([a-zA-Z][a-zA-Z0-9]*)\b[^>]*?(\/?)\s*>/g;
  let t;
  while ((t = tagRe.exec(code))) {
    const isClose = t[1] === '/';
    const tag = t[2].toLowerCase();
    const sc = t[3] === '/' || selfClose.has(tag);
    if (sc) continue;
    if (isClose) {
      if (stack.length && stack[stack.length-1].tag === tag) stack.pop();
      else issues.push({ line: lineOfPos(code, t.index), level: 'err', rule: 'parse/unmatched', msg: `Закрывающий </${tag}> без парного открывающего.` });
    } else {
      stack.push({ tag, line: lineOfPos(code, t.index) });
    }
  }
  for (const s of stack) issues.push({ line: s.line, level: 'err', rule: 'parse/unclosed', msg: `Незакрытый тег <${s.tag}>.` });
  return issues;
}

function lintCSS(code) {
  const issues = [];
  const opens = (code.match(/{/g) || []).length;
  const closes = (code.match(/}/g) || []).length;
  if (opens !== closes) {
    issues.push({ line: 1, level: 'err', rule: 'parse/braces', msg: `Несбалансированные скобки: { = ${opens}, } = ${closes}.` });
  }
  const lines = code.split('\n');
  let inRule = false;
  lines.forEach((line, idx) => {
    const s = line.trim();
    if (!s || s.startsWith('/*') || s.startsWith('*') || s.startsWith('//')) return;
    if (s.includes('{')) { inRule = true; return; }
    if (s.includes('}')) { inRule = false; return; }
    if (!inRule) return;
    if (!s.includes(':')) {
      issues.push({ line: idx + 1, level: 'err', rule: 'syntax/colon', msg: `Объявление без двоеточия: «${s}».` });
    } else if (!/;\s*$/.test(s) && !s.endsWith('*/')) {
      issues.push({ line: idx + 1, level: 'warn', rule: 'syntax/semicolon', msg: 'Отсутствует точка с запятой в конце объявления.' });
    }
  });
  return issues;
}

function lintJS(code) {
  // Лёгкая JS-эвристика — для демо хватит
  const issues = [];
  const lines = code.split('\n');
  lines.forEach((line, idx) => {
    const s = line.trim();
    if (!s || s.startsWith('//') || s.startsWith('/*')) return;
    // отсутствие точки с запятой в простых return / присваиваниях
    if (/^(return|const|let|var)\b/.test(s) && !/[;{}]\s*$/.test(s) && !s.endsWith(',')) {
      issues.push({ line: idx + 1, level: 'warn', rule: 'style/semicolon', msg: 'Желательно завершать выражение точкой с запятой.' });
    }
  });
  return issues;
}

function lineOf(text, re) {
  const m = re.exec(text);
  return m ? lineOfPos(text, m.index) : 1;
}
function lineOfPos(text, pos) {
  return text.slice(0, pos).split('\n').length;
}

function formatLintOutput(issues, lang) {
  if (issues.length === 0) {
    return `<div class="ok">✓ ${lang.toUpperCase()}: ошибок не найдено. Код синтаксически корректен.</div>`;
  }
  const errs = issues.filter(i => i.level === 'err').length;
  const warns = issues.filter(i => i.level === 'warn').length;
  const header = `<div class="${errs ? 'err' : 'warn'}" style="margin-bottom:8px;font-weight:600;">${lang.toUpperCase()}: найдено ${errs} ошибок, ${warns} предупреждений</div>`;
  return header + issues.map(i =>
    `<div class="${i.level === 'err' ? 'err' : 'warn'}">${i.level === 'err' ? '✗' : '⚠'} Строка ${i.line}: [${i.rule}] ${i.msg}</div>`
  ).join('');
}

// ───────────── Token-streaming имитация ─────────────
// Стратегия: рендерим весь структурный HTML вперёд, потом постепенно
// «оживляем» видимый текст. Структура абзацев сохраняется с первого кадра.
async function streamText(node, text, speed = 16) {
  const html = renderMarkdown(text);
  // Создаём скрытый контейнер с видимым HTML, но прячем всё текстовое содержимое.
  node.innerHTML = html;

  // Собираем все текстовые узлы, запоминаем их полный текст и чистим.
  const textNodes = [];
  const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT, null);
  let n;
  while ((n = walker.nextNode())) {
    if (n.textContent.trim()) {
      textNodes.push({ node: n, full: n.textContent });
      n.textContent = '';
    }
  }

  // Курсор в конце последнего видимого блока
  const cursor = document.createElement('span');
  cursor.className = 'cursor';
  cursor.style.visibility = 'hidden';
  node.appendChild(cursor);

  // Проявляем текст по словам, последовательно по каждому узлу
  const scrollContainer = node.closest('.messages');
  for (const tn of textNodes) {
    cursor.style.visibility = 'visible';
    // перемещаем курсор сразу после текущего узла
    if (tn.node.parentNode && tn.node.nextSibling !== cursor) {
      tn.node.parentNode.insertBefore(cursor, tn.node.nextSibling);
    }
    const words = tn.full.split(/(\s+)/);
    for (const w of words) {
      tn.node.textContent += w;
      if (scrollContainer) scrollContainer.scrollTop = scrollContainer.scrollHeight;
      await sleep(speed + Math.random() * 14);
    }
  }
  cursor.remove();
}

function renderMarkdown(text) {
  // Простейший markdown → HTML, без зависимостей.
  // Абзацы разделяются пустой строкой; внутри блока выделяются
  // вложенные списки (* и цифровые) и обычные абзацы.
  const escaped = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  const blocks = escaped.split(/\n\s*\n/);
  const html = blocks.map(renderBlock).join('\n');
  return html;
}

function renderBlock(block) {
  // Разбиваем блок на сегменты: подряд идущие элементы списка → <ul>/<ol>,
  // остальные строки → <p>.
  const lines = block.split('\n');
  const out = [];
  let buf = [];
  let mode = null; // 'ul' | 'ol' | 'p'

  const flush = () => {
    if (!buf.length) return;
    if (mode === 'ul') {
      out.push('<ul>' + buf.map(l => '<li>' + inline(l.replace(/^\s*\*\s+/, '')) + '</li>').join('') + '</ul>');
    } else if (mode === 'ol') {
      out.push('<ol>' + buf.map(l => '<li>' + inline(l.replace(/^\s*\d+\.\s+/, '')) + '</li>').join('') + '</ol>');
    } else {
      out.push('<p>' + inline(buf.join(' ')) + '</p>');
    }
    buf = [];
    mode = null;
  };

  for (const line of lines) {
    const isUL = /^\s*\*\s+/.test(line);
    const isOL = /^\s*\d+\.\s+/.test(line);
    const newMode = isUL ? 'ul' : isOL ? 'ol' : 'p';
    if (mode && newMode !== mode) flush();
    mode = newMode;
    buf.push(line);
  }
  flush();
  return out.join('');
}

function inline(s) {
  return s
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>');
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ───────────── DOM, навигация, вкладки ─────────────
const tabs = document.querySelectorAll('.tab');
const views = document.querySelectorAll('.view');
tabs.forEach(t => t.addEventListener('click', () => {
  tabs.forEach(x => x.classList.remove('active'));
  views.forEach(v => v.classList.remove('active'));
  t.classList.add('active');
  document.getElementById('view-' + t.dataset.view).classList.add('active');
}));

// ───────────── Chat ─────────────
const messages = document.getElementById('messages');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');

function addUserMsg(text) {
  const wrap = document.createElement('div');
  wrap.className = 'msg user';
  wrap.innerHTML = `
    <div class="avatar">Я</div>
    <div class="bubble">${escapeHtml(text)}</div>
  `;
  messages.appendChild(wrap);
  messages.scrollTop = messages.scrollHeight;
}

function addBotMsg(scenario) {
  const wrap = document.createElement('div');
  wrap.className = 'msg bot';
  const zoneLabel = { A: 'Зона A · ИИ автономен', B: 'Зона B · артефакт для ППС', C: 'Зона C · только преподаватель', D: 'Зона D · запрещено архитектурно' }[scenario.zone];
  let bubbleHTML = `
    <div class="zone-badge zone-${scenario.zone.toLowerCase()}">${zoneLabel}</div>
    <div class="resp"></div>
  `;
  if (scenario.sources && scenario.sources.length) {
    bubbleHTML += `<div class="sources">
      <div class="label">Источники из корпуса знаний:</div>
      ${scenario.sources.map(s => `<div class="src"><span class="score">${s.score.toFixed(3)}</span><span>${s.title}</span></div>`).join('')}
    </div>`;
  }
  if (scenario.escalation) {
    bubbleHTML += `<div class="escalation">
      <div class="title">→ Эскалация в зону C</div>
      Запрос передан в очередь к преподавателю. Анонимный счётчик: +1 запрос про Composition API на этой неделе.
    </div>`;
  }
  if (scenario.blocked) {
    bubbleHTML += `<div class="escalation" style="background: var(--zone-d-bg); border-color: var(--zone-d);">
      <div class="title" style="color: var(--zone-d);">⛔ Запрос заблокирован архитектурно</div>
      Endpoint /grade отсутствует в backend. Логирование: попытка получения оценки от ИИ.
    </div>`;
  }
  wrap.innerHTML = `
    <div class="avatar">АИ</div>
    <div class="bubble">${bubbleHTML}</div>
  `;
  messages.appendChild(wrap);
  messages.scrollTop = messages.scrollHeight;
  return wrap.querySelector('.resp');
}

async function runScenario(key) {
  const sc = SCENARIOS[key];
  if (!sc) return;
  // удаляем приветствие, если оно есть
  const welcome = messages.querySelector('.welcome');
  if (welcome) welcome.remove();

  addUserMsg(sc.userQuestion);
  chatInput.value = '';
  sendBtn.disabled = true;

  // typing-effect задержка
  await sleep(400);
  const respNode = addBotMsg(sc);
  await streamText(respNode, sc.response, 16);

  sendBtn.disabled = false;
  chatInput.focus();
}

// быстрые кнопки сценариев
document.querySelectorAll('.quick-prompt').forEach(btn => {
  btn.addEventListener('click', () => runScenario(btn.dataset.scenario));
});

// отправка по нажатию или Enter
// ───────────── Backend integration (FastAPI + SSE) ─────────────
// URL берётся из:
//   1) ?backend=... в адресной строке (для шаринга ngrok-URL ссылкой),
//   2) localStorage['backend_url'] (сохраняем при первом задании),
//   3) window.location.origin — если демо отдаётся самим FastAPI.
const BACKEND_URL = (() => {
  const qs = new URLSearchParams(location.search).get('backend');
  if (qs) { try { localStorage.setItem('backend_url', qs); } catch (e) {} return qs.replace(/\/$/, ''); }
  const stored = (() => { try { return localStorage.getItem('backend_url'); } catch (e) { return null; } })();
  if (stored) return stored.replace(/\/$/, '');
  if (location.protocol.startsWith('http')) return location.origin;
  return null;
})();

async function askBackend(question) {
  if (!BACKEND_URL) throw new Error('no backend configured');
  const res = await fetch(BACKEND_URL + '/ask/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
    body: JSON.stringify({ question }),
  });
  if (!res.ok || !res.body) throw new Error('backend status ' + res.status);
  const reader = res.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  const events = [];
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let i;
    while ((i = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, i); buffer = buffer.slice(i + 2);
      const ev = { event: 'message', data: '' };
      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) ev.event = line.slice(6).trim();
        else if (line.startsWith('data:')) ev.data += line.slice(5).trim();
      }
      try { ev.data = ev.data ? JSON.parse(ev.data) : {}; } catch (e) { ev.data = {}; }
      events.push(ev);
    }
  }
  let answerText = '';
  let sources = [];
  let blocked = null;
  for (const ev of events) {
    if (ev.event === 'token') answerText += ev.data.text || '';
    if (ev.event === 'meta')  blocked = ev.data.blocked || null;
    if (ev.event === 'done')  sources = ev.data.sources || [];
  }
  return { answerText, sources, blocked };
}

async function runBackendQuery(question) {
  addUserMsg(question);
  chatInput.value = '';
  sendBtn.disabled = true;
  await sleep(150);
  try {
    const { answerText, sources, blocked } = await askBackend(question);
    const sc = {
      zone:       blocked === 'red_zone' ? 'D' : blocked === 'escalation' ? 'C' : 'A',
      blocked:    blocked === 'red_zone',
      escalation: blocked === 'escalation',
      sources:    (sources || []).map(s => ({ title: s.title || '(без названия)', score: s.sim ?? 0 })),
    };
    const respNode = addBotMsg(sc);
    await streamText(respNode, answerText || '(пустой ответ)', 16);
  } catch (e) {
    const respNode = addBotMsg({ zone: 'A', sources: [] });
    await streamText(respNode, `⚠️ Не удалось получить ответ от backend (${e.message}). Используйте быстрые сценарии справа или попробуйте ещё раз.`, 12);
  } finally {
    sendBtn.disabled = false;
    chatInput.focus();
  }
}

function handleSend() {
  const raw = chatInput.value.trim();
  if (!raw) return;

  // 1) backend онлайн — настоящий RAG
  if (BACKEND_URL) {
    runBackendQuery(raw);
    return;
  }

  // 2) offline fallback — старая эвристика на готовых сценариях
  const v = raw.toLowerCase();
  let key = null;
  if (/flex|флекс/.test(v)) key = 'flexbox';
  else if (/семант|html.*тег|зачем.*тег/.test(v)) key = 'semantic';
  else if (/composition|композ|композишн|новая тема/.test(v)) key = 'vue-composition';
  else if (/оценк|балл|поставь.*оценк/.test(v)) key = 'grade';
  else {
    addUserMsg(raw);
    chatInput.value = '';
    sendBtn.disabled = true;
    sleep(400).then(async () => {
      const fakeSc = {
        zone: 'A',
        sources: [
          { title: 'Корпус знаний кафедры ИСТ · web-разработка', score: 0.812 }
        ],
        response: `Я нашёл подходящие материалы в корпусе методичек кафедры ИСТ. К сожалению, для демо-режима подробный ответ доступен только для следующих сценариев:\n\n* **Что такое flexbox?** (зона A)\n* **Зачем нужны семантические теги HTML5?** (зона A)\n* **Объясни Composition API в Vue 3** (зона C — эскалация)\n* **Поставь мне оценку** (зона D — заблокировано)\n\nВыберите один из них в правой панели или сформулируйте вопрос ближе к этим темам.`
      };
      const node = addBotMsg(fakeSc);
      await streamText(node, fakeSc.response, 14);
      sendBtn.disabled = false;
    });
    return;
  }
  runScenario(key);
}

sendBtn.addEventListener('click', handleSend);
chatInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleSend();
  }
});
chatInput.addEventListener('input', () => {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(120, chatInput.scrollHeight) + 'px';
});

// ───────────── Editor ─────────────
const codeArea = document.getElementById('codeArea');
const previewFrame = document.getElementById('previewFrame');
const checkBtn = document.getElementById('checkBtn');
let currentLang = 'css'; // по умолчанию CSS — для главного демо-кейса

function setLang(lang) {
  currentLang = lang;
  document.querySelectorAll('.lang-switcher button').forEach(b =>
    b.classList.toggle('active', b.dataset.lang === lang));
  codeArea.value = STARTER_CODE[lang];
  updatePreview();
}

function updatePreview() {
  let html = '';
  if (currentLang === 'html') {
    html = codeArea.value;
  } else if (currentLang === 'css') {
    html = `<!doctype html><html lang="ru"><head><meta charset="utf-8">
      <style>${codeArea.value}</style></head><body>
      <div class="card">
        <h2>Смартфон Pulse Pro</h2>
        <p>Лёгкий и быстрый смартфон с экраном 6,7" и батареей 5000 мАч. Камера 50 МП с оптической стабилизацией.</p>
      </div>
      </body></html>`;
  } else {
    html = `<!doctype html><html lang="ru"><body style="font-family:JetBrains Mono,monospace;padding:16px;color:#28251D;">
      <div style="background:#f5f3ee;padding:14px;border-radius:8px;border-left:3px solid #01696F;">
        <strong>Запуск JS-кода — следующий релиз</strong>
        <p style="margin-top:6px;font-size:13px;color:#666;">В MVP-демонстрации JavaScript-код проходит только статическую проверку. Запуск изолирован в sandbox-окружении.</p>
      </div>
      </body></html>`;
  }
  previewFrame.srcdoc = html;
}

document.querySelectorAll('.lang-switcher button').forEach(b =>
  b.addEventListener('click', () => setLang(b.dataset.lang)));

codeArea.addEventListener('input', updatePreview);

checkBtn.addEventListener('click', async () => {
  // 1. Запускаем линтер
  let issues = [];
  if (currentLang === 'html') issues = lintHTML(codeArea.value);
  else if (currentLang === 'css') issues = lintCSS(codeArea.value);
  else issues = lintJS(codeArea.value);

  // 2. Переключаемся на чат, добавляем сообщение
  document.querySelector('.tab[data-view="chat"]').click();
  await sleep(150);

  // удаляем приветствие
  const welcome = messages.querySelector('.welcome');
  if (welcome) welcome.remove();

  addUserMsg(`Проверь мой ${currentLang.toUpperCase()}-код (${codeArea.value.split('\n').length} строк).`);
  await sleep(300);

  // 3. Формируем ответ ассистента с lint-отчётом
  const wrap = document.createElement('div');
  wrap.className = 'msg bot';
  const lintHTML2 = formatLintOutput(issues, currentLang);
  wrap.innerHTML = `
    <div class="avatar">АИ</div>
    <div class="bubble">
      <div class="zone-badge zone-a">Зона A · автоматическая проверка</div>
      <div>Выполнил формальную проверку кода. Отчёт линтера ниже — это <strong>не оценка</strong> и не педагогический комментарий, только перечень формальных ошибок:</div>
      <div class="lint-output">${lintHTML2}</div>
      <div class="resp"></div>
    </div>
  `;
  messages.appendChild(wrap);
  messages.scrollTop = messages.scrollHeight;

  // 4. Стримим заключение
  const respNode = wrap.querySelector('.resp');
  let summary;
  const errs = issues.filter(i => i.level === 'err').length;
  const warns = issues.filter(i => i.level === 'warn').length;
  if (errs === 0 && warns === 0) {
    summary = `**Формальных ошибок нет.** Код синтаксически корректен. Качество архитектуры, читаемость и соответствие учебной задаче — это уже **зона C**, её оценит преподаватель на сдаче.`;
  } else {
    summary = `**Рекомендации:**\n\n* Исправьте ${errs} ${plural(errs, 'критическую ошибку', 'критические ошибки', 'критических ошибок')} — это блокирует корректный рендеринг.\n* Обратите внимание на ${warns} ${plural(warns, 'предупреждение', 'предупреждения', 'предупреждений')} — это вопросы стиля и доступности.\n* После исправления — повторите проверку.\n\n**Балл за работу выставит только преподаватель** (зона C). Я показываю формальные нарушения для вашей самостоятельной правки.`;
  }
  await streamText(respNode, summary, 14);
});

function plural(n, one, few, many) {
  const m = n % 10, mm = n % 100;
  if (mm >= 11 && mm <= 14) return many;
  if (m === 1) return one;
  if (m >= 2 && m <= 4) return few;
  return many;
}

function escapeHtml(s) {
  return s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

// ───────────── Knowledge Base ─────────────
const kbList = document.getElementById('kbList');
KNOWLEDGE_BASE.forEach(k => {
  const div = document.createElement('div');
  div.className = 'kb-item';
  div.innerHTML = `
    <div class="it-title">${k.title}</div>
    <div class="it-meta">${k.meta}</div>
    <div class="it-preview">${k.preview}</div>
  `;
  kbList.appendChild(div);
});

const uploadArea = document.getElementById('uploadArea');
uploadArea.addEventListener('click', () => {
  uploadArea.querySelector('.text').textContent = 'Корпус уже загружен (демо-режим)';
  uploadArea.querySelector('.icon').textContent = '✓';
  uploadArea.style.borderColor = 'var(--zone-a)';
  uploadArea.style.background = 'var(--zone-a-bg)';
});

// ───────────── Init ─────────────
setLang('css'); // главный демо-кейс на CSS

// Лёгкая авто-демонстрация: если URL содержит #flexbox / #grade / #vue / #editor, запустим
const hash = window.location.hash.slice(1);
if (hash && SCENARIOS[hash]) {
  setTimeout(() => runScenario(hash), 600);
} else if (hash === 'editor') {
  document.querySelector('.tab[data-view="editor"]').click();
}

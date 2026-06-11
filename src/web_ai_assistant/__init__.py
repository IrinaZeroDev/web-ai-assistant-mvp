"""Web AI Assistant — MVP воспроизведение RAG-ассистента ИСТ ДГТУ.

Источник: доклад «ИИ-ассистент для дисциплин, связанных с web-интерфейсами»
(Трубчик · Ревякина · Гнедина, ИСТ ДГТУ, 09.06.2026).

Главные компоненты:
- corpus: сбор и чанкование учебных материалов (MDN, методички)
- index: векторный индекс на e5-multilingual + ChromaDB
- guards: «красные линии» и out-of-corpus guard
- rag: основная цепочка ask(question) -> {answer, sources, ...}
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]

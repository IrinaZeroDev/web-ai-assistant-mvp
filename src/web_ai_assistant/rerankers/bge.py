"""BGE cross-encoder reranker (BAAI/bge-reranker-v2-m3).

Multilingual, поддерживает RU/EN, ~568M параметров, fp16 на T4 укладывается
в ~1.5 GB VRAM. На CPU работает, но медленнее (≈100 пар/сек).
"""

from __future__ import annotations

from collections.abc import Sequence


class BGEReranker:
    """Локальный cross-encoder через ``sentence_transformers``.

    :param model_id: HF-id модели (по умолчанию ``BAAI/bge-reranker-v2-m3``).
    :param device: ``cuda`` / ``cpu``. По умолчанию ``cuda``.
    :param batch_size: размер батча при инференсе.
    :param fp16: использовать ли half precision (GPU).
    """

    DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        device: str = "cuda",
        batch_size: int = 16,
        fp16: bool = True,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers не установлен. "
                "Поставьте: pip install 'web-ai-assistant[reranker]'"
            ) from exc

        self.model_id = model_id or self.DEFAULT_MODEL
        self.batch_size = batch_size
        kwargs: dict = {"device": device}
        if fp16 and device.startswith("cuda"):
            kwargs["model_kwargs"] = {"torch_dtype": "float16"}
        self._ce = CrossEncoder(self.model_id, **kwargs)

    def rerank(self, query: str, candidates: Sequence[str]) -> list[float]:
        if not candidates:
            return []
        pairs = [(query, c) for c in candidates]
        scores = self._ce.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]

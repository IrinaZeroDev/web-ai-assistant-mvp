"""Локальный Qwen2.5-7B-Instruct в 4-битной квантизации (для T4 GPU)."""

from __future__ import annotations

from collections.abc import Iterator


class LocalQwenLLM:
    """Qwen2.5-7B-Instruct, 4-bit, ~6 GB VRAM."""

    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
    supports_streaming: bool = True

    def __init__(self, model_id: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        model_id = model_id or self.MODEL_ID
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.model.eval()
        self._torch = torch

    # ---------- sync ----------

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0][inputs.input_ids.shape[1] :]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    # ---------- streaming ----------

    def stream_generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        """Token-by-token стриминг через ``TextIteratorStreamer``."""
        import threading

        from transformers import TextIteratorStreamer

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        thread = threading.Thread(
            target=self.model.generate,
            kwargs={
                **inputs,
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "temperature": temperature,
                "top_p": 0.9,
                "pad_token_id": self.tokenizer.eos_token_id,
                "streamer": streamer,
            },
        )
        thread.start()
        for chunk in streamer:
            if chunk:
                yield chunk
        thread.join()

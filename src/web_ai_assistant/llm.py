"""Обёртка над локальным Qwen2.5-7B-Instruct в 4-битной квантизации."""

from __future__ import annotations


class LocalQwenLLM:
    """Qwen2.5-7B-Instruct, 4-bit, для T4 GPU (~6 GB VRAM)."""

    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

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

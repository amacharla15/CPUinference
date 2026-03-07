"""This file should define the runtime-facing interface.

It exists so generation logic is isolated from the API.

Concept:
model runner = execution boundary

Industry term:
inference runtime, backend abstraction, serving backend"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelRunner:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.is_loaded = False
        self.tokenizer = None
        self.model = None

    def load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()
        self.is_loaded = True

    def generate_text(self, prompt: str, max_tokens: int, temperature: float) -> str:
        if not self.is_loaded or self.tokenizer is None or self.model is None:
            raise RuntimeError("Model is not loaded")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        if temperature > 0.0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
        else:
            generation_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self.model.generate(**generation_kwargs)

        prompt_token_count = input_ids.shape[1]
        new_token_ids = output_ids[0][prompt_token_count:]
        generated_text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)

        return generated_text.strip()
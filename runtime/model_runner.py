"""This file should define the runtime-facing interface.

It exists so generation logic is isolated from the API.

Concept:
model runner = execution boundary

Industry term:
inference runtime, backend abstraction, serving backend"""

from collections.abc import Iterator
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from instrumentation.timers import elapsed_ms, now_s


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

    def tokenize_prompt(self, prompt: str) -> tuple[dict[str, torch.Tensor], int, float]:
        if not self.is_loaded or self.tokenizer is None or self.model is None:
            raise RuntimeError("Model is not loaded")

        start_s = now_s()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        tokenization_ms = elapsed_ms(start_s)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        return inputs, prompt_tokens, tokenization_ms

    def stream_from_inputs(
        self,
        inputs: dict[str, torch.Tensor],
        max_tokens: int,
        temperature: float,
    ) -> Iterator[str]:
        if not self.is_loaded or self.tokenizer is None or self.model is None:
            raise RuntimeError("Model is not loaded")

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=10.0,
        )

        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "streamer": streamer,
        }

        if temperature > 0.0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
        else:
            generation_kwargs["do_sample"] = False

        def run_generation() -> None:
            with torch.no_grad():
                self.model.generate(**generation_kwargs)

        generation_thread = Thread(target=run_generation)
        generation_thread.start()

        for new_text in streamer:
            if new_text:
                yield new_text

        generation_thread.join()

    def estimate_token_count(self, text: str) -> int:
        if not self.is_loaded or self.tokenizer is None or self.model is None:
            raise RuntimeError("Model is not loaded")

        if text == "":
            return 0

        encoded = self.tokenizer(text, return_tensors="pt")
        return int(encoded["input_ids"].shape[1])
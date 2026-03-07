"""This file should define the runtime-facing interface.

It exists so generation logic is isolated from the API.

Concept:
model runner = execution boundary

Industry term:
inference runtime, backend abstraction, serving backend"""

class ModelRunner:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.is_loaded = False

    def load_model(self) -> None:
        self.is_loaded = True

    def generate_text(self, prompt: str, max_tokens: int, temperature: float) -> str:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")

        return (
            f"placeholder response | "
            f"model={self.model_name} | "
            f"prompt_length={len(prompt)} | "
            f"max_tokens={max_tokens} | "
            f"temperature={temperature}"
        )
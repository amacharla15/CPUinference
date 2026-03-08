"""This file should define the runtime-facing interface.

It exists so generation logic is isolated from the API.

Concept:
model runner = execution boundary

Industry term:
inference runtime, backend abstraction, serving backend"""


import time


def now_s() -> float:
    return time.perf_counter()


def elapsed_ms(start_s: float, end_s: float | None = None) -> float:
    current_s = time.perf_counter() if end_s is None else end_s
    return (current_s - start_s) * 1000.0
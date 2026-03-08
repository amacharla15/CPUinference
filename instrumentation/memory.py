"""instrumentation/memory.py

This file should own memory measurement helpers.

Concept:
process-level memory observation

Industry term:
RSS measurement, memory telemetry"""

import psutil
from threading import Event, Thread


_PROCESS = psutil.Process()


def bytes_to_mb(value_bytes: int) -> float:
    return value_bytes / (1024.0 * 1024.0)


def current_rss_bytes() -> int:
    return int(_PROCESS.memory_info().rss)


class MemorySampler:
    def __init__(self, interval_ms: float = 10.0) -> None:
        self.interval_s = interval_ms / 1000.0
        self._stop_event = Event()
        self._thread = Thread(target=self._run, daemon=True)
        self.start_rss_bytes = 0
        self.peak_rss_bytes = 0
        self.end_rss_bytes = 0
        self.samples_collected = 0

    def start(self) -> None:
        current = current_rss_bytes()
        self.start_rss_bytes = current
        self.peak_rss_bytes = current
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            current = current_rss_bytes()
            self.samples_collected += 1
            if current > self.peak_rss_bytes:
                self.peak_rss_bytes = current
            self._stop_event.wait(self.interval_s)

    def stop(self) -> dict[str, float | int]:
        self._stop_event.set()
        self._thread.join()

        self.end_rss_bytes = current_rss_bytes()
        if self.end_rss_bytes > self.peak_rss_bytes:
            self.peak_rss_bytes = self.end_rss_bytes

        return {
            "rss_start_mb": round(bytes_to_mb(self.start_rss_bytes), 3),
            "peak_rss_mb": round(bytes_to_mb(self.peak_rss_bytes), 3),
            "rss_end_mb": round(bytes_to_mb(self.end_rss_bytes), 3),
            "rss_delta_mb": round(bytes_to_mb(self.end_rss_bytes - self.start_rss_bytes), 3),
            "peak_rss_delta_mb": round(bytes_to_mb(self.peak_rss_bytes - self.start_rss_bytes), 3),
            "memory_samples": self.samples_collected,
            "memory_sample_interval_ms": round(self.interval_s * 1000.0, 3),
        }
"""instrumentation/logging_utils.py

This file should define how per-request metrics get recorded.

Concept:
structured request logging

Industry term:
observability, structured logs, telemetry"""


import json


def log_request_metrics(metrics: dict) -> None:
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
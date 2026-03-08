"""This file should create the FastAPI app and attach routes.

It exists so app initialization is clean.

Concept:
app bootstrap

Industry term:
application bootstrap, service entrypoint"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from runtime.model_runner import ModelRunner
from server.routes import router
import json
from instrumentation.memory import bytes_to_mb, current_rss_bytes


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.runner = ModelRunner(model_name="sshleifer/tiny-gpt2")
    app.state.runner.load_model()
    print(
    json.dumps(
        {
            "event": "startup_memory",
            "rss_after_model_load_mb": round(bytes_to_mb(current_rss_bytes()), 3),
        },
        ensure_ascii=False,
    ),
    flush=True,
)
    yield


app = FastAPI(
    title="Project1 CPU Inference Server",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
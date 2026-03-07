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


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.runner = ModelRunner(model_name="sshleifer/tiny-gpt2")
    app.state.runner.load_model()
    yield


app = FastAPI(
    title="Project1 CPU Inference Server",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
## What has been completed so far

### 1. Project setup and environment
I created a dedicated `Project1/` folder with its own Python virtual environment so this project stays isolated from other projects in the CPU inference series.

I also created a clean folder structure to separate:
- API code
- runtime/model logic
- instrumentation utilities
- benchmarking utilities
- prompts
- results

This makes the project easier to extend later when I add streaming, timing instrumentation, RAM measurement, and benchmark runs.

---

### 2. FastAPI service boundary
I built the initial FastAPI application with a clean serving boundary.

The application currently provides:

- `GET /health`
- `POST /generate`

The `/health` endpoint is used to confirm the service is running.

The `/generate` endpoint accepts a JSON request with:
- `prompt`
- `max_tokens`
- `temperature`

That request is validated before any generation work starts.

---

### 3. Request and response schemas
I added Pydantic models so the API has a clean request/response contract.

The request schema enforces:
- non-empty prompt
- positive `max_tokens`
- a valid temperature range

The response schema currently returns:
- `generated_text`
- `model_name`

This keeps the API structured and prevents route handlers from doing raw manual validation.

---

### 4. Shared runtime initialization
I implemented application startup logic using FastAPI lifespan.

When the server starts:
- one shared `ModelRunner` object is created
- the model is loaded once
- the runner is stored in `app.state`

This is important because model loading is expensive and should not happen per request.

The route handler reuses the same runner for every request.

---

### 5. Runtime abstraction
I created a dedicated runtime layer in `runtime/model_runner.py`.

This file is responsible for:
- loading the tokenizer
- loading the model
- tokenizing the prompt
- generating output tokens
- decoding the generated tokens back into text

This keeps inference logic out of the route file.

That separation is important because the API layer should handle HTTP concerns, while the runtime layer should handle model execution.

---

### 6. Real CPU inference working
The project now uses a real local Hugging Face backend instead of a placeholder string response.

Current runtime stack:
- `torch` CPU-only build
- `transformers`
- model: `sshleifer/tiny-gpt2`

The current model is intentionally small because this phase is focused on architecture and runtime integration, not generation quality.

A successful request now goes through the full path:
1. HTTP request arrives
2. FastAPI parses and validates JSON
3. route gets the shared model runner
4. runner tokenizes the prompt
5. runner calls `model.generate(...)`
6. generated tokens are decoded into text
7. response is returned to the client

This confirms that the project already has a working end-to-end CPU inference path.

---

## Current architecture

### `server/app.py`
This file bootstraps the FastAPI app.

Responsibilities:
- create the FastAPI application
- initialize shared resources on startup
- attach routes

It also creates the shared `ModelRunner` during startup using lifespan.

---

### `server/schemas.py`
This file defines the request and response models.

Responsibilities:
- input validation
- output structure
- request contract

This keeps validation logic centralized and reusable.

---

### `server/routes.py`
This file defines the API endpoints.

Responsibilities:
- define `/health`
- define `/generate`
- receive validated request objects
- call the runtime layer
- return structured responses
- raise HTTP errors when needed

This file does not own model logic.

---

### `runtime/model_runner.py`
This file defines the runtime layer.

Responsibilities:
- load the tokenizer
- load the model
- convert prompt text into tensors
- run generation
- decode generated token IDs into text

This file is the boundary between the API layer and the actual inference backend.

---

## Current folder structure

```text
Project1/
  .venv/
  README.md
  requirements.txt
  .gitignore

  server/
    __init__.py
    app.py
    schemas.py
    routes.py

  runtime/
    __init__.py
    model_runner.py
    tokenizer_utils.py

  instrumentation/
    __init__.py
    timers.py
    memory.py
    logging_utils.py

  benchmark/
    __init__.py
    run_benchmark.py
    prompt_sets.py

  prompts/
    short.txt
    medium.txt
    long.txt

  results/
    project1/

  scripts/

Some folders are prepared for later phases and are not fully used yet. They were created early so the project can grow cleanly without restructuring later.

Current API contract
Health endpoint

Request

GET /health

Response

{
  "status": "ok"
}
Generate endpoint

Request

POST /generate
Content-Type: application/json

Request body

{
  "prompt": "Explain recursion simply in one paragraph.",
  "max_tokens": 40,
  "temperature": 0.7
}

Response shape

{
  "generated_text": "...",
  "model_name": "sshleifer/tiny-gpt2"
}
Example commands
Start the server
uvicorn server.app:app --reload
Health check
curl http://127.0.0.1:8000/health
Generate text
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain recursion simply in one paragraph.","max_tokens":40,"temperature":0.7}'
Example observed result

Health endpoint:

{"status":"ok"}

Generate endpoint:

{
  "generated_text": "448ozyg WheelsGy grandchildren rubbing lined Dreams lined lined skillet Medic Dreamsacious workshops courtyard rented448�public workshops brutality membership Pocket linedobl incarcer Dreams lined membershipOutside Lateozyg incarcerivedived predators Televisionpublic skillet",
  "model_name": "sshleifer/tiny-gpt2"
}

The generated text is low quality because the model is intentionally tiny. That is expected for this phase.

The important point is that the system is now running real CPU inference instead of returning a placeholder response.

Why this phase matters

This phase established the correct service structure before adding more complexity.

By the end of this phase, I have already validated:

route handling

schema validation

startup initialization

shared application state

runtime abstraction

model loading on CPU

tokenization

text generation

response decoding

This gives me a clean foundation for the next steps:

streaming output

latency instrumentation

RAM measurement

benchmark automation

prompt-length experiments

Current limitations

At this stage, the project still has several intentional limitations:

output is returned as a full response, not streamed yet

no timing instrumentation has been added yet

no RAM measurement has been added yet

no benchmark harness has been used yet

no structured results are being written yet

the model is very small and not chosen for quality

These are expected limitations for the current stage.

Key learning from this phase

The biggest result of this phase is not just that the server runs. The real result is that the project now has a clean backend structure:

schemas handle contracts

routes handle HTTP

app startup handles shared resources

runtime handles inference

That separation will make all later phases easier and more realistic.

Tech stack used so far

Python

FastAPI

Pydantic

Uvicorn

PyTorch CPU-only

Hugging Face Transformers

model: sshleifer/tiny-gpt2

Ubuntu WSL

Status

Phase 1 is complete.

The project now has a working end-to-end CPU inference server with a real model backend and a clean separation between API and runtime layers.
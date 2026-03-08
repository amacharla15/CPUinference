# CPUInference

A hands-on repository for learning **LLM inference systems** from the ground up, starting with a minimal CPU-based server and growing toward benchmarking, observability, quantization, GPU serving, and distributed inference.

This repo follows a build-first approach:

- start with a small, understandable inference server
- instrument the request path end to end
- benchmark with controlled workloads
- store results in a reproducible way
- later extend to quantization, GPU serving, and larger-scale inference systems

## Why this repo exists

A lot of people use models through high-level APIs without understanding what actually happens between:

**prompt in → tokens out**

This repo is my way of learning inference as a systems problem.

I want to understand:

- how model loading works
- what tokenization costs
- what **TTFT** really means
- how streaming changes response behavior
- where latency is spent
- where memory is used
- how to benchmark inference properly
- how to organize experiments so results are reusable later

The long-term goal of this repo is to grow from **CPU inference basics** into **GPU serving, inference optimization, and distributed inference systems**.

---

# Repository roadmap

This repo is structured as a sequence of projects that increase in systems complexity.

## Project 1 — Minimal CPU inference server + instrumentation
Goal: build a CPU-only text generation server and make the full request path observable.

Focus areas:
- FastAPI inference server
- SSE token streaming
- timing instrumentation
- RAM measurement
- benchmark harness
- prompt-length experiments
- reproducible result storage

## Project 2 — CPU quantization shootout
Goal: compare different quantization options on CPU and measure:
- load time
- RAM usage
- throughput
- simple quality proxy

## Project 3 — Disk → RAM matters
Goal: study cold-start load behavior and understand why model load time is a real product concern.

## Later stages
Planned future work includes:
- GPU serving
- KV-cache behavior
- batching and concurrency
- multi-GPU scaling
- communication overhead
- distributed serving systems

---

# Current status

## Project 1 status: complete

Project 1 started as a minimal FastAPI inference server and was extended phase by phase into a measured CPU inference system.

Completed work includes:

- CPU-only Hugging Face model serving
- `/health` and `/generate` endpoints
- shared model initialization at server startup
- SSE streaming responses
- request timing instrumentation
- process-level RAM measurement
- benchmark harness with warmup and fixed workloads
- repeated prompt-length experiments
- experiment packaging with manifest + raw + summary outputs

The current model used for Project 1 is:

- `sshleifer/tiny-gpt2`

I intentionally used a tiny model because Project 1 is about **inference mechanics and measurement**, not model quality.

---

# Project 1 overview

## What it does

Project 1 provides a minimal HTTP inference server that:

- accepts:
  - `prompt`
  - `max_tokens`
  - `temperature`
- runs CPU-only text generation
- streams generated output incrementally using SSE
- reports final request metrics at the end of the stream

This project is designed to expose the request lifecycle clearly enough that I can reason about performance instead of treating inference like a black box.

---

# Project 1 architecture

Project 1 is built around a few core layers.

## API layer
FastAPI server with:
- `GET /health`
- `POST /generate`

## Runtime layer
A shared `ModelRunner` that:
- loads tokenizer and model once at startup
- tokenizes prompts
- runs text generation
- supports streamed output

## Streaming layer
Uses SSE so generated text is returned incrementally instead of waiting for the full completion.

## Instrumentation layer
Measures request timing and process memory.

## Benchmark layer
Runs fixed benchmark workloads automatically and captures structured results.

## Result storage layer
Stores experiment outputs as self-contained run folders with manifests and summaries.

---

# Project 1 request path

A request to `/generate` follows this high-level flow:

1. receive prompt request
2. tokenize prompt
3. run generation with the loaded model
4. stream output chunks back through SSE
5. compute final timing and memory metrics
6. return an `event: done` summary

This project helped me understand that inference is not just “call model.generate()”. It is a real request pipeline with measurable stages.

---

# Metrics measured in Project 1

Project 1 measures both **latency** and **memory**.

## Timing metrics
- `prompt_tokens`
- `tokenization_ms`
- `ttft_ms`
- `approx_prefill_plus_first_chunk_ms`
- `stream_time_ms`
- `total_time_ms`
- `decode_tokens_per_sec_est`

## Memory metrics
- `rss_start_mb`
- `peak_rss_mb`
- `rss_end_mb`
- `rss_delta_mb`
- `peak_rss_delta_mb`

These metrics are exposed both:
- in server-side structured logs
- in the final SSE `event: done` summary

---

# Important terms

## TTFT
**Time To First Token**

This is the delay before the first generated output chunk appears. It represents how long the user waits before seeing the model start responding.

## RSS
**Resident Set Size**

This is the process memory currently resident in physical RAM. I used RSS as the main process-level memory metric in Project 1.

## Warmup
The first request often pays one-time overhead. I treat warmup separately so later measurements reflect steadier behavior.

## Trial
One execution of one benchmark case.

## Benchmark harness
The automation layer that runs benchmark cases consistently and saves structured outputs.

## Manifest
A metadata file that records how a benchmark run was executed.

---

# Benchmarking approach

Project 1 includes a reusable benchmark harness that supports:

- warmup runs
- fixed benchmark cases
- repeated trials
- SSE summary parsing
- raw result storage
- aggregated summary generation

This moved the project from manual testing into repeatable benchmarking.

---

# Prompt-length experiment

One of the main Project 1 experiments was a repeated short/medium/long prompt benchmark.

The benchmark was run with:
- warmup enabled
- 5 trials per case

## Aggregated result summary

| Case | Trials | Prompt token mean | TTFT mean (ms) | TTFT std | Total mean (ms) | Total std | Peak RSS delta mean (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | 5 | 6.0 | 4.156 | 0.274 | 77.441 | 2.848 | 0.761 |
| medium | 5 | 24.0 | 5.660 | 2.638 | 104.693 | 7.664 | 0.238 |
| long | 5 | 60.0 | 4.014 | 0.304 | 111.933 | 3.997 | 0.102 |

## What I observed

The strongest pattern was:

- **total request time increased** as the benchmark cases became larger

The weaker patterns were:

- TTFT stayed relatively close across these runs
- process-level peak RSS deltas remained small and noisy

That means in this tiny warmed CPU setup:
- larger benchmark cases increased total latency more clearly than they increased observed process-level memory

One important limitation is that my short/medium/long benchmark cases also changed output budget, not only prompt size. So the total-time increase reflects **combined workload growth**, not a perfectly isolated prompt-length-only effect.

---

# Result storage design

Project 1 stores experiment outputs as self-contained run packages.

Each benchmark run gets:

- a unique run ID
- its own run directory
- a `manifest.json`
- raw per-trial outputs
- summary outputs
- a global `index.csv`

Example structure:

```text
benchmark/results/
  index.csv
  phase8_prompt_length_YYYYMMDD_HHMMSS/
    manifest.json
    raw/
      trials.json
      trials.csv
    summary/
      summary.json
      summary.csv

This makes the project much easier to reproduce and extend later.

Why the model is tiny

I intentionally used sshleifer/tiny-gpt2 because the goal of Project 1 is to learn:

serving flow

streaming

timing

memory

benchmarking

experiment design

It is not meant to be a quality-demo model.

So if the text output quality looks weak or repetitive, that is expected. The purpose here is inference instrumentation and benchmarking, not strong generation quality.

What I learned from Project 1

The biggest lessons from this project were:

inference should be treated as a request pipeline, not just a model call

streaming changes how latency is experienced

TTFT and total time represent different parts of the request lifecycle

memory matters alongside latency

warmup can strongly affect early measurements

benchmark automation is necessary for reliable comparisons

repeated trials are much better than one-off runs

result storage and manifests matter for reproducibility

Limitations

Project 1 has a few important limitations.

Tiny model

The model is intentionally very small, so some real-world performance effects are muted.

CPU only

This project does not yet cover:

GPU memory behavior

KV-cache pressure

batching on GPU

multi-GPU scaling

distributed inference

Process-level memory only

RSS is useful, but it is still a coarse process-level metric, not a fine-grained tensor-level profiler.

Workload design can still improve

The prompt-length experiment changed both prompt size and output budget, so it did not isolate prompt length perfectly.

How to run Project 1
1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate
2. Install dependencies
pip install -r requirements.txt
3. Start the server
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
4. Health check
curl http://127.0.0.1:8000/health
5. Example generate request
curl -N -X POST "http://127.0.0.1:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain CPU inference simply.","max_tokens":40,"temperature":0.0}'
6. Run the benchmark harness
python benchmark/run_benchmark.py --warmup --trials 5
Repository structure
CPUInference/
  Project1/
    benchmark/
    instrumentation/
    prompts/
    runtime/
    server/
    benchmark/results/

Key files and folders:

server/ — FastAPI app, routes, request schemas

runtime/ — model loading, tokenization, generation, streaming logic

instrumentation/ — timing and memory utilities

prompts/ — benchmark prompt files

benchmark/ — harness and experiment runner

benchmark/results/ — stored experiment outputs

Why this project matters for inference roles

This project is small, but it teaches the right habits for inference engineering:

understand the request path

measure before guessing

separate warmup from steady state

collect structured results

run repeated trials

store experiment context with the outputs

That is the mindset I want before moving into:

CPU quantization

GPU serving

KV-cache analysis

batching

multi-GPU scaling

distributed inference systems

What comes next

Planned next steps for this repo include:

CPU quantization experiments

cold-start load-time experiments

GPU-based serving projects

KV-cache and memory budget studies

multi-GPU communication-focused work

distributed serving systems

Final takeaway

Project 1 turned a minimal CPU text generation server into a measured inference system.

I built:

a streaming API

timing instrumentation

RAM measurement

a benchmark harness

repeated experiments

reproducible result packaging

The most important outcome was not the tiny model’s output quality. The most important outcome was learning how to reason about inference as a systems problem: request flow, observability, benchmarking, and reproducibility.
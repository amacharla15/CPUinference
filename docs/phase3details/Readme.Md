# Phase 3 - CPU Timing Instrumentation

## What I did in this phase

In this phase, I turned the streaming inference server into something that can actually measure CPU-side inference behavior.

Before this phase, the server could:
- accept a prompt
- run generation on CPU
- stream output back to the client

But it could not clearly tell me:
- how long tokenization took
- how long it took before the first visible output appeared
- how long the rest of the stream took
- how fast the output continued after streaming started

So in this phase, I added timing instrumentation around the request lifecycle.

The `/generate` endpoint still streams output, but now it also measures and reports timing information for each request.

---

## Goal of this phase

The goal of this phase was to make the project actually teach CPU inference behavior instead of only serving text.

At this point, the project is no longer just a small API demo. It now measures important parts of the CPU inference path.

I specifically wanted to observe:
- tokenization cost
- time to first visible output
- time spent after streaming starts
- approximate output throughput

---

## What changed in this phase

I added timing logic to the server so that one request now goes through this measured flow:

1. request arrives
2. prompt is tokenized
3. generation starts
4. first streamed chunk appears
5. more chunks continue streaming
6. stream finishes
7. final metrics summary is produced

This means the server now measures not just whether generation works, but how long different parts of the request take.

---

## Important concepts in this phase

### Tokenization
Tokenization means converting raw input text into token IDs that the model can read.

Example idea:
- input text: `"Explain recursion simply in one paragraph."`
- tokenizer converts it into token IDs
- the model uses token IDs, not raw text

This is CPU work.

### Prompt tokens
This is the number of input tokens produced from the prompt after tokenization.

In my test run:
- `prompt_tokens = 9`

That means the input prompt became 9 model input tokens.

### TTFT
TTFT means **Time To First Token**.

In this project, because I am streaming decoded text chunks instead of exposing raw internal model tokens directly, it is more accurate to think of this as:

**Time To First Visible Chunk**

This tells me how long the user waits before seeing the first piece of streamed output.

### Prefill
Prefill is the model work that happens after reading the prompt and before useful streamed output starts appearing.

Because I am using a high-level Hugging Face streaming path, I do not get a perfect direct internal prefill timestamp.

So I used an approximation:

- `approx_prefill_plus_first_chunk_ms`

This means:
- after tokenization finished
- how long it took until the first visible streamed chunk appeared

### Chunk
A chunk is one piece of streamed text that gets sent to the client.

Examples from my run looked like:

- `data: Most`
- `data: rented`
- `data: bravery`

These are streamed text pieces.

A chunk is **not guaranteed to be exactly one token**.

It can be:
- a word
- part of a word
- multiple tokens decoded together
- messy text fragments

### Chunk gap
Chunk gap means the time difference between one streamed chunk and the next streamed chunk.

This gives a rough idea of how quickly output continues once generation has started streaming.

### Stream time
Stream time means:
- from the first visible chunk
- until the stream fully finishes
- how long that part took

### Total time
Total time means the full request duration from start to finish.

### Decode throughput
This estimates how fast the output continued after the first chunk.

In this project, it is estimated from:
- output token count estimate
- post-first-chunk stream time

---

## Files changed in this phase

### `instrumentation/timers.py`
This file was used to centralize timing helpers.

It provides:
- `now_s()`
- `elapsed_ms(...)`

This keeps timing code clean and reusable.

---

### `instrumentation/logging_utils.py`
This file was used to print one structured JSON metrics record per request.

This gives me a clean server-side log line for each request.

---

### `runtime/model_runner.py`
This file was extended so the runtime could support measured tokenization and measured streaming.

New logic added:
- `tokenize_prompt(...)`
- `stream_from_inputs(...)`
- `estimate_token_count(...)`

This changed the runtime from:
- just generating text

to:
- exposing measurable steps in the CPU inference path

---

### `server/routes.py`
This file was updated so the route could:
- record request start time
- measure tokenization separately
- detect first chunk time
- track chunk gaps
- estimate stream-time throughput
- print structured metrics
- send a final `event: done` summary back to the client

This made the route responsible not just for streaming, but also for collecting request-level timing results.

---

## How the measured request flow works

The `/generate` request now behaves like this:

1. request enters the FastAPI route
2. request start time is recorded
3. prompt is tokenized and tokenization time is measured
4. runtime starts streaming generation
5. when the first chunk appears, TTFT is recorded
6. every later chunk is tracked
7. chunk gaps are recorded
8. when the stream ends, total metrics are computed
9. final metrics are:
   - printed in the server log
   - also sent to the client as `event: done`

---

## Example observed client output

```text
data: Most 

data: rented 

data: bravery 

data: grandchildrenOutside653 

data: boils 

data: grandchildrenPros 

data: Medic�

data: Wheels 

data: courtyard

data: predatorsozyg 

data: rubbingPros448ived 

data: Television

data: perhaps

data: Pocketived 

data: Redux

data: clearer

data: membership 

data: Bendozyg

data: bravery 

data: clearer

data: bravery

data: bravery

data: boils448 

data: courtyard

data: Late

data: courtyard 

data: brutality

data: BendOutside

event: done
data: {"prompt_tokens": 9, "output_tokens_est": 42, "tokenization_ms": 38.768, "ttft_ms": 1131.036, "approx_prefill_plus_first_chunk_ms": 1092.268, "stream_time_ms": 168.169, "total_time_ms": 1299.205, "decode_tokens_per_sec_est": 243.802}

The generated text is still low quality because the model is intentionally tiny, but the important point is that the timing instrumentation worked correctly.

Example observed server-side metrics log
{"event": "request_metrics", "prompt_chars": 42, "prompt_tokens": 9, "output_tokens_est": 42, "max_tokens_requested": 40, "temperature": 0.7, "tokenization_ms": 38.768, "ttft_ms": 1131.036, "approx_prefill_plus_first_chunk_ms": 1092.268, "stream_time_ms": 168.169, "total_time_ms": 1299.205, "chunk_count": 28, "chunk_gap_ms": [3.73, 4.066, 9.787, 4.876, 9.815, 9.164, 5.182, 5.051, 8.55, 16.28, 4.719, 4.591, 9.28, 4.436, 5.388, 4.915, 8.036, 4.649, 4.243, 4.197, 4.0, 8.747, 4.657, 4.8, 4.822, 3.939, 5.568], "decode_tokens_per_sec_est": 243.802, "error": null}

This is the structured metrics record for one request.

What these numbers mean

From my observed run:

tokenization_ms = 38.768

ttft_ms = 1131.036

approx_prefill_plus_first_chunk_ms = 1092.268

stream_time_ms = 168.169

total_time_ms = 1299.205

decode_tokens_per_sec_est = 243.802

Interpretation

This tells me that:

tokenization was relatively small

most of the request latency happened before the first visible output

once streaming started, the rest of the output finished much more quickly

So for this warm CPU request, the expensive part was getting to the first output, not continuing the stream afterward.

That is the main CPU inference insight from this phase.

Why this phase is important for CPU inference

This is the first phase where the project clearly shows CPU inference timing behavior.

Up to this point, the project had mostly focused on:

API setup

runtime setup

streaming behavior

But in this phase, I finally measured:

CPU preprocessing cost

first visible output delay

post-first-chunk streaming behavior

total request latency

This makes the project much more useful for understanding inference behavior on CPU.

Limitations of this phase

A few things are still approximate:

chunks are not exact raw model tokens

TextIteratorStreamer gives decoded text pieces, not guaranteed one-token outputs

output_tokens_est is estimated by tokenizing the final generated text afterward

approx_prefill_plus_first_chunk_ms is not a perfect internal prefill measurement

These limitations are expected because I am currently using a high-level streaming API instead of lower-level model internals.

Even with that limitation, the phase still gives a meaningful latency breakdown at the server boundary.

Main learning from this phase

The biggest learning from this phase is:

CPU inference should be measured in stages

tokenization is separate from generation

first visible output is a very important latency milestone

once generation starts streaming, it may proceed much faster than the initial prompt-processing path

This is the phase where the project started behaving like a CPU inference measurement project instead of only a streaming API.

Status

Phase 3 is complete.

The project now:

streams output

measures tokenization time

measures time to first visible chunk

measures stream completion time

estimates decode throughput

prints structured request metrics

sends a final timing summary back to the client


## Main things to remember from Phase 3

These are the few things you should remember, not every line of code:

### 1. What Phase 3 was really about
Phase 3 was about **measuring CPU inference timing**, not just making the server work.

### 2. Tokenization
Tokenization = text → token IDs  
This is CPU preprocessing work.

### 3. TTFT
TTFT = time until the first visible streamed chunk appears.  
This is one of the most important user-facing latency metrics.

### 4. Chunk
A chunk is one streamed piece of text sent to the client.  
A chunk is **not always exactly one token**.

### 5. Why the thread exists
`model.generate()` is blocking.  
So we use a background thread to let generation keep running while the main request path keeps reading available streamed text pieces and sending them to the client.

This is the most important threading idea to remember:

> one path produces output, another path forwards output

### 6. What the main CPU lesson was
For your run:
- tokenization was small
- most latency happened before the first visible output
- after the first chunk, streaming continued much faster
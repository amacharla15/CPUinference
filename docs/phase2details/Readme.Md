# Phase 2 - Streaming Output

## What was done in this phase

In this phase, I changed the `/generate` endpoint from a full-response API into a streaming API.

Before this phase, the server waited until the model finished generating all output and then returned one final response.

After this phase, the server sends generated text incrementally while generation is still running.

This means the client can start seeing output immediately instead of waiting for the entire completion.

---

## Goal of this phase

The goal of this phase was to make the inference server behave more like a real LLM serving system.

Instead of:

1. receive request
2. run generation
3. return full response at the end

the server now works like this:

1. receive request
2. start generation
3. send chunks of output as they become available
4. close the stream when generation finishes

This improves perceived responsiveness and better matches how production model APIs usually behave.

---

## Why this phase matters

Streaming matters because users usually care about how quickly they see the first visible output, not just how quickly the entire request finishes.

Even if total generation time is the same, streaming makes the system feel much faster because the user sees progress early.

This phase also changed the API behavior from a normal one-shot response into a long-lived streaming response.

That is a meaningful backend design change.

---

## Important concepts used in this phase

### Streaming
Streaming means sending output incrementally over time instead of returning everything at once.

### Chunk
A chunk is one small piece of streamed text sent to the client.

### SSE
SSE stands for Server-Sent Events.  
This is a simple HTTP streaming format where each event is sent like:

```text
data: some text

data: more text
StreamingResponse

This is FastAPI’s response type for returning data gradually instead of as one final body.

Generator

A generator is a Python function that uses yield to produce values one by one over time.

Blocking generation

model.generate() is a blocking call, so it does not naturally return partial output while it is still running.

TextIteratorStreamer

This Hugging Face utility allows generated text to be consumed incrementally while the model is generating.

Files changed in this phase
runtime/model_runner.py

This file was updated to support streaming output.

A new method was added:

stream_text(prompt, max_tokens, temperature)

This method:

tokenizes the prompt

creates a TextIteratorStreamer

starts model.generate() in a background thread

yields text chunks as they arrive from the streamer

The existing generate_text() method still exists, but the streaming path is now used by the API.

server/routes.py

This file was updated so /generate now returns a streaming response instead of a normal JSON response.

The route now:

gets the shared runner from app.state

calls runner.stream_text(...)

wraps the yielded chunks in SSE format

returns StreamingResponse(..., media_type="text/event-stream")

This changed the endpoint behavior from full-response to streaming-response.

Files that did not need major changes
server/schemas.py

The request schema did not need to change because the input is still:

prompt

max_tokens

temperature

Only the output behavior changed.

server/app.py

The startup logic remained the same.

The shared runner is still created once at application startup and reused across requests.

Current streaming flow

The /generate request now follows this path:

client sends POST /generate

FastAPI validates the request body

route gets the shared ModelRunner

route calls runner.stream_text(...)

runtime starts generation in a background thread

TextIteratorStreamer receives generated text chunks

route yields each chunk as SSE output

client sees data: ... lines incrementally

stream closes when generation finishes

Example request
curl -N -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain recursion simply in one paragraph.","max_tokens":40,"temperature":0.7}'

The -N flag is important because it disables curl buffering and allows streamed chunks to appear immediately.

Example observed output
data:  

data: factors 

data: grandchildren 

data: grandchildren 

data: factors� 

data: Bend 

data: Singapore

data: praying 

data: Medic 

data: grandchildrenacious 

data: factors 

data: representations 

data: factors

data: workshops 

data: mutualMost 

data: Boone

data: rubbingSexual 

data: Pocket 

data: lined

data: predators 

data: Bend 

data: courtyardMiniSexual 

data: omegaoblSexualshows 

data: deflect 

data: Pocket 

data: membership

data: clearer 

data: predators 

data: perhapsived 

data: skillet 

data: Medic

The generated content is low quality because the model is intentionally tiny (sshleifer/tiny-gpt2), but the important result is that output is clearly streaming incrementally.

What this phase proved

By the end of this phase, I verified that:

the model can stream output incrementally

FastAPI StreamingResponse works correctly

SSE formatting is working

the runtime layer can expose streamed text

the route layer can forward streamed chunks to the client

the client can consume the stream using curl -N

This means the server now supports real-time output delivery instead of only final-response delivery.

Key implementation details
Runtime side

The runtime uses:

AutoTokenizer

AutoModelForCausalLM

TextIteratorStreamer

a background Thread

The thread runs model.generate(...) while the streamer yields decoded text chunks.

API side

The route uses:

a nested generator function

SSE formatting via yield f"data: {chunk}\n\n"

StreamingResponse with media_type="text/event-stream"

Main learning from this phase

The main lesson from this phase is that streaming requires changes in both:

the runtime layer

the API response layer

The runtime must expose incremental output, and the route must return a response type that can forward that incremental output to the client.

This phase also reinforced the importance of keeping API logic and runtime logic separate.

Because the project already had a clean runtime boundary, I was able to add streaming without redesigning the whole application.
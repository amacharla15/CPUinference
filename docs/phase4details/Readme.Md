RAM Measurement and Memory Analysis
What I did in this phase

In Phase 4, I added RAM measurement to my CPU inference server so I could observe not only how long each request took, but also how much process memory the request used while it was running.

Before this phase, my server already measured timing metrics such as:

prompt token count

tokenization time

TTFT

prefill plus first chunk time
stream time

total time and 
decode tokens/sec estimate

In this phase, I extended the same request lifecycle to also measure:

rss_start_mb

peak_rss_mb

rss_end_mb

rss_delta_mb

peak_rss_delta_mb

I also included these memory metrics in the final SSE event: done summary so I could see them directly from curl -N output.

Why this phase matters

This phase matters because an inference server is not only about latency. It is also about resource usage.

A model can respond quickly but still be expensive or unstable if it uses too much memory. In real inference systems, memory matters for:

model fit on the machine

request concurrency

prompt scaling

avoiding out-of-memory failures

deployment sizing

understanding warm vs cold behavior

So this phase helped me move from “the server works” to “the server is observable.”

Important terminology
RAM

RAM is the computer’s working memory. It stores the live Python process, loaded model, tokenizer, tensors, and temporary request-time data while the server is running.

RSS

RSS means Resident Set Size. It is the amount of the process currently resident in physical memory.

In this phase, I used RSS as my main process-level RAM metric.

Baseline memory

Baseline memory is the process memory already in use before a request starts. In my project, this already includes:

FastAPI process

Python runtime

loaded tokenizer

loaded model weights

Peak memory

Peak memory is the highest RSS observed while the request was active.

Delta memory

Delta memory is the difference between current memory and starting memory.

For example:

if request starts at 372 MB

peaks at 374 MB

then the peak delta is 2 MB

Memory sampler

A memory sampler is a lightweight background thread that checks process RSS repeatedly during the request so that short-lived peaks are not missed.

Relevant interview questions
1. Why is peak memory more useful than only measuring end memory?

Because memory can spike during generation and then fall before the request ends. End memory may hide the true highest usage.

2. Why use RSS in this phase?

RSS is a simple and practical process-level memory metric that is easy to collect with psutil.

3. Why sample memory continuously instead of checking only before and after?

Because temporary spikes may happen in the middle of tokenization or generation.

4. Why separate baseline memory from request memory?

Because the model is already loaded before the request starts, so not all process memory belongs to the current request.

What I implemented

I added a memory helper using psutil and a background sampler thread.

The implementation flow was:

start request timer

start memory sampler

tokenize the prompt

stream generated chunks

stop memory sampler in the final cleanup block

compute start, peak, end, and delta RSS values

add memory fields to:

server-side structured metrics log

client-side final SSE event: done summary

This matched the same request boundary I already built in Phase 3.

Files involved in this phase
instrumentation/memory.py

I created the memory helper and sampler logic here.

server/routes.py

I integrated RAM measurement into the /generate request lifecycle here.

server/app.py

I also logged startup memory after model load so I could see the baseline process RSS once the model was resident in memory.

Startup memory observation

When the app started and loaded sshleifer/tiny-gpt2, I observed:

rss_after_model_load_mb: 350.551

This gave me a baseline idea of how much RAM the process already used immediately after loading the model.

How I tested Phase 4

I first ran a warmup request and ignored it, because first requests often include one-time warmup overhead.

Then I ran three measured requests:

short prompt

medium prompt

long prompt

All runs used:

CPU-only inference

sshleifer/tiny-gpt2

SSE streaming

final event: done summary

Measured results
Run	Prompt tokens	Output tokens est	Tokenization ms	TTFT ms	Prefill+first chunk ms	Stream time ms	Total time ms	Decode tok/s est	RSS start MB	Peak RSS MB	RSS end MB	RSS delta MB	Peak RSS delta MB
Short	6	40	0.325	4.783	5.738	117.202	122.941	341.290	372.461	374.461	372.527	0.066	2.000
Medium	24	50	0.308	4.545	5.251	112.323	117.574	445.147	372.492	373.504	373.504	1.012	1.012
Long	60	60	0.429	7.722	8.542	171.051	179.594	350.772	373.711	373.719	373.719	0.008	0.008
Analysis
1. Longer prompts increased latency more clearly than memory

As prompt length increased from 6 to 24 to 60 tokens, the most visible effect was on timing, especially for the long prompt:

TTFT rose to 7.722 ms

total time rose to 179.594 ms

This showed that longer prompts had a clearer impact on request latency than on observed process-level RSS.

2. Process memory stayed relatively stable after warmup

Across all three measured runs, process memory stayed in a narrow range around:

372 MB to 374 MB

This means the server had already reached a fairly stable warm baseline after the model was loaded and the first request had already warmed the pipeline.

3. Peak per-request memory deltas were small

The largest temporary peak delta was:

2.000 MB in the short run

The medium and long runs showed even smaller peak deltas:

1.012 MB

0.008 MB

This tells me that for this tiny CPU model and single-request warmed setup, the additional request-time RAM cost was small compared to the already-loaded baseline model memory.

4. Warmed behavior was more stable than the first observed request

Earlier in development, my first request after startup showed much larger timing and memory effects. After warmup, the server produced much more stable measurements.

This reinforced an important inference-engineering lesson:

the first request is often not representative of steady-state performance

5. Baseline memory dominated total process RSS

The model and tokenizer were already loaded before the request started, so most process memory was part of the baseline server footprint.

That baseline dominated RSS much more than the temporary memory required by one prompt in this tiny model setup.

What I learned in this phase

The biggest things I learned were:

latency and memory are different dimensions of inference cost

process-level RAM should be measured during the request, not only before and after

peak RSS is more useful than end RSS alone

baseline model memory is often much larger than temporary per-request memory

warmup matters, and first-run measurements should be treated carefully

for a tiny warmed CPU model, prompt length changed latency more visibly than process-level RSS

Limitations of this phase

This phase measures process-level RSS, not perfectly isolated per-request memory.

That means:

RSS includes the whole Python process

memory from the already-loaded model is always present

under future concurrency, one request could affect another request’s observed RSS

very small temporary allocations may not stand out much with process-level sampling

So this phase gives a strong practical server-level memory view, but it is not a fine-grained tensor-level profiler.

Why the memory increase was small in these runs

The small peak deltas do not mean longer prompts never increase memory.

They were small here because:

the model is tiny (sshleifer/tiny-gpt2)

the process was already warm

the baseline process memory was already high relative to the request-time additions

RSS is a coarse process-level metric

So the correct conclusion is not “long prompts use no extra memory,” but rather:

in this warmed tiny-model CPU setup, latency changes were much easier to observe than RAM growth

Phase 4 conclusion

In Phase 4, I successfully added RAM measurement to my streaming CPU inference server. I instrumented request-start, peak, and end RSS, and I exposed these values through both server-side metrics logs and the client-side final SSE summary. My results showed that after warmup, the process memory stayed relatively stable around 372–374 MB, while longer prompts increased latency more clearly than they increased process-level RAM. This phase gave me a much better understanding of inference as a resource-management problem, not just a timing problem.

Main things I want to remember

RSS = process memory currently resident in RAM

baseline memory = already-loaded server and model memory before request start

peak memory = highest RAM seen during request

delta memory = extra RAM added relative to request start

warmup matters

latency and memory are separate metrics

for this tiny warmed model, latency effects were clearer than RAM effects
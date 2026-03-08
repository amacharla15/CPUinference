Benchmark Harness
What I did in this phase

In Phase 6, I built a benchmark harness for my CPU inference server so I could stop testing manually with one curl command at a time and instead run a fixed workload in a repeatable way.

Before this phase, I had already built:

a FastAPI inference server

SSE streaming output

timing instrumentation

RAM measurement

But my testing process was still mostly manual. I had to:

type requests myself

wait for output

copy final numbers by hand

compare results manually

In this phase, I turned that process into an automated benchmark workflow.

The benchmark harness now:

optionally runs a warmup request

runs a fixed set of benchmark cases

sends requests automatically to /generate

parses the SSE stream

captures the final event: done metrics

saves structured results to JSON and CSV

prints a small summary table in the terminal

This phase was about building the measurement system that later experiments can use.

Why this phase matters

This phase matters because performance work is only useful when it is repeatable.

If I only test with manual commands, then:

results are harder to compare

I can make copy-paste mistakes

my workload may not stay consistent

later experiments become messy

A benchmark harness fixes that problem by making the measurement process controlled and reusable.

That matters in real inference engineering because teams care about:

reproducibility

fair comparisons

fixed workloads

saved result files

easy re-runs after code changes

So this phase moved my project from manual testing to structured benchmarking.

Important terminology
Benchmark

A benchmark is a controlled performance test.

Benchmark harness

A benchmark harness is the script or system that runs the benchmark consistently.

Workload

A workload is the set of prompts, parameters, and request conditions used for testing.

Warmup

A warmup run is an initial request used to absorb one-time overhead before measuring steady-state behavior.

Steady state

Steady state means the more stable behavior after warmup, when the system is already initialized.

Structured results

Structured results are saved outputs in formats like JSON or CSV, instead of only reading values from terminal text.

Relevant interview questions
1. Why is a benchmark harness better than manual curl testing?

Because it makes measurements repeatable, reduces human error, and saves structured outputs for comparison.

2. Why include a warmup run?

Because the first request often pays extra one-time overhead and is usually not representative of steady-state behavior.

3. Why save both JSON and CSV?

JSON is useful for full structured records, and CSV is useful for quick tables, spreadsheets, and later analysis.

4. What is the difference between a benchmark harness and an experiment?

The harness is the reusable measurement tool. The experiment is one specific study run through that tool.

What I implemented

I created a prompt-based benchmark workflow with fixed cases.

The workflow was:

read benchmark case definitions from a JSON file

load prompt text from files

optionally run one warmup request

send each request to the server

parse the streaming SSE response

capture the final event: done JSON summary

save all case results to JSON and CSV

print a terminal summary table

This made the benchmark process automatic and consistent.

Files involved in this phase
prompts/bench_warmup.txt

Warmup prompt used before measured runs.

prompts/bench_short.txt

Short benchmark prompt.

prompts/bench_medium.txt

Medium benchmark prompt.

prompts/bench_long.txt

Long benchmark prompt.

benchmark/cases_phase6.json

Defines the benchmark cases, prompt files, and request parameters.

benchmark/run_benchmark.py

The main benchmark harness script. It runs the cases, parses SSE output, and saves results.

benchmark/results/

Stores the generated JSON and CSV benchmark results.

Benchmark cases used

I used three measured benchmark cases:

short

medium

long

Each case used a fixed prompt file and fixed generation settings.

This was important because benchmark comparisons are only fair when the workload is controlled.

How I ran Phase 6

I first started the server normally.

Then I ran the benchmark harness with warmup enabled.

Example command:

python benchmark/run_benchmark.py --warmup

This performed:

one warmup request

one short measured run

one medium measured run

one long measured run

After running, the script printed a summary and saved results to:

a JSON file

a CSV file

Phase 6 benchmark summary
Case	Prompt tokens	Output tokens	TTFT ms	Total ms	Peak RSS delta MB	Client wall ms
short	6	40	4.644	83.010	1.082	85.261
medium	24	50	9.065	98.906	0.125	101.680
long	60	60	4.392	116.825	0.336	119.116

Saved output files:

benchmark/results/benchmark_20260308_080154.json

benchmark/results/benchmark_20260308_080154.csv

Analysis
1. The benchmark harness worked correctly

The harness successfully:

completed warmup

ran all three benchmark cases

parsed SSE output correctly

captured final event: done metrics

saved structured output files

printed a usable summary table

So the main goal of this phase was achieved.

2. Total time increased as workload became larger

The total request time increased from:

83.010 ms for short

98.906 ms for medium

116.825 ms for long

This showed that the harness was sensitive enough to reflect larger workload costs.

3. Client wall time closely matched server total time

The client-side wall time stayed close to the server-reported total time in all cases.

That is a good sign because it means the harness is not introducing large measurement distortion.

4. Peak RSS delta stayed small

The observed peak RSS deltas were:

1.082 MB for short

0.125 MB for medium

0.336 MB for long

These are small, which is expected for a warmed tiny CPU model. Most of the process memory was already part of the baseline loaded model footprint.

5. TTFT showed some variation

The medium prompt showed a higher TTFT than both short and long in this one run. That means there is still normal run-to-run noise in the system.

This is an important benchmark lesson:

one run per case is enough to validate the harness, but not enough to draw a final latency law

That is exactly why the next phase exists.

What I learned in this phase

The biggest things I learned were:

manual testing is not enough for serious measurement

a benchmark harness makes experiments repeatable

warmup should be separated from measured runs

structured result storage is important for later analysis

one single run validates the harness, but multiple runs are better for drawing conclusions

a reusable benchmark system is the foundation for later experiments like prompt-length scaling

Limitations of this phase

This phase focused on building the benchmark infrastructure, not on drawing final experimental conclusions.

So the main limitations were:

only one measured pass per case

no averaging across repeated runs yet

no statistical summary yet

no graphing yet

That is fine, because those belong more naturally in the next experiment phase.

Phase 6 conclusion

In Phase 6, I built a benchmark harness for my streaming CPU inference server. The harness automated warmup, fixed workload execution, SSE summary parsing, and structured result storage in JSON and CSV. This phase was important because it turned my project from manual request testing into a repeatable benchmarking workflow. With this harness in place, I can now run later experiments more cleanly and compare inference behavior under controlled conditions.

Main things I want to remember

Phase 6 is about how to measure

a benchmark harness makes results repeatable

warmup helps isolate steady-state behavior

JSON and CSV outputs make later analysis easier

one run validates the harness

multiple runs are needed for stronger conclusions
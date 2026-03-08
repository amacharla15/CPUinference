Prompt-Length Experiments
What I did in this phase

In Phase 7, I used the benchmark harness from Phase 6 to run a prompt-length experiment. The goal of this phase was to study how changing the size of the input prompt affected inference behavior.

I kept the general server setup the same:

same FastAPI server

same CPU-only runtime

same streaming endpoint

same timing instrumentation

same RAM measurement

same benchmark harness

Then I ran three benchmark cases:

short prompt

medium prompt

long prompt

Instead of running each case only once, I ran 5 trials per case and then computed aggregated statistics such as:

mean

standard deviation

min

max

for important metrics like:

ttft_ms

total_time_ms

peak_rss_delta_mb

This phase was my first real repeated experiment, not just one-off benchmarking.

Why this phase matters

This phase matters because real performance analysis is not about one lucky run. It is about finding patterns across repeated trials.

In inference engineering, prompt length is an important workload variable because longer prompts can change:

prompt processing cost

time before the first token appears

total request time

memory behavior

So this phase helped me move from:

“I built a benchmark tool”

to:

“I used that tool to study how workload size affects performance”

That is an important transition from tooling to experimentation.

Important terminology
Prompt length

Prompt length is the amount of input text sent to the model. In my project, I measured this through prompt_tokens.

Independent variable

The thing I intentionally changed in the experiment.
In this phase, that was the benchmark case size: short, medium, and long.

Dependent variables

The things I measured in response to the workload change.
In this phase, these included TTFT, total time, and peak RSS delta.

Trial

A trial is one execution of one benchmark case.

Repeated trials

Running the same case multiple times to reduce noise and avoid trusting a single result too much.

Mean

The average value across repeated trials.

Standard deviation

A measure of how much the values varied from run to run.

Noise

Small run-to-run fluctuations caused by normal system variation.

Relevant interview questions
1. Why use repeated trials instead of one run per case?

Because one run can be noisy or unrepresentative. Repeated trials make the experiment more trustworthy.

2. What was the independent variable in this phase?

The size of the benchmark case, represented by short, medium, and long prompt workloads.

3. What metrics mattered most in this experiment?

TTFT, total request time, and peak RSS delta.

4. Why is mean useful?

Because it summarizes the typical behavior across repeated runs.

5. Why is standard deviation useful?

Because it shows how stable or noisy the measurements were across trials.

What I implemented

I upgraded the benchmark runner so that each benchmark case could run multiple times.

The updated workflow was:

optionally run a warmup request

run the short case 5 times

run the medium case 5 times

run the long case 5 times

save raw per-trial results

compute aggregated summary statistics

save the summary to CSV

print both per-trial and aggregated summaries in the terminal

This allowed me to analyze repeated prompt-length behavior instead of just one benchmark pass.

Files involved in this phase
benchmark/run_benchmark.py

I extended the benchmark harness to support repeated trials and aggregated summaries.

benchmark/results/benchmark_trials_20260308_090032.json

Saved raw per-trial results.

benchmark/results/benchmark_trials_20260308_090032.csv

Saved per-trial CSV results.

benchmark/results/benchmark_summary_20260308_090032.csv

Saved aggregated summary results.

Experiment setup

I ran the benchmark harness with:

warmup enabled

5 trials per case

This means the workload was:

short × 5

medium × 5

long × 5

for a total of 15 measured runs, after one warmup run.

Per-trial summary
Case	Trial	Prompt tokens	Output tokens	TTFT ms	Total ms	Peak RSS delta MB	Client wall ms
short	1	6	40	4.885	91.477	1.281	93.491
short	2	6	40	4.052	86.858	0.035	88.755
short	3	6	40	4.253	78.265	1.219	80.151
short	4	6	40	4.022	75.306	1.238	77.166
short	5	6	40	5.229	80.349	1.258	82.604
medium	1	24	50	8.053	96.576	0.137	98.446
medium	2	24	50	4.349	95.158	1.035	96.866
medium	3	24	50	3.828	97.635	0.008	99.462
medium	4	24	50	3.910	98.020	0.012	99.779
medium	5	24	50	3.941	94.443	0.000	96.434
long	1	60	60	3.787	119.114	0.285	121.238
long	2	60	60	3.758	118.257	0.000	120.099
long	3	60	60	4.355	122.488	0.000	124.281
long	4	60	60	7.069	138.604	0.000	140.847
long	5	60	60	4.069	122.971	0.004	125.208
Aggregated prompt-length summary
Case	Trials	Prompt token mean	TTFT mean	TTFT std	Total mean	Total std	Peak RSS delta mean
short	5	6.0	4.488	0.484	82.451	5.897	1.006
medium	5	24.0	4.816	1.628	96.366	1.382	0.238
long	5	60.0	4.608	1.250	124.287	7.390	0.058
Analysis
1. Total request time increased as the cases became larger

This was the clearest pattern in the experiment.

The average total request times were:

short: 82.451 ms

medium: 96.366 ms

long: 124.287 ms

This showed that larger benchmark cases produced higher overall request time.

2. TTFT stayed relatively similar across cases

The average TTFT values were:

short: 4.488 ms

medium: 4.816 ms

long: 4.608 ms

These values were close to each other. So in this experiment, TTFT did not show a strong clean upward trend with case size.

That means the strongest workload signal in this phase appeared in total time, not TTFT.

3. Memory deltas stayed small and noisy

The average peak RSS deltas were:

short: 1.006 MB

medium: 0.238 MB

long: 0.058 MB

I do not interpret this as “longer prompts use less memory.” Instead, I interpret it as a limitation of this setup:

the model is tiny

the process was already warm

baseline process memory dominated RSS

request-level memory changes were small and noisy at process level

So the memory signal was weak in this experiment.

4. Repeated trials were useful because there was visible noise

Some individual trials behaved differently from others.

For example:

medium trial 1 had a noticeably higher TTFT than the other medium trials

long trial 4 had a noticeably higher total time than the other long trials

This showed exactly why repeated trials matter. A single run could easily give a misleading impression.

5. This was not a perfectly isolated prompt-length-only experiment

One important limitation of my design was that I changed two things together across cases:

prompt length increased

max_tokens also increased from 40 to 50 to 60

That means the total-time increase reflects combined workload growth, not just input prompt growth.

So the correct interpretation is:

larger benchmark cases led to larger total request time

but I did not isolate prompt length completely, because output budget also changed

That is an important experimental-design lesson.

What I learned in this phase

The biggest things I learned were:

repeated trials are much better than one-off runs

total time gave a clearer workload signal than TTFT in this setup

process-level RSS was too coarse to show a strong clean prompt-size memory trend here

warmup and repeated runs are necessary for believable benchmarking

experiment design matters, because changing multiple variables at once makes conclusions less clean

Limitations of this phase

This phase had a few important limitations:

1. Prompt length was not isolated perfectly

Because max_tokens increased with case size, total work increased in two ways at once.

2. The model was very small

sshleifer/tiny-gpt2 is useful for learning mechanics, but it is too small to create strong memory effects in this setup.

3. RSS is a coarse process-level metric

It is useful, but it is not a fine-grained tensor-level memory profiler.

4. Only 5 trials were used

Five trials are enough to improve reliability, but not enough for a deep statistical study.

Phase 7 conclusion

In Phase 7, I used my benchmark harness to run repeated short, medium, and long benchmark cases and compute aggregated statistics across five trials per case. The strongest result was that average total request time increased as the benchmark cases became larger. TTFT remained relatively similar across cases, and peak RSS deltas stayed small and noisy in this warmed tiny-model CPU setup. This phase taught me that repeated trials are necessary for trustworthy benchmarking and that experimental design matters when interpreting performance results.

Main things I want to remember

Phase 7 is about using the benchmark harness to answer a workload question

repeated trials are better than one-off runs

total time showed the clearest trend in this experiment

TTFT stayed relatively flat in this setup

memory signal was weak because the model is tiny and RSS is coarse

changing prompt length and output budget together makes interpretation less clean
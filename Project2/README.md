# Project 2 — CPU Quantization Shootout

## What this project is

In Project 2, I tested whether storing a model in a smaller compressed form would help CPU inference.

The simple question was:

**If I use a smaller version of the same model, will it load faster, use less memory, or generate text faster?**

To answer that, I compared three versions of the same model:

- original model
- int8 version
- int4 version

I used the same prompts and the same general measurement flow so the comparison stayed fair.

---

## Why this project matters

People often assume quantization automatically makes inference better.

That is not always true.

A compressed model might:

- use less memory
- load faster
- run faster

But it can also:

- run slower
- use more memory than expected
- produce worse output

So the real lesson of this project is:

**do not assume — measure**

---

## Model used

For this project, I used:

- `facebook/opt-125m`

I chose this model because it is still small enough to run on CPU, but it is large enough that quantization effects are easier to observe than with a tiny toy model.

---

## What “quantization” means in simple words

A model is made of lots of internal numbers.

Quantization means storing those numbers in a smaller format.

In this project, I compared:

- original model = normal version
- int8 = smaller compressed version
- int4 = even smaller and more aggressive compressed version

The tradeoff I wanted to study was:

**memory + speed vs output behavior**

---

## What I measured

For each mode, I measured:

- model load time
- RAM before and after model load
- average generation time
- average output speed in tokens per second
- average peak temporary RAM increase during generation

I also saved the outputs to JSON files for later inspection.

---

## How I tested it

I used a fixed prompt set stored in:

- `Project2/prompts/quality_prompts.json`

The script used for all runs was:

- `Project2/scripts/run_quant_compare.py`

The script did this:

1. loaded the model
2. ran one small warmup generation
3. ran the same prompts
4. measured time and RAM
5. saved the results

The warmup run was important because the first generation often behaves differently and can distort the real measurements.

---

## Final comparison results

| Mode | Load time (ms) | RAM after load (MB) | Avg generate time (ms) | Avg tokens/sec | Avg peak RSS delta (MB) |
|---|---:|---:|---:|---:|---:|
| Original | 3000.191 | 583.227 | 825.038 | 48.608 | 0.406 |
| int8 | 5972.884 | 710.457 | 5566.115 | 7.190 | 9.094 |
| int4 | 86960.988 | 653.422 | 7112.283 | 4.504 | 15.123 |

---

## What I observed

### 1. The original model was the best in my setup

The original `facebook/opt-125m` model was:

- the fastest
- the lightest after load
- the best overall on my CPU

### 2. int8 was worse than the original

The int8 version:

- loaded slower
- used more RAM after load
- generated much slower

### 3. int4 was even worse

The int4 version:

- loaded much slower
- generated the slowest
- had the largest temporary RAM spikes

I also noticed that some int4 runs produced fewer output tokens than expected, which suggests output behavior changed too.

---

## Main conclusion

The main conclusion of Project 2 is:

**Quantization is not automatically a win.**

For my machine, model, and quantization method:

- int8 did not improve inference
- int4 did not improve inference
- the normal/original model performed best

This was actually the most important lesson from the project.

It showed that quantization depends on:

- the model
- the quantization backend/tool
- the hardware
- the runtime behavior

So real inference engineering requires benchmarking the exact setup instead of assuming smaller precision will always be better.

---

## What I learned

The biggest things I learned were:

- a smaller compressed model is not always better
- model/backend compatibility matters
- CPU quantization must be measured, not assumed
- warmup matters before measuring generation
- fair comparison means keeping the model and prompts fixed while changing only the quantization mode

---

## Project files

Main files for Project 2:

- `Project2/README.md`
- `Project2/prompts/quality_prompts.json`
- `Project2/scripts/run_quant_compare.py`
- `Project2/results/`

---

## How to run it

Activate the virtual environment first:

```bash
source .venv/bin/activate

Run the original model:

python Project2/scripts/run_quant_compare.py

Run int8:

python Project2/scripts/run_quant_compare.py --mode int8

Run int4:

python Project2/scripts/run_quant_compare.py --mode int4
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from threading import Event, Thread

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="facebook/opt-125m")
    parser.add_argument("--prompts_file", default="Project2/prompts/quality_prompts.json")
    parser.add_argument("--output_dir", default="Project2/results")
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--mode", default="fp", choices=["fp","int8","int4"])
    return parser.parse_args()


class MemorySampler:
    def __init__(self, interval_ms: float = 10.0):
        self.interval_s = interval_ms / 1000.0
        self.process = psutil.Process()
        self.start_rss_bytes = 0
        self.peak_rss_bytes = 0
        self.end_rss_bytes = 0
        self.samples = 0
        self._stop_event = Event()
        self._thread = Thread(target=self._run, daemon=True)

    def current_rss_bytes(self):
        return int(self.process.memory_info().rss)

    def start(self):
        current = self.current_rss_bytes()
        self.start_rss_bytes = current
        self.peak_rss_bytes = current
        self._thread.start()

    def _run(self):
        while not self._stop_event.is_set():
            current = self.current_rss_bytes()
            self.samples += 1
            if current > self.peak_rss_bytes:
                self.peak_rss_bytes = current
            self._stop_event.wait(self.interval_s)

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        self.end_rss_bytes = self.current_rss_bytes()
        if self.end_rss_bytes > self.peak_rss_bytes:
            self.peak_rss_bytes = self.end_rss_bytes

    def snapshot(self):
        return {
            "rss_start_mb": round(self.start_rss_bytes / (1024.0 * 1024.0), 3),
            "peak_rss_mb": round(self.peak_rss_bytes / (1024.0 * 1024.0), 3),
            "rss_end_mb": round(self.end_rss_bytes / (1024.0 * 1024.0), 3),
            "rss_delta_mb": round((self.end_rss_bytes - self.start_rss_bytes) / (1024.0 * 1024.0), 3),
            "peak_rss_delta_mb": round((self.peak_rss_bytes - self.start_rss_bytes) / (1024.0 * 1024.0), 3),
            "memory_samples": self.samples,
        }


def load_prompts(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    prompts_path = repo_root / args.prompts_file
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(prompts_path)
    process = psutil.Process()

    rss_before_load_mb = round(process.memory_info().rss / (1024.0 * 1024.0), 3)

    load_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.mode == "fp":
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
    elif args.mode == "int8":
        quant_config = QuantoConfig(weights="int8")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=quant_config,
        )
    elif args.mode == "int4":
        quant_config = QuantoConfig(weights="int4")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=quant_config,
        )
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    warmup_inputs = tokenizer("Warmup run.", return_tensors="pt")
    with torch.no_grad():
        _ = model.generate(
            input_ids=warmup_inputs["input_ids"],
            attention_mask=warmup_inputs["attention_mask"],
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    load_time_ms = round((time.perf_counter() - load_start) * 1000.0, 3)
    rss_after_load_mb = round(process.memory_info().rss / (1024.0 * 1024.0), 3)

    prompt_results = []

    for item in prompts:
        prompt_name = item["name"]
        prompt_text = item["prompt"]

        tokenization_start = time.perf_counter()
        inputs = tokenizer(prompt_text, return_tensors="pt")
        tokenization_ms = round((time.perf_counter() - tokenization_start) * 1000.0, 3)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        prompt_tokens = int(input_ids.shape[-1])

        sampler = MemorySampler(interval_ms=10.0)
        sampler.start()

        generate_start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generate_time_ms = round((time.perf_counter() - generate_start) * 1000.0, 3)

        sampler.stop()
        memory_metrics = sampler.snapshot()

        generated_ids = outputs[0][input_ids.shape[-1]:]
        output_tokens = int(generated_ids.shape[-1])
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        tokens_per_sec = 0.0
        if generate_time_ms > 0.0 and output_tokens > 0:
            tokens_per_sec = round(output_tokens / (generate_time_ms / 1000.0), 3)

        prompt_results.append(
            {
                "prompt_name": prompt_name,
                "prompt_text": prompt_text,
                "prompt_tokens": prompt_tokens,
                "tokenization_ms": tokenization_ms,
                "generate_time_ms": generate_time_ms,
                "output_tokens": output_tokens,
                "tokens_per_sec": tokens_per_sec,
                "generated_text": generated_text,
                **memory_metrics,
            }
        )

    summary = {
        "mode": args.mode,
        "model_name": args.model_name,
        "load_time_ms": load_time_ms,
        "rss_before_load_mb": rss_before_load_mb,
        "rss_after_load_mb": rss_after_load_mb,
        "avg_generate_time_ms": round(mean([row["generate_time_ms"] for row in prompt_results]), 3),
        "avg_tokens_per_sec": round(mean([row["tokens_per_sec"] for row in prompt_results]), 3),
        "avg_peak_rss_delta_mb": round(mean([row["peak_rss_delta_mb"] for row in prompt_results]), 3),
    }

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "run_id": f"project2_baseline_{run_stamp}",
        "mode": args.mode,
        "model_name": args.model_name,
        "load_time_ms": load_time_ms,
        "rss_before_load_mb": rss_before_load_mb,
        "rss_after_load_mb": rss_after_load_mb,
        "summary": summary,
        "prompts": prompt_results,
    }

    output_path = output_dir / f"baseline_{run_stamp}.json"
    output_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("Project 2 baseline summary")
    print("-" * 100)
    print(f"model_name          : {args.model_name}")
    print(f"mode                : {args.mode}")
    print(f"load_time_ms        : {load_time_ms}")
    print(f"rss_before_load_mb  : {rss_before_load_mb}")
    print(f"rss_after_load_mb   : {rss_after_load_mb}")
    print(f"avg_generate_time_ms: {summary['avg_generate_time_ms']}")
    print(f"avg_tokens_per_sec  : {summary['avg_tokens_per_sec']}")
    print(f"avg_peak_rss_delta_mb: {summary['avg_peak_rss_delta_mb']}")
    print("-" * 100)
    print()

    print("Per-prompt summary")
    print("-" * 140)
    print(
        f"{'prompt':<15} {'prompt_tok':>11} {'gen_ms':>10} {'out_tok':>10} "
        f"{'tok_per_sec':>12} {'peak_rss_delta_mb':>18}"
    )
    print("-" * 140)

    for row in prompt_results:
        print(
            f"{row['prompt_name']:<15} "
            f"{row['prompt_tokens']:>11} "
            f"{row['generate_time_ms']:>10} "
            f"{row['output_tokens']:>10} "
            f"{row['tokens_per_sec']:>12} "
            f"{row['peak_rss_delta_mb']:>18}"
        )

    print("-" * 140)
    print(f"saved_json          : {output_path}")


if __name__ == "__main__":
    main()
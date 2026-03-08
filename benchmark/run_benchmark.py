import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import httpx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/generate")
    parser.add_argument("--cases", default="benchmark/cases_phase6.json")
    parser.add_argument("--output_dir", default="benchmark/results")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--warmup_prompt_file", default="prompts/bench_warmup.txt")
    parser.add_argument("--warmup_max_tokens", type=int, default=20)
    parser.add_argument("--warmup_temperature", type=float, default=0.0)
    return parser.parse_args()


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_cases(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def extract_sse_data_value(line: str) -> str:
    value = line[5:]
    if value.startswith(" "):
        value = value[1:]
    return value


def parse_sse_stream(response):
    event_type = "message"
    data_lines = []
    generated_parts = []
    final_summary = None
    error_message = None

    def flush_event(current_event_type, current_data_lines):
        nonlocal final_summary, error_message
        if not current_data_lines and current_event_type == "message":
            return

        payload = "\n".join(current_data_lines)

        if current_event_type == "message":
            generated_parts.append(payload)
        elif current_event_type == "done":
            final_summary = json.loads(payload)
        elif current_event_type == "error":
            error_message = payload

    for line in response.iter_lines():
        if line == "":
            flush_event(event_type, data_lines)
            event_type = "message"
            data_lines = []
            continue

        if line.startswith("event:"):
            event_type = line.split(":", 1)[1].strip()
            continue

        if line.startswith("data:"):
            data_lines.append(extract_sse_data_value(line))
            continue

    if data_lines:
        flush_event(event_type, data_lines)

    return "".join(generated_parts), final_summary, error_message


def run_case(client, url: str, case_name: str, prompt_text: str, max_tokens: int, temperature: float, timeout: float):
    payload = {
        "prompt": prompt_text,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    start = time.perf_counter()

    with client.stream("POST", url, json=payload, timeout=timeout) as response:
        response.raise_for_status()
        generated_text, final_summary, error_message = parse_sse_stream(response)

    client_wall_ms = (time.perf_counter() - start) * 1000.0

    if final_summary is None:
        raise RuntimeError(f"No final SSE done summary received for case '{case_name}'")

    result = {
        "case_name": case_name,
        "prompt_chars": len(prompt_text),
        "client_wall_ms": round(client_wall_ms, 3),
        "generated_chars": len(generated_text),
        "generated_text": generated_text,
        "error_event": error_message,
        **final_summary,
    }

    return result


def save_results(results, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"benchmark_{stamp}.json"
    csv_path = output_dir / f"benchmark_{stamp}.csv"

    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_rows = []
    for item in results:
        row = dict(item)
        row.pop("generated_text", None)
        csv_rows.append(row)

    fieldnames = []
    for row in csv_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    return json_path, csv_path


def print_summary(results):
    print()
    print("Benchmark summary")
    print("-" * 120)
    print(
        f"{'case':<10} {'prompt_tokens':>13} {'output_tokens':>13} {'ttft_ms':>10} "
        f"{'total_ms':>10} {'peak_rss_delta_mb':>18} {'client_wall_ms':>15}"
    )
    print("-" * 120)

    for item in results:
        print(
            f"{item['case_name']:<10} "
            f"{item.get('prompt_tokens', ''):>13} "
            f"{item.get('output_tokens_est', ''):>13} "
            f"{item.get('ttft_ms', ''):>10} "
            f"{item.get('total_time_ms', ''):>10} "
            f"{item.get('peak_rss_delta_mb', ''):>18} "
            f"{item.get('client_wall_ms', ''):>15}"
        )

    print("-" * 120)


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cases_path = project_root / args.cases
    output_dir = project_root / args.output_dir

    cases = load_cases(cases_path)

    results = []

    with httpx.Client() as client:
        if args.warmup:
            warmup_prompt = load_text(project_root / args.warmup_prompt_file)
            print("Running warmup...")
            run_case(
                client=client,
                url=args.url,
                case_name="warmup",
                prompt_text=warmup_prompt,
                max_tokens=args.warmup_max_tokens,
                temperature=args.warmup_temperature,
                timeout=args.timeout,
            )
            print("Warmup complete.")
            print()

        for case in cases:
            case_name = case["name"]
            prompt_text = load_text(project_root / case["prompt_file"])
            max_tokens = int(case["max_tokens"])
            temperature = float(case["temperature"])

            print(f"Running case: {case_name}")
            result = run_case(
                client=client,
                url=args.url,
                case_name=case_name,
                prompt_text=prompt_text,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=args.timeout,
            )
            results.append(result)

    json_path, csv_path = save_results(results, output_dir)
    print_summary(results)
    print()
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")


if __name__ == "__main__":
    main()
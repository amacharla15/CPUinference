import argparse
import csv
import json
import math
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
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--phase_name", default="phase8")
    parser.add_argument("--experiment_name", default="prompt_length")
    parser.add_argument("--tag", default="")
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


def run_case(client, url: str, case_name: str, prompt_text: str, max_tokens: int, temperature: float, timeout: float, trial_index: int):
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
        raise RuntimeError(f"No final SSE done summary received for case '{case_name}' trial {trial_index}")

    result = {
        "case_name": case_name,
        "trial_index": trial_index,
        "prompt_chars": len(prompt_text),
        "client_wall_ms": round(client_wall_ms, 3),
        "generated_chars": len(generated_text),
        "generated_text": generated_text,
        "error_event": error_message,
        **final_summary,
    }

    return result


def mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def stddev(values):
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    variance = sum((value - m) * (value - m) for value in values) / len(values)
    return math.sqrt(variance)


def summarize_case(rows):
    metrics_to_aggregate = [
        "prompt_tokens",
        "output_tokens_est",
        "tokenization_ms",
        "ttft_ms",
        "approx_prefill_plus_first_chunk_ms",
        "stream_time_ms",
        "total_time_ms",
        "decode_tokens_per_sec_est",
        "rss_start_mb",
        "peak_rss_mb",
        "rss_end_mb",
        "rss_delta_mb",
        "peak_rss_delta_mb",
        "client_wall_ms",
    ]

    summary = {
        "case_name": rows[0]["case_name"],
        "trials": len(rows),
    }

    for metric in metrics_to_aggregate:
        values = [row[metric] for row in rows if row.get(metric) is not None]
        if not values:
            continue

        summary[f"{metric}_mean"] = round(mean(values), 3)
        summary[f"{metric}_std"] = round(stddev(values), 3)
        summary[f"{metric}_min"] = round(min(values), 3)
        summary[f"{metric}_max"] = round(max(values), 3)

    return summary


def make_run_id(phase_name: str, experiment_name: str, tag: str, stamp: str) -> str:
    parts = [phase_name, experiment_name]
    if tag.strip():
        parts.append(tag.strip())
    parts.append(stamp)
    return "_".join(parts)


def create_run_dirs(output_root: Path, run_id: str):
    run_dir = output_root / run_id
    raw_dir = run_dir / "raw"
    summary_dir = run_dir / "summary"

    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    return run_dir, raw_dir, summary_dir


def save_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv_rows(path: Path, rows):
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_raw_results(results, raw_dir: Path):
    json_path = raw_dir / "trials.json"
    csv_path = raw_dir / "trials.csv"

    save_json(json_path, results)

    csv_rows = []
    for item in results:
        row = dict(item)
        row.pop("generated_text", None)
        csv_rows.append(row)

    save_csv_rows(csv_path, csv_rows)

    return json_path, csv_path


def save_summary_results(summary_rows, summary_dir: Path):
    json_path = summary_dir / "summary.json"
    csv_path = summary_dir / "summary.csv"

    save_json(json_path, summary_rows)
    save_csv_rows(csv_path, summary_rows)

    return json_path, csv_path


def write_manifest(
    run_dir: Path,
    run_id: str,
    args,
    cases,
    summary_rows,
    raw_json_path: Path,
    raw_csv_path: Path,
    summary_json_path: Path,
    summary_csv_path: Path,
):
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "phase_name": args.phase_name,
        "experiment_name": args.experiment_name,
        "tag": args.tag,
        "url": args.url,
        "cases_file": args.cases,
        "output_dir": args.output_dir,
        "timeout": args.timeout,
        "warmup": args.warmup,
        "warmup_prompt_file": args.warmup_prompt_file,
        "warmup_max_tokens": args.warmup_max_tokens,
        "warmup_temperature": args.warmup_temperature,
        "trials": args.trials,
        "cases": cases,
        "case_names": [case["name"] for case in cases],
        "result_files": {
            "raw_json": str(raw_json_path.relative_to(run_dir)),
            "raw_csv": str(raw_csv_path.relative_to(run_dir)),
            "summary_json": str(summary_json_path.relative_to(run_dir)),
            "summary_csv": str(summary_csv_path.relative_to(run_dir)),
        },
        "summary_preview": summary_rows,
    }

    manifest_path = run_dir / "manifest.json"
    save_json(manifest_path, manifest)
    return manifest_path


def update_index(output_root: Path, run_id: str, manifest_path: Path, args):
    index_path = output_root / "index.csv"
    row = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "phase_name": args.phase_name,
        "experiment_name": args.experiment_name,
        "tag": args.tag,
        "trials": args.trials,
        "warmup": args.warmup,
        "cases_file": args.cases,
        "manifest_path": str(manifest_path.relative_to(output_root)),
    }

    existing_rows = []
    if index_path.exists():
        with index_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for item in reader:
                existing_rows.append(item)

    existing_rows.append(row)
    save_csv_rows(index_path, existing_rows)
    return index_path


def print_trial_summary(results):
    print()
    print("Per-trial summary")
    print("-" * 140)
    print(
        f"{'case':<10} {'trial':>7} {'prompt_tokens':>13} {'output_tokens':>13} "
        f"{'ttft_ms':>10} {'total_ms':>10} {'peak_rss_delta_mb':>18} {'client_wall_ms':>15}"
    )
    print("-" * 140)

    for item in results:
        print(
            f"{item['case_name']:<10} "
            f"{item['trial_index']:>7} "
            f"{item.get('prompt_tokens', ''):>13} "
            f"{item.get('output_tokens_est', ''):>13} "
            f"{item.get('ttft_ms', ''):>10} "
            f"{item.get('total_time_ms', ''):>10} "
            f"{item.get('peak_rss_delta_mb', ''):>18} "
            f"{item.get('client_wall_ms', ''):>15}"
        )

    print("-" * 140)


def print_aggregate_summary(summary_rows):
    print()
    print("Aggregated prompt-length summary")
    print("-" * 140)
    print(
        f"{'case':<10} {'trials':>8} {'prompt_tok_mean':>17} {'ttft_mean':>12} {'ttft_std':>10} "
        f"{'total_mean':>12} {'total_std':>11} {'peak_rss_delta_mean':>21}"
    )
    print("-" * 140)

    for row in summary_rows:
        print(
            f"{row['case_name']:<10} "
            f"{row['trials']:>8} "
            f"{row.get('prompt_tokens_mean', ''):>17} "
            f"{row.get('ttft_ms_mean', ''):>12} "
            f"{row.get('ttft_ms_std', ''):>10} "
            f"{row.get('total_time_ms_mean', ''):>12} "
            f"{row.get('total_time_ms_std', ''):>11} "
            f"{row.get('peak_rss_delta_mb_mean', ''):>21}"
        )

    print("-" * 140)


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cases_path = project_root / args.cases
    output_root = project_root / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

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
                trial_index=0,
            )
            print("Warmup complete.")
            print()

        for case in cases:
            case_name = case["name"]
            prompt_text = load_text(project_root / case["prompt_file"])
            max_tokens = int(case["max_tokens"])
            temperature = float(case["temperature"])

            for trial_index in range(1, args.trials + 1):
                print(f"Running case: {case_name} trial {trial_index}/{args.trials}")
                result = run_case(
                    client=client,
                    url=args.url,
                    case_name=case_name,
                    prompt_text=prompt_text,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=args.timeout,
                    trial_index=trial_index,
                )
                results.append(result)

    grouped = {}
    for row in results:
        grouped.setdefault(row["case_name"], []).append(row)

    summary_rows = []
    for case_name in grouped:
        summary_rows.append(summarize_case(grouped[case_name]))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = make_run_id(args.phase_name, args.experiment_name, args.tag, stamp)
    run_dir, raw_dir, summary_dir = create_run_dirs(output_root, run_id)

    raw_json_path, raw_csv_path = save_raw_results(results, raw_dir)
    summary_json_path, summary_csv_path = save_summary_results(summary_rows, summary_dir)
    manifest_path = write_manifest(
        run_dir=run_dir,
        run_id=run_id,
        args=args,
        cases=cases,
        summary_rows=summary_rows,
        raw_json_path=raw_json_path,
        raw_csv_path=raw_csv_path,
        summary_json_path=summary_json_path,
        summary_csv_path=summary_csv_path,
    )
    index_path = update_index(output_root, run_id, manifest_path, args)

    print_trial_summary(results)
    print_aggregate_summary(summary_rows)
    print()
    print(f"Run ID:            {run_id}")
    print(f"Run directory:     {run_dir}")
    print(f"Manifest:          {manifest_path}")
    print(f"Raw JSON:          {raw_json_path}")
    print(f"Raw CSV:           {raw_csv_path}")
    print(f"Summary JSON:      {summary_json_path}")
    print(f"Summary CSV:       {summary_csv_path}")
    print(f"Results index CSV: {index_path}")


if __name__ == "__main__":
    main()
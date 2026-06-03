"""HumanEval code generation benchmark.

Generates n completions per problem, extracts function bodies, executes
against hidden tests, and computes pass@k.

Usage::

    python scripts/tools/evaluate_humaneval.py --param_path ./params \
        --data_path HumanEval.jsonl.gz --output results.json \
        --num_samples 200 --temperature 0.8 --max_tokens 512
"""

import argparse
import json
import os
import re
import signal
import sys
from math import prod
from multiprocessing import Process, Queue
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import tqdm

from astrai.inference import InferenceEngine
from astrai.model import AutoModel
from astrai.tokenize import AutoTokenizer

HUMANEVAL_URL = (
    "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
)

_STOP_SEQUENCES = [
    "\nclass ",
    "\ndef ",
    "\n# ",
    "\nif __name__",
    "\nprint(",
    "\n\n\n",
]


def _download_humaneval(data_path: str):
    if os.path.exists(data_path):
        return
    import gzip
    import urllib.request

    os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
    print(f"Downloading HumanEval from {HUMANEVAL_URL} ...")
    tmp = data_path + ".tmp"
    urllib.request.urlretrieve(HUMANEVAL_URL, tmp)
    with gzip.open(tmp, "rb") as f_in:
        with open(data_path, "wb") as f_out:
            f_out.write(f_in.read())
    os.remove(tmp)
    print(f"  saved to {data_path}")


def _load_problems(data_path: str) -> List[dict]:
    problems = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def _extract_function_body(code: str, entry_point: str) -> Optional[str]:
    """Extract the function body from a completion."""
    pattern = rf"def\s+{re.escape(entry_point)}\b[^:]*:"
    match = re.search(pattern, code)
    if not match:
        # Use the full code as-is if we can't find the function
        return code

    body_start = match.end()
    lines = code[body_start:].split("\n")
    body_lines = []
    started = False

    for line in lines:
        stripped = line.rstrip()
        if not stripped and not started:
            continue
        if not stripped and started:
            body_lines.append("")
            continue
        if not started:
            started = True
        if stripped.lstrip() == stripped and started:
            break
        body_lines.append(stripped)

    body = "\n".join(body_lines)
    if not body.strip():
        return None
    return body


def _trim_stop_sequences(text: str) -> str:
    for stop in _STOP_SEQUENCES:
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx]
    return text


def _execute_code(problem: dict, completion: str, timeout: float = 3.0) -> bool:
    """Run the completion against hidden tests in a subprocess."""

    def _worker(queue, full_code):
        try:
            namespace = {}
            exec(full_code, namespace)
            check = namespace.get("check")
            if check is None:
                queue.put(False)
                return
            check(namespace.get(problem["entry_point"]))
            queue.put(True)
        except Exception:
            queue.put(False)

    full_code = problem["prompt"] + completion + "\n" + problem["test"]

    queue: Queue = Queue()
    proc = Process(target=_worker, args=(queue, full_code))
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return False

    try:
        return queue.get_nowait()
    except Exception:
        return False


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k."""
    if n - c < k:
        return 1.0
    return 1.0 - float(prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def _deduplicate(completions: List[str]) -> List[str]:
    seen = set()
    unique = []
    for c in completions:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def _generate(
    engine: InferenceEngine,
    prompt: str,
    num_samples: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    batch_size: int,
) -> List[str]:
    batches = [prompt] * min(batch_size, num_samples)
    completions = []
    remaining = num_samples

    while remaining > 0:
        current = min(batch_size, remaining)
        batch_prompts = batches[:current]
        outputs = engine.generate(
            prompt=batch_prompts,
            stream=False,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        if isinstance(outputs, str):
            outputs = [outputs]
        completions.extend(outputs)
        remaining -= current

    return _deduplicate(completions)


def evaluate(
    engine: InferenceEngine,
    problems: List[dict],
    num_samples: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    batch_size: int,
    k_values: Tuple[int, ...] = (1, 10, 100),
) -> Dict:
    results = {}
    all_pass_at_k = {k: [] for k in k_values}

    for problem in tqdm.tqdm(problems, desc="HumanEval", unit="problem"):
        task_id = problem["task_id"]
        prompt = problem["prompt"]
        entry_point = problem["entry_point"]

        raw_completions = _generate(
            engine,
            prompt,
            num_samples,
            max_tokens,
            temperature,
            top_p,
            top_k,
            batch_size,
        )

        completions = []
        for raw in raw_completions:
            trimmed = _trim_stop_sequences(raw)
            body = _extract_function_body(trimmed, entry_point)
            if body:
                completions.append(body)

        passed = 0
        for comp in completions:
            if _execute_code(problem, comp):
                passed += 1

        n = len(completions)
        c = passed
        result = {"task_id": task_id, "n": n, "passed": c}
        for k in k_values:
            result[f"pass@{k}"] = round(_pass_at_k(n, c, k), 4)
            all_pass_at_k[k].append(_pass_at_k(n, c, k))
        results[task_id] = result

    summary = {}
    for k in k_values:
        vals = all_pass_at_k[k]
        summary[f"pass@{k}"] = round(float(np.mean(vals)), 4)
    results["_summary"] = summary

    return results


def main():
    parser = argparse.ArgumentParser(description="HumanEval benchmark")
    parser.add_argument(
        "--param_path", type=str, default="./params", help="Model directory"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./humaneval/HumanEval.jsonl",
        help="HumanEval JSONL file (auto-download if missing)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="Completions per problem",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=512, help="Max generation tokens"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature"
    )
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Inference batch size"
    )
    parser.add_argument(
        "--problems",
        type=int,
        nargs="+",
        default=None,
        help="Specific problem indices (0-based)",
    )
    args = parser.parse_args()

    _download_humaneval(args.data_path)
    problems = _load_problems(args.data_path)
    if args.problems:
        problems = [problems[i] for i in args.problems if i < len(problems)]

    model = AutoModel.from_pretrained(args.param_path)
    tokenizer = AutoTokenizer.from_pretrained(args.param_path)
    model.to(device="cuda", dtype=torch.bfloat16)

    engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=args.batch_size,
    )

    results = evaluate(
        engine=engine,
        problems=problems,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        batch_size=args.batch_size,
        k_values=(1, 10, 100),
    )

    summary = results.pop("_summary")
    print(f"\n{'=' * 60}")
    for k, v in summary.items():
        print(f"  {k}: {v:.2%}")
    print(f"{'=' * 60}")

    if args.output:
        results["_summary"] = summary
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")

    engine.shutdown()


if __name__ == "__main__":
    main()

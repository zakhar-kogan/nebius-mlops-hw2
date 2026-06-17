"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
import sqlite3
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"
MAX_TRACKED_ITERATIONS = 3


def infer_inference_backend(base_url: str | None) -> str:
    url = (base_url or "").lower()
    if ":8000" in url and (
        "localhost" in url
        or "127.0.0.1" in url
        or "host.docker.internal" in url
        or "0.0.0.0" in url
    ):
        return "vllm"
    return "hosted_api"


def default_run_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}"


def camel_run_metadata(run_metadata: dict[str, str]) -> dict[str, str]:
    eval_run_id = run_metadata["eval_run_id"]
    return {
        "environment": run_metadata["environment"],
        "inferenceBackend": run_metadata["inference_backend"],
        "promptVersion": run_metadata["prompt_version"],
        "agentVersion": run_metadata["agent_version"],
        "evalRunId": eval_run_id,
        "sessionId": eval_run_id,
    }


# ---------- Helpers (provided) -----------------------------------------

def run_sql(
    db_id: str,
    sql: str,
    timeout: float = 5.0,
) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def eval_one(question: dict, agent_url: str, run_metadata: dict[str, str]) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness."""
    db_id = question["db_id"]
    gold_sql = question["gold_sql"]
    gold_ok, gold_rows, gold_error = run_sql(db_id, gold_sql)

    t0 = time.monotonic()
    agent_status = "ok"
    agent_error = None
    response_data: dict = {}
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                agent_url,
                json={
                    "question": question["question"],
                    "db": db_id,
                    "tags": {
                        "run_type": "eval",
                        "eval_db_id": db_id,
                        **run_metadata,
                    },
                },
            )
            resp.raise_for_status()
            response_data = resp.json()
    except Exception as e:  # noqa: BLE001
        agent_status = "error"
        agent_error = f"{type(e).__name__}: {e}"
    latency = time.monotonic() - t0

    final_sql = response_data.get("sql", "")
    pred_ok, pred_rows, pred_error = (
        run_sql(db_id, final_sql) if final_sql else (False, None, "missing SQL")
    )
    final_correct = gold_ok and pred_ok and matches(gold_rows, pred_rows)

    attempts: list[dict] = []
    seen_iterations: set[int] = set()
    for event in response_data.get("history", []):
        if event.get("node") not in {"generate_sql", "revise"}:
            continue
        sql = event.get("sql", "")
        iteration = int(event.get("iteration") or len(attempts) + 1)
        if not sql or iteration in seen_iterations:
            continue
        seen_iterations.add(iteration)
        attempt_ok, attempt_rows, attempt_error = run_sql(db_id, sql)
        attempts.append({
            "iteration": iteration,
            "sql": sql,
            "execution_ok": attempt_ok,
            "error": attempt_error,
            "correct": gold_ok and attempt_ok and matches(gold_rows, attempt_rows),
        })

    if final_sql and not attempts:
        attempts.append({
            "iteration": int(response_data.get("iterations") or 1),
            "sql": final_sql,
            "execution_ok": pred_ok,
            "error": pred_error,
            "correct": final_correct,
        })

    failure_category = "correct"
    if agent_status != "ok":
        failure_category = "agent_http_error"
    elif not gold_ok:
        failure_category = "gold_sql_error"
    elif not pred_ok:
        failure_category = "pred_sql_error"
    elif not final_correct:
        failure_category = "wrong_rows"

    return {
        "question": question["question"],
        "db_id": db_id,
        "gold_sql": gold_sql,
        "gold_ok": gold_ok,
        "gold_error": gold_error,
        "pred_sql": final_sql,
        "pred_execution_ok": pred_ok,
        "pred_error": pred_error,
        "correct": final_correct,
        "iterations": int(
            response_data.get("iterations")
            or (attempts[-1]["iteration"] if attempts else 0)
        ),
        "latency_seconds": latency,
        "agent_status": agent_status,
        "agent_error": agent_error,
        "failure_category": failure_category,
        "attempts": attempts,
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    first_pass = sum(1 for r in results if r.get("attempts") and r["attempts"][0].get("correct"))
    max_iter = max(
        [MAX_TRACKED_ITERATIONS]
        + [int(r.get("iterations") or 0) for r in results]
        + [int(a.get("iteration") or 0) for r in results for a in r.get("attempts", [])]
    )

    per_iteration: dict[str, float] = {}
    per_iteration_counts: dict[str, int] = {}
    for iteration in range(1, max_iter + 1):
        passed = 0
        for result in results:
            attempts = sorted(
                result.get("attempts", []),
                key=lambda a: int(a.get("iteration") or 0),
            )
            carried = None
            for attempt in attempts:
                if int(attempt.get("iteration") or 0) <= iteration:
                    carried = attempt
                else:
                    break
            if carried is not None and carried.get("correct"):
                passed += 1
        per_iteration[str(iteration)] = (passed / total) if total else 0.0
        per_iteration_counts[str(iteration)] = passed

    latencies = sorted(r["latency_seconds"] for r in results if r.get("agent_status") == "ok")

    def pct(p: float) -> float | None:
        if not latencies:
            return None
        idx = int(round(p * (len(latencies) - 1)))
        return latencies[idx]

    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "first_pass_correct": first_pass,
        "first_pass_accuracy": (first_pass / total) if total else 0.0,
        "per_iteration_correct": per_iteration_counts,
        "per_iteration_accuracy": per_iteration,
        "avg_iterations": (
            sum(int(r.get("iterations") or 0) for r in results) / total
        ) if total else 0.0,
        "failure_categories": dict(Counter(r.get("failure_category", "unknown") for r in results)),
        "latency_p50": pct(0.50),
        "latency_p95": pct(0.95),
        "latency_max": latencies[-1] if latencies else None,
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    parser.add_argument("--limit", type=int, default=None, help="optional smoke-test limit")
    parser.add_argument("--environment", default="staging")
    parser.add_argument(
        "--inference-backend",
        default=None,
        choices=["hosted_api", "vllm"],
        help="defaults from VLLM_BASE_URL: localhost:8000 => vllm, otherwise hosted_api",
    )
    parser.add_argument("--prompt-version", default="p0_baseline")
    parser.add_argument("--agent-version", default="a0_baseline")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    vllm_base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    run_metadata = {
        "environment": args.environment,
        "inference_backend": args.inference_backend or infer_inference_backend(vllm_base_url),
        "prompt_version": args.prompt_version,
        "agent_version": args.agent_version,
        "eval_run_id": args.run_id or default_run_id("eval"),
    }

    questions = [
        json.loads(line)
        for line in args.eval_set.read_text().splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        questions = questions[: args.limit]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url, run_metadata))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "metadata": {
            **camel_run_metadata(run_metadata),
            "agentUrl": args.agent_url,
            "evalSet": str(args.eval_set),
            "limit": args.limit,
        },
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

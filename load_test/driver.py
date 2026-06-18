"""Async load driver for the agent endpoint.

Samples questions from load_test/perf_pool.jsonl and fires them at the
agent at the requested RPS for the requested duration, recording per-
request latency and outcome.

Run:
    uv run python load_test/driver.py --rps 8 --duration 300

Writes a JSON file (default results/load_test.json) with summary + raw
per-request data.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
import random
import time
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
PERF_POOL = ROOT / "load_test" / "perf_pool.jsonl"
DEFAULT_OUT = ROOT / "results" / "load_test.json"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


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
    load_run_id = run_metadata["load_run_id"]
    return {
        "environment": run_metadata["environment"],
        "inferenceBackend": run_metadata["inference_backend"],
        "promptVersion": run_metadata["prompt_version"],
        "agentVersion": run_metadata["agent_version"],
        "verifyMode": run_metadata["verify_mode"],
        "maxIterations": run_metadata["max_iterations"],
        "promptProfile": run_metadata["prompt_profile"],
        "llmCacheBust": run_metadata["llm_cache_bust"],
        "loadRunId": load_run_id,
        "sessionId": load_run_id,
    }


async def fire_one(
    session: aiohttp.ClientSession,
    url: str,
    question: dict,
    results: list[dict],
    run_metadata: dict[str, str],
) -> None:
    payload = {
        "question": question["question"],
        "db": question["db_id"],
        "tags": {
            "run_type": "load",
            "load_db_id": question["db_id"],
            **run_metadata,
        },
    }
    t0 = time.monotonic()
    status = "ok"
    err: str | None = None
    try:
        timeout = aiohttp.ClientTimeout(total=120)
        async with session.post(url, json=payload, timeout=timeout) as resp:
            await resp.read()
            if resp.status != 200:
                status = "http_error"
                err = f"HTTP {resp.status}"
    except asyncio.TimeoutError:
        status = "timeout"
    except Exception as e:  # noqa: BLE001
        status = "client_error"
        err = f"{type(e).__name__}: {e}"
    results.append({
        "latency_seconds": time.monotonic() - t0,
        "status": status,
        "error": err,
    })


async def drive(args: argparse.Namespace) -> None:
    if not PERF_POOL.exists():
        raise SystemExit(f"{PERF_POOL} not found - run scripts/load_data.py first")
    questions = [json.loads(line) for line in PERF_POOL.read_text().splitlines() if line.strip()]
    if not questions:
        raise SystemExit(f"{PERF_POOL} is empty")

    rnd = random.Random(0)
    results: list[dict] = []
    vllm_base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    run_metadata = {
        "environment": args.environment,
        "inference_backend": args.inference_backend or infer_inference_backend(vllm_base_url),
        "prompt_version": args.prompt_version,
        "agent_version": args.agent_version,
        "load_run_id": args.run_id or default_run_id("load"),
        "verify_mode": os.environ.get("AGENT_VERIFY_MODE", "full"),
        "max_iterations": os.environ.get("AGENT_MAX_ITERATIONS", "3"),
        "prompt_profile": os.environ.get("AGENT_PROMPT_PROFILE", "normal"),
        "llm_cache_bust": os.environ.get("AGENT_LLM_CACHE_BUST", "0"),
    }
    interval = 1.0 / args.rps

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        start = time.monotonic()
        deadline = start + args.duration
        tasks: list[asyncio.Task] = []
        next_fire = start
        while time.monotonic() < deadline:
            q = rnd.choice(questions)
            tasks.append(asyncio.create_task(
                fire_one(session, args.agent_url, q, results, run_metadata)
            ))
            next_fire += interval
            sleep_for = next_fire - time.monotonic()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
        # let in-flight finish (cap drain at 60s)
        if tasks:
            await asyncio.wait(tasks, timeout=60.0)
        wall = time.monotonic() - start

    latencies = sorted(r["latency_seconds"] for r in results if r["status"] == "ok")

    def pct(p: float) -> float:
        if not latencies:
            return float("nan")
        k = int(round(p * (len(latencies) - 1)))
        return latencies[k]

    summary = {
        "requested_rps": args.rps,
        "run_id": run_metadata["load_run_id"],
        "duration_seconds": args.duration,
        "wall_clock_seconds": wall,
        "total_requests": len(results),
        "achieved_rps": (len(results) / wall) if wall > 0 else 0.0,
        "ok": sum(1 for r in results if r["status"] == "ok"),
        "timeouts": sum(1 for r in results if r["status"] == "timeout"),
        "http_errors": sum(1 for r in results if r["status"] == "http_error"),
        "client_errors": sum(1 for r in results if r["status"] == "client_error"),
        "latency_p50": pct(0.50),
        "latency_p95": pct(0.95),
        "latency_p99": pct(0.99),
        "latency_max": latencies[-1] if latencies else float("nan"),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "metadata": {
            **camel_run_metadata(run_metadata),
            "agentUrl": args.agent_url,
            "perfPool": str(PERF_POOL),
        },
        "summary": summary,
        "results": results,
    }, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rps", type=float, default=8.0, help="target requests/second")
    p.add_argument("--duration", type=int, default=300, help="seconds to drive load")
    p.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--run-id",
        default=None,
        help="metadata tag attached to Langfuse traces",
    )
    p.add_argument("--environment", default="staging")
    p.add_argument(
        "--inference-backend",
        default=None,
        choices=["hosted_api", "vllm"],
        help="defaults from VLLM_BASE_URL: localhost:8000 => vllm, otherwise hosted_api",
    )
    p.add_argument("--prompt-version", default="p0_baseline")
    p.add_argument("--agent-version", default="a0_baseline")
    args = p.parse_args()
    asyncio.run(drive(args))


if __name__ == "__main__":
    main()

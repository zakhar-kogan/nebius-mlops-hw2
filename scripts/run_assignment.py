"""Run clean assignment eval/load jobs with one explicit agent environment.

This wrapper starts the FastAPI agent, verifies local services, runs one eval or
load command with the same agent env, and writes a manifest for later reporting.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MANIFESTS = RESULTS / "manifests"
LOGS = RESULTS / "run_logs"
AGENT_URL = "http://localhost:8001"
VLLM_URL = "http://localhost:8000"
PROMETHEUS_URL = "http://localhost:9090"
LANGFUSE_URL = "http://localhost:3001"
VERIFY_MODES = {"llm_only", "full", "fast"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_text(cmd: list[str], *, check: bool = False) -> str:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def json_get(url: str, *, timeout: float = 5.0) -> Any:
    req = Request(url, headers={"User-Agent": "assignment-runner"})
    with urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body) if body else {}


def http_ok(url: str, *, timeout: float = 5.0) -> dict[str, Any]:
    try:
        req = Request(url, headers={"User-Agent": "assignment-runner"})
        with urlopen(req, timeout=timeout) as resp:
            return {"ok": 200 <= resp.status < 400, "status": resp.status, "url": url}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "url": url, "error": f"{type(exc).__name__}: {exc}"}


def prometheus_query(query: str) -> float:
    data = json_get(f"{PROMETHEUS_URL}/api/v1/query?{urlencode({'query': query})}")
    result = data.get("data", {}).get("result", [])
    if not result:
        return 0.0
    return float(result[0]["value"][1])


def wait_for_http(url: str, *, timeout_seconds: float, label: str) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        try:
            data = json_get(url, timeout=3.0)
            return {"ok": True, "url": url, "data": data}
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.5)
    raise RuntimeError(f"{label} did not become ready at {url}: {last_error}")


def wait_for_vllm_drain(timeout_seconds: float = 180.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last = {"running": None, "waiting": None}
    while time.monotonic() < deadline:
        running = prometheus_query("sum(vllm:num_requests_running)")
        waiting = prometheus_query("sum(vllm:num_requests_waiting)")
        last = {"running": running, "waiting": waiting}
        if running == 0.0 and waiting == 0.0:
            return {"ok": True, **last}
        time.sleep(2.0)
    raise RuntimeError(f"vLLM did not drain before run: {last}")


def pids_from_command(cmd: list[str]) -> set[int]:
    out = run_text(cmd)
    pids: set[int] = set()
    for token in out.replace("\n", " ").split():
        if token.isdigit():
            pids.add(int(token))
    return pids


def stale_agent_pids() -> set[int]:
    pids: set[int] = set()
    pids.update(pids_from_command(["pgrep", "-f", r"uvicorn .*agent\.server:app"]))
    pids.update(pids_from_command(["pgrep", "-f", r"agent\.server:app"]))
    try:
        pids.update(pids_from_command(["fuser", "-n", "tcp", "8001"]))
    except Exception:
        pass
    protected = {os.getpid(), os.getppid()}
    return {pid for pid in pids if pid not in protected}


def stop_stale_agents() -> list[int]:
    pids = sorted(stale_agent_pids())
    if not pids:
        return []
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if not any(Path(f"/proc/{pid}").exists() for pid in pids):
            return pids
        time.sleep(0.2)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return pids


def build_agent_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["AGENT_VERIFY_MODE"] = args.mode
    env["AGENT_MAX_ITERATIONS"] = str(args.max_iterations)
    env["AGENT_PROMPT_PROFILE"] = args.prompt_profile
    if args.schema_cache_dir:
        env["AGENT_SCHEMA_CACHE_DIR"] = str(args.schema_cache_dir)
    else:
        env.pop("AGENT_SCHEMA_CACHE_DIR", None)
    if args.llm_cache_bust:
        env["AGENT_LLM_CACHE_BUST"] = "1"
    else:
        env.pop("AGENT_LLM_CACHE_BUST", None)
    return env


def start_agent(env: dict[str, str], run_id: str) -> tuple[subprocess.Popen[bytes], Path]:
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"agent_{run_id}.log"
    log_file = log_path.open("wb")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "agent.server:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8001",
        ],
        cwd=ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    proc._assignment_log_file = log_file  # type: ignore[attr-defined]
    return proc, log_path


def stop_agent(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    log_file = getattr(proc, "_assignment_log_file", None)
    if log_file is not None:
        log_file.close()


def check_services() -> dict[str, Any]:
    langfuse_checks = [
        http_ok(f"{LANGFUSE_URL}/api/public/health"),
        http_ok(f"{LANGFUSE_URL}/api/health"),
        http_ok(LANGFUSE_URL),
    ]
    langfuse = next((check for check in langfuse_checks if check["ok"]), langfuse_checks[-1])
    checks = {
        "agent": wait_for_http(f"{AGENT_URL}/health", timeout_seconds=60, label="agent"),
        "vllm": http_ok(f"{VLLM_URL}/health"),
        "prometheus": http_ok(f"{PROMETHEUS_URL}/-/ready"),
        "langfuse": langfuse,
    }
    failed = {name: check for name, check in checks.items() if not check["ok"]}
    if failed:
        raise RuntimeError(f"service preflight failed: {json.dumps(failed, indent=2)}")
    return checks


def vllm_flags() -> list[str]:
    script = ROOT / "scripts" / "start_vllm.sh"
    flags: list[str] = []
    for raw in script.read_text().splitlines():
        line = raw.strip().rstrip("\\").strip()
        if line.startswith("--"):
            flags.append(line)
    return flags


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def run_job(args: argparse.Namespace) -> int:
    if args.mode not in VERIFY_MODES:
        raise SystemExit(f"invalid mode: {args.mode}")
    run_id = args.run_id
    env = build_agent_env(args)
    out_path = args.out.resolve()
    manifest_path = MANIFESTS / f"{run_id}.json"
    stopped_pids = stop_stale_agents()
    agent_proc: subprocess.Popen[bytes] | None = None
    started_at = utc_now()
    command: list[str]
    if args.command == "eval":
        command = [
            sys.executable,
            "evals/run_eval.py",
            "--out",
            str(out_path),
            "--environment",
            args.environment,
            "--inference-backend",
            args.inference_backend,
            "--prompt-version",
            args.prompt_version,
            "--agent-version",
            args.agent_version,
            "--run-id",
            run_id,
        ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
    else:
        command = [
            sys.executable,
            "load_test/driver.py",
            "--rps",
            str(args.rps),
            "--duration",
            str(args.duration),
            "--out",
            str(out_path),
            "--environment",
            args.environment,
            "--inference-backend",
            args.inference_backend,
            "--prompt-version",
            args.prompt_version,
            "--agent-version",
            args.agent_version,
            "--run-id",
            run_id,
        ]

    manifest: dict[str, Any] = {
        "runId": run_id,
        "commandType": args.command,
        "startedAt": started_at,
        "endedAt": None,
        "gitSha": run_text(["git", "rev-parse", "HEAD"]),
        "gitStatusShort": run_text(["git", "status", "--short"]),
        "environment": args.environment,
        "inferenceBackend": args.inference_backend,
        "promptVersion": args.prompt_version,
        "agentVersion": args.agent_version,
        "verifyMode": args.mode,
        "maxIterations": str(args.max_iterations),
        "promptProfile": args.prompt_profile,
        "llmCacheBust": str(args.llm_cache_bust).lower(),
        "schemaCacheDir": str(args.schema_cache_dir) if args.schema_cache_dir else "",
        "resultPath": str(out_path),
        "vllmFlags": vllm_flags(),
        "stoppedAgentPids": stopped_pids,
        "prometheusWindow": {"start": started_at, "end": None},
        "runnerCommand": command,
    }

    try:
        agent_proc, log_path = start_agent(env, run_id)
        manifest["agentPid"] = agent_proc.pid
        manifest["agentLogPath"] = str(log_path)
        manifest["preflight"] = check_services()
        manifest["vllmDrainBefore"] = wait_for_vllm_drain()
        write_manifest(manifest_path, manifest)
        proc = subprocess.run(command, cwd=ROOT, env=env, check=False)
        manifest["returnCode"] = proc.returncode
        try:
            manifest["vllmDrainAfter"] = wait_for_vllm_drain()
        except Exception as exc:  # noqa: BLE001
            manifest["vllmDrainAfter"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return proc.returncode
    finally:
        ended_at = utc_now()
        manifest["endedAt"] = ended_at
        manifest["prometheusWindow"]["end"] = ended_at
        if agent_proc is not None:
            if args.keep_agent:
                manifest["agentKeptRunning"] = True
            else:
                stop_agent(agent_proc)
                manifest["agentStopped"] = True
        write_manifest(manifest_path, manifest)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", default="llm_only", choices=sorted(VERIFY_MODES))
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--prompt-profile", default="normal", choices=["normal", "short"])
    parser.add_argument("--schema-cache-dir", type=Path, default=None)
    parser.add_argument("--llm-cache-bust", action="store_true")
    parser.add_argument("--environment", default="prod")
    parser.add_argument("--inference-backend", default="vllm", choices=["hosted_api", "vllm"])
    parser.add_argument("--prompt-version", default="p0_baseline")
    parser.add_argument("--agent-version", default="a0_llm_only")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--keep-agent", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    eval_parser = sub.add_parser("eval")
    add_common_args(eval_parser)
    eval_parser.add_argument("--limit", type=int, default=None)
    load_parser = sub.add_parser("load")
    add_common_args(load_parser)
    load_parser.add_argument("--rps", type=float, required=True)
    load_parser.add_argument("--duration", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    raise SystemExit(run_job(parse_args()))


if __name__ == "__main__":
    main()

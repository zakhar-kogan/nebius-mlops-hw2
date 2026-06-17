# Experiment Log

Use this as raw working memory. Distill only the important evidence into
`REPORT.md`.

## Environment

- Staging backend: Nebius OpenAI-compatible endpoint.
- Final backend: local vLLM on 1x H100 80GB.
- Final model: `Qwen/Qwen3-30B-A3B-Instruct-2507`.
- Final metrics rule: only local H100 vLLM runs count for reported latency,
  throughput, eval accuracy, and screenshots.

## Setup Commands

```bash
uv sync
uv run python scripts/load_data.py
docker compose up -d
```

## Agent Smoke

Command:

```bash
uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001
```

Questions checked:

| Time | Backend | DB | Outcome | Notes |
|---|---|---|---|---|
| TBD | Nebius | TBD | TBD | Confirm at least one verify -> revise loop. |

## Eval Runs

| Time | Backend | Command | Accuracy | First pass | P95 latency | Result file | Notes |
|---|---|---|---:|---:|---:|---|---|
| TBD | Nebius | `uv run python evals/run_eval.py --limit 3` | TBD | TBD | TBD | TBD | Smoke only. |
| TBD | H100 vLLM | `uv run python evals/run_eval.py --out results/eval_baseline.json` | TBD | TBD | TBD | `results/eval_baseline.json` | Final baseline. |
| TBD | H100 vLLM | `uv run python evals/run_eval.py --out results/eval_after_tuning.json` | TBD | TBD | TBD | `results/eval_after_tuning.json` | After serving tuning. |

## Load And Tuning Runs

Each row should follow: saw X -> hypothesized Y -> changed Z -> result W.

| Iteration | Command | Evidence | Change | Result | Screenshot |
|---:|---|---|---|---|---|
| 0 | `uv run python load_test/driver.py --rps 10 --duration 300 --run-id h100-baseline` | TBD | Baseline | TBD | `screenshots/grafana_before.png` |
| 1 | TBD | saw TBD -> hypothesized TBD | changed TBD | result TBD | TBD |


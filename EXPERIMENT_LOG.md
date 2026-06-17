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
| 2026-06-17 | Nebius hosted API | `.venv/bin/python evals/run_eval.py --out results/eval_baseline_tagged.json --environment staging --inference-backend hosted_api --prompt-version p0_baseline --agent-version a0_baseline --run-id baseline_tagged_20260617` | 36.7% | 36.7% | 0.419s | `results/eval_baseline_tagged.json` | Locked hosted baseline. 11/30 correct; all 19 failures were `wrong_rows`; per-iteration accuracy stayed flat, so revise did not recover errors. |
| TBD | H100 vLLM | `uv run python evals/run_eval.py --out results/eval_baseline.json` | TBD | TBD | TBD | `results/eval_baseline.json` | Final baseline. |
| TBD | H100 vLLM | `uv run python evals/run_eval.py --out results/eval_after_tuning.json` | TBD | TBD | TBD | `results/eval_after_tuning.json` | After serving tuning. |

## Agent And Prompt Iterations

| Iteration | Run | Observation | Hypothesis | Change | Result |
|---:|---|---|---|---|---|
| 0 | `baseline_tagged_20260617` | Executable SQL often returned the wrong rows. Examples: missing `DISTINCT`, wrong output column, literal/case/date mismatches, bad `mm:ss.xxx` time parsing, and aggregation over the wrong target. | The verifier is too permissive: it accepts plausible non-empty results instead of checking whether the selected columns, literals, date formats, and transformations match the question. | Baseline prompts `p0_baseline`; no change. | 11/30 correct, 36.7% accuracy; `wrong_rows` for every failure. |

## Load And Tuning Runs

Each row should follow: saw X -> hypothesized Y -> changed Z -> result W.

| Iteration | Command | Evidence | Change | Result | Screenshot |
|---:|---|---|---|---|---|
| 0 | `uv run python load_test/driver.py --rps 10 --duration 300 --run-id h100-baseline` | TBD | Baseline | TBD | `screenshots/grafana_before.png` |
| 1 | TBD | saw TBD -> hypothesized TBD | changed TBD | result TBD | TBD |

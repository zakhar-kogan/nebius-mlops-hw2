# MLOps Assignment Report

## Final Environment

- Model: `Qwen/Qwen3-30B-A3B-Instruct-2507`.
- Final serving backend: local vLLM on 1x H100 80GB.
- Development backend: Nebius OpenAI-compatible endpoint.
- Final-metric rule: hosted/backend staging runs are not used for reported SLO
  or accuracy claims.

## Serving Configuration

Initial H100 command:

```bash
scripts/start_vllm.sh
```

Current flags:

- `--gpu-memory-utilization 0.92`: leave a small memory margin while maximizing KV cache.
- `--max-model-len 4096`: covers the expected 1.5-3K token prompts with margin.
- `--max-num-seqs 32`: bounded concurrency for the 10+ RPS target.
- `--max-num-batched-tokens 8192`: keep batching bounded for latency.
- `--enable-prefix-caching`: reuse common system/schema prompt prefixes.
- `--disable-log-requests`: avoid request logging overhead during load tests.

Final tuned flags and rationale: TBD.

## Agent Design

The LangGraph flow is:

`attach_schema -> generate_sql -> execute -> verify -> end|revise`

The loop is capped at 3 total SQL attempts. The verifier returns JSON with
`ok` and `issue`. Revision receives the question, schema, prior SQL, execution
output, and verifier issue.

Evidence of at least one revise loop: TBD.

## Observability

Grafana dashboard: `infra/grafana/provisioning/dashboards/serving.json`.

Panels cover:

- End-to-end latency p50/p95/p99.
- Time to first token and per-output-token latency.
- Request throughput.
- Prompt and generation token rates.
- Running/waiting requests.
- KV cache usage.
- Prompt and generation length p95.

Screenshots:

- `screenshots/grafana_serving.png`: TBD.
- `screenshots/langfuse_trace.png`: TBD.
- `screenshots/langfuse_tags.png`: TBD.

## Eval Results

Baseline result file: `results/eval_baseline.json`.

| Run | Accuracy | First pass | Iteration 1 | Iteration 2 | Iteration 3 |
|---|---:|---:|---:|---:|---:|
| Baseline | TBD | TBD | TBD | TBD | TBD |
| After tuning | TBD | TBD | TBD | TBD | TBD |

## SLO Iteration Log

Target: p95 end-to-end agent latency under 5 seconds at 10+ RPS for 5 minutes.

| Iteration | Observation | Hypothesis | Change | Result |
|---:|---|---|---|---|
| 0 | TBD | TBD | Baseline | TBD |
| 1 | TBD | TBD | TBD | TBD |

Final SLO status: TBD.

## What I Would Do With More Time

- TBD after final H100 evidence.


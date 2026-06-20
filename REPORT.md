# MLOps Assignment Report

## Executive Summary

- Model/backend: `Qwen/Qwen3-30B-A3B-Instruct-2507` served by local vLLM on
  1x H100 80GB.
- Final verdict: the 10 RPS SLO was missed, but the gap is quantified and the
  bottleneck is metric-grounded.
- Best reliable final load run: 10 RPS for 5 minutes, 2990/3000 OK responses,
  p95 end-to-end agent latency 6.66s.
- Quality improved from 9/30 execution accuracy in the baseline to 17/30 after
  tuning.
- Main bottleneck: prompt and call pressure under concurrency. Larger prompts
  increased prefill work, context-window pressure, and KV-cache use, reducing
  effective serving headroom.

Final metrics in this report come from local H100 vLLM runs. Hosted
OpenAI-compatible runs were used only for development diagnostics.

## Serving and Observability

vLLM is started with `scripts/start_vllm.sh`. Final serving flags:

- `--gpu-memory-utilization 0.92`: maximize KV-cache capacity while leaving a
  small H100 memory margin.
- `--max-model-len 8192`: raised from 4096 after load tests showed vLLM 400
  errors from over-context prompts; KV usage stayed below 50%.
- `--max-num-seqs 32`: bound concurrent active sequences for the 10 RPS target.
- `--max-num-batched-tokens 8192`: keep batching useful without making latency
  tails worse.
- `--enable-prefix-caching`: reuse the repeated system and schema prompt
  prefixes.
- `--disable-log-requests`: avoid per-request logging overhead during load.

The Grafana dashboard is committed at
`infra/grafana/provisioning/dashboards/serving.json`. It covers end-to-end
latency percentiles, time-to-first-token, per-output-token latency, request
throughput, prompt/generation token rates, running/waiting requests, KV-cache
usage, and prompt/generation length p95.

Evidence artifacts:

- `screenshots/vllm_manual_query.png`
- `screenshots/grafana_serving.png`
- `screenshots/grafana_eval_run.png`
- `screenshots/grafana_before.png`
- `screenshots/grafana_after.png`
- `screenshots/langfuse_trace.png`
- `screenshots/langfuse_tags.png`

## Agent and Eval Results

The LangGraph agent uses:

`attach_schema -> generate_sql -> execute -> verify -> end|revise`

The loop is capped at 3 SQL attempts. The baseline uses LLM-only verification
with normal prompts. The final candidate uses `AGENT_VERIFY_MODE=fast`,
`AGENT_PROMPT_PROFILE=short`, compact schema rendering, `AGENT_MAX_ITERATIONS=3`,
and 4 uvicorn workers during load.

The agent loop helped. In the baseline, first-pass accuracy was 7/30 and final
accuracy was 9/30. In the final run, first-pass accuracy was 10/30 and final
accuracy was 17/30, with most of the gain appearing after revise attempts.

| Run | Result file | Agent/profile | Accuracy | First pass | Per-iteration correct | Eval p95 |
|---|---|---|---:|---:|---:|---:|
| Baseline | `results/eval_baseline.json` | `llm_only` / `normal` | 9/30 (30.0%) | 7/30 | 7 -> 8 -> 9 | 1.595s |
| Best quality seen | experiment log | `fast` / `short` | 18/30 (60.0%) | 10/30 | 10 -> 17 -> 18 | 2.190s |
| Final after tuning | `results/eval_after_tuning.json` | `fast` / `short` + compact schema | 17/30 (56.7%) | 10/30 | 10 -> 16 -> 17 | 2.293s |

The final configuration trades one eval question versus the best-quality
candidate to reduce serving pressure. This is the useful framing: the target is
not raw quality alone, but SLO-constrained quality.

One caveat: `fast` mode skips the LLM verifier after deterministic checks pass,
so the remaining schema-token cost is mostly in SQL generation and revision. I
did not run a clean no-schema ablation, so I treat schema as a measured
prompt-pressure contributor rather than the sole cause of the SLO miss.

## SLO Iteration Log

Target: p95 end-to-end agent latency under 5s at 10+ RPS for 5 minutes.

| Step | Saw | Hypothesized | Changed | Result |
|---:|---|---|---|---|
| 0 | Baseline 10 RPS load returned only 469/3000 OK responses, 1611 timeouts, and 113.08s p95 OK latency. | The baseline agent and serving stack could not sustain 10 full agent RPS; vLLM and the app were building backlog. | Ran baseline `llm_only` / normal prompts with the initial vLLM config. | SLO missed by a wide margin. |
| 1 | `fast` verifier + short prompts improved eval quality to 18/30 and reduced LLM verifier calls. vLLM p95 improved from about 4.63s to 3.42s; waiting max fell from 13 to 4; KV max fell from 36% to 26%. | Reducing LLM calls and prompt pressure should improve serving health, but app queueing might still dominate. | Selected `AGENT_VERIFY_MODE=fast` and `AGENT_PROMPT_PROFILE=short`. | OK responses rose to 2685/3000, but p95 OK latency was still 82.28s. |
| 2 | vLLM was relatively healthy, but end-to-end latency remained high. | A single agent worker was causing application-layer backlog. | Kept agent/vLLM fixed and ran uvicorn with 4 workers. | p95 OK latency dropped to 6.32s, with 2711/3000 OK responses. SLO still missed and HTTP 500s remained. |
| 3 | Captured 500s were mostly vLLM context errors; prompts exceeded `--max-model-len 4096` by up to about 830 tokens. | Remaining failures were context-window rejections, not queue depth. KV headroom made a longer context feasible. | Increased `--max-model-len` from 4096 to 8192, kept 4 workers. | Reliability improved to 2989/3000 OK, but p95 worsened to 7.13s because long prompts were now admitted instead of failing fast. |
| 4 | Prompt p95 was about 4.8k tokens and large schemas dominated the prompt. | Reducing schema annotation/sample verbosity should lower prefill cost without destroying quality. | Used compact schema rendering with `fast` + short prompts, 4 workers, and 8192 context. | Final candidate: 2990/3000 OK, p95 6.66s, vLLM p95 about 2.66s, KV max about 34%, eval 17/30. SLO still missed, but the targeted metrics moved correctly. |

I also tested more aggressive schema compaction and larger batching. Aggressive
schema recovered 18/30 eval accuracy but regressed the 10 RPS load to 21.06s
p95. Increasing `--max-num-batched-tokens` to 16384 made tail latency worse.
Those details are kept in `EXPERIMENT_LOG.md`; they support the same conclusion:
prompt size and generation tails, not SQLite execution, were the limiting
factors.

## Lessons and Next Steps

The assignment is about balancing quality with serving constraints. I initially
optimized for text-to-SQL quality, then found that each quality gain had a
serving cost: extra revise calls, larger schemas, longer execution previews, and
larger generations. The right objective is agent value per LLM call and per
prompt token.

Large prompts were the clearest serving pressure. They increased prefill work,
pushed context-window limits, consumed more KV-cache headroom, and reduced
effective concurrency. Schema compaction moved vLLM p95 and KV usage in the
right direction, but did not get the full system under 5s p95.

With more time, I would:

- Make schema context demand-driven per LLM call. The LLM verifier should not
  receive the full rendered DB schema by default; start from the question, SQL,
  and execution result, then include only relevant table/column context when the
  verifier needs it.
- Replace full-schema prompting for generation and revision with table retrieval
  plus required foreign-key bridge tables and compact value samples.
- Add hard token budgets before every LLM call, with controlled truncation
  instead of vLLM 400s.
- Shrink revise prompts separately from generate prompts: relevant tables,
  related keys, short execution preview, and verifier issue only.
- Set small per-node output limits for SQL generation, revision, and verifier
  JSON to reduce generation-tail spikes.
- Add app-side backpressure and tune `max-num-seqs`; larger prefill batching
  worsened tails in this workload.

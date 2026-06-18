# MLOps Assignment Report

## Final Environment

- Model: `Qwen/Qwen3-30B-A3B-Instruct-2507`.
- Final serving backend: local vLLM on 1x H100 80GB.
- Development backend: Nebius OpenAI-compatible endpoint.
- Final-metric rule: hosted/backend staging runs are not used for reported SLO
  or accuracy claims.

## Serving Configuration

H100 start command:

```bash
scripts/start_vllm.sh
```

Baseline flags before SLO tuning:

- `--gpu-memory-utilization 0.92`: leave a small memory margin while maximizing KV cache.
- `--max-model-len 4096`: covers the expected 1.5-3K token prompts with margin.
- `--max-num-seqs 32`: bounded concurrency for the 10+ RPS target.
- `--max-num-batched-tokens 8192`: keep batching bounded for latency.
- `--enable-prefix-caching`: reuse common system/schema prompt prefixes.
- `--disable-log-requests`: avoid request logging overhead during load tests.

The final serving run keeps those flags except `--max-model-len` is raised to
8192. The change is justified by observed vLLM 400 errors at 4096 tokens during
load; KV usage stayed below 50%, so the longer context was feasible on the H100.

## Agent Design

The baseline agent run is intentionally LLM-only verification:

```bash
AGENT_VERIFY_MODE=llm_only
AGENT_MAX_ITERATIONS=3
AGENT_PROMPT_PROFILE=normal
```

Schema handling: the FastAPI server prewarms rendered DB schemas at startup, so
the measured request path uses the in-memory schema cache. The optional
file-backed schema cache was tested separately but was not enabled for the clean
baseline; it mainly reduces cold server startup/restart work, not steady-state
request latency after prewarm. `AGENT_LLM_CACHE_BUST` is kept off for local vLLM
because changing every prompt would defeat prefix caching; it was only a hosted
API diagnostic.

The tuned agent candidates use the same graph with deterministic verifier modes
enabled. The LangGraph flow is:

`attach_schema -> generate_sql -> execute -> verify -> end|revise`

The loop is capped at 3 total SQL attempts. The verifier returns JSON with
`ok` and `issue`. Revision receives the question, schema, prior SQL, execution
output, and verifier issue.

Evidence of at least one revise loop: baseline eval reached iteration 2/3 on
some questions, improving from 7/30 first-pass correct to 9/30 final correct.

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

- `screenshots/vllm_manual_query.png`: vLLM OpenAI-compatible endpoint returning SQL.
- `screenshots/grafana_serving.png`: Grafana dashboard with vLLM panels during load.
- `screenshots/grafana_eval_run.png`: Grafana dashboard during a baseline eval run.
- `screenshots/grafana_before.png`: before/early tuning dashboard view.
- `screenshots/grafana_after.png`: after tuning dashboard view.
- `screenshots/langfuse_trace.png`: Langfuse trace showing the agent loop.
- `screenshots/langfuse_tags.png`: Langfuse trace list with run metadata tags.

## Eval Results

Baseline result file: `results/eval_baseline.json`.

| Run | Mode/profile | Accuracy | First pass | Iteration 1 | Iteration 2 | Iteration 3 | Eval p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | `llm_only` / `normal` | 9/30 (30.0%) | 7/30 (23.3%) | 7/30 | 8/30 | 9/30 | 1.595s |
| Full verifier | `full` / `normal` | 14/30 (46.7%) | 7/30 (23.3%) | 7/30 | 13/30 | 14/30 | 2.319s |
| Fast verifier | `fast` / `normal` | 14/30 (46.7%) | 8/30 (26.7%) | 8/30 | 13/30 | 14/30 | 2.314s |
| Short prompt | `full` / `short` | 17/30 (56.7%) | 9/30 (30.0%) | 9/30 | 16/30 | 17/30 | 2.450s |
| Fast + short | `fast` / `short` | 18/30 (60.0%) | 10/30 (33.3%) | 10/30 | 17/30 | 18/30 | 2.190s |
| Compact schema | `fast` / `short` | 17/30 (56.7%) | 10/30 (33.3%) | 10/30 | 16/30 | 17/30 | 2.288s |
| Budget schema | `fast` / `short` + budget schema | 16/30 (53.3%) | 10/30 (33.3%) | 10/30 | 14/30 | 16/30 | 2.269s |
| Aggressive schema | `fast` / `short` + aggressive schema | 18/30 (60.0%) | 11/30 (36.7%) | 11/30 | 16/30 | 18/30 | 2.551s |
| After tuning | `fast` / `short` + compact schema | 17/30 (56.7%) | 10/30 (33.3%) | 10/30 | 16/30 | 17/30 | 2.293s |

Baseline commentary: the loop does some work even without deterministic checks,
but the quality ceiling is low. The clean baseline metadata is
`verifyMode=llm_only`, `maxIterations=3`, `promptProfile=normal`, and local
`vllm`.

Agent selection: `fast` verifier with `short` prompts is the best candidate for
SLO tuning. It improves accuracy from 9/30 to 18/30, removes the baseline agent
HTTP errors, and avoids the LLM verifier on deterministic passes. In eval it
needed 50 SQL executions / deterministic checks across 30 questions, versus 55
for `full` + `short`.

Schema compaction remains promising rather than discarded: the aggressive schema
profile preserved the best observed eval accuracy, 18/30, and improved first-pass
accuracy to 11/30. I would explore it further because compact, well-targeted
schema context should help quality; in this run it became a serving factor only
because the aggressive profile still produced long-tail generation and context
pressure under load.

## SLO Iteration Log

Target: p95 end-to-end agent latency under 5 seconds at 10+ RPS for 5 minutes.

| Iteration | Observation | Hypothesis | Change | Result |
|---:|---|---|---|---|
| 0 | Baseline 10 RPS load produced 469 OK responses out of 3000, with 1611 timeouts and p95 OK latency 113.08s. Grafana window: 2026-06-18T09:35:37Z to 09:43:58Z. | Current agent+serving stack cannot sustain 10 full agent RPS; vLLM reaches high running-request pressure and request backlog builds. | Baseline: `llm_only`, normal prompts, current vLLM flags. | SLO missed by a wide margin; this is the official before state. |
| 1 | Agent evals showed `fast` + `short` improved quality to 18/30 and reduced LLM verifier calls. In 10 RPS load, vLLM p95 improved from ~4.63s to ~3.42s, waiting max from 13 to 4, KV max from 36% to 26%, and prefix hit rate from 59% to 82%. | Reducing prompt/call pressure should improve serving health, but end-to-end latency may still be dominated by agent-server queueing. | Selected `AGENT_VERIFY_MODE=fast`, `AGENT_PROMPT_PROFILE=short`, same vLLM flags. | OK responses improved to 2685/3000 and timeouts fell to 6, but p95 OK latency was still 82.28s. SLO still missed; next target is agent serving/backlog, not schema or cache-bust settings. |
| 2 | With selected agent, vLLM was healthy during load: p95 ~3.42s and waiting max 4, while end-to-end p95 was 82.28s. | Single agent worker was causing application-layer backlog before/around vLLM calls. | Kept agent/vLLM fixed and ran uvicorn with 4 workers. | p95 OK latency dropped to 6.32s, OK responses rose to 2711/3000, vLLM p95 was ~2.36s, waiting max 0, KV max 24%. SLO still missed narrowly and HTTP 500s remained. |
| 3 | Raising workers from 4 to 8 kept vLLM healthy but did not improve p95: 6.55s, vLLM p95 ~2.30s, waiting max 0. Captured error bodies showed most HTTP 500s were vLLM context errors: prompts exceeded `--max-model-len 4096` by up to ~830 tokens. | Remaining failures are from context-window truncation/rejection, not queue depth. KV cache headroom was still large (~25%), so a longer context should be feasible. | Keep selected agent and workers=4, increase vLLM `--max-model-len` from 4096 to 8192. | Reliability improved sharply: 2989/3000 OK, 10 timeouts, 1 HTTP 500. But p95 OK latency worsened to 7.13s because long prompts were now admitted instead of failing fast. vLLM prompt p95 rose from ~2.0k to ~4.8k tokens, vLLM p95 was ~3.40s, waiting max 4, KV max 42%. SLO still missed; next target is schema/prompt compaction. |
| 4 | Prompt p95 was ~4.8k tokens and large schemas dominated the prompt. The biggest rendered schemas were `european_football_2` (~16.7k chars), `card_games` (~14.3k), and `california_schools` (~13.3k). | Reducing schema annotation/sample verbosity should lower prefill cost without destroying quality. | Compact schema comments to 100 chars, include at most 3 sample values, and only sample categorical-like columns. Keep `fast` + `short`, 4 workers, and `max-model-len=8192`. | Quality dropped one question to 17/30. In load, prompt p95 fell to ~4.64k tokens, vLLM p95 improved to ~2.66s, KV max fell to 34%, and p95 OK latency improved to 6.66s with 2990/3000 OK. SLO still missed, but the targeted metrics moved in the expected direction. |
| 5 | Compact schema helped serving but cost one eval question. The largest schemas were still above 8k rendered chars, so two hard-cap variants were tested on the full 30-question eval before load. | A token budget might keep quality while reducing prefill further; overly aggressive compaction may remove useful descriptions or produce burstier workload behavior. | Added `AGENT_SCHEMA_PROFILE=budget` and `aggressive`. Budget keeps capped samples/descriptions; aggressive removes most samples for huge schemas and omits table blocks if needed. | Budget eval fell to 16/30, so it was not load-tested. Aggressive eval recovered 18/30 with prompt p95 ~3.8k tokens in the 300s load, but serving saturated at 32 running and 91 waiting requests; p95 OK latency worsened to 21.06s with 2975/3000 OK. Short diagnostic loads showed schema rendering was not the cost (`attach_schema` p95 0ms) and revision count was not higher; the regression correlated with much longer generations/tails. Aggressive schema is therefore not the final config. |
| 6 | Short diagnostic load also exposed context blow-ups during revise: one request tried to send ~294k input tokens, and several others were just over the 8192-token limit. | Execution previews fed back into revise were bounded by row count but not cell length; text-heavy result cells could create enormous prompts. | Added prompt-only execution rendering caps: truncate cell values to 160 chars and total rendered execution context to 4000 chars. | In a 60s diagnostic aggressive run, vLLM generation p95 max fell from ~5000 to ~182 tokens and vLLM p95 max fell from ~49s to ~4.21s. The giant 294k-token failure disappeared, but two near-8192 context errors remained and client p95 was still ~6.39s, so this is a promising candidate that still needs a full 300s load before becoming the final config. |
| 7 | The 60s render-cap diagnostic looked promising, and the prior aggressive 300s run showed late vLLM waiting pressure. | Larger prefill batches might absorb bursts once huge execution previews were capped. | Combined render capping + aggressive schema with `--max-num-batched-tokens=16384` and ran a full 300s 10 RPS load. | Rejected. The run returned 2971/2997 OK with 21 timeouts and p95 OK latency 35.06s. vLLM still saturated late: running max 32, waiting max 107, prompt p95 ~4.3k, generation p95 max ~5000, vLLM e2e p95 max ~189.6s. The combined change worsened tail latency rather than fixing queue buildup. |

Final SLO status: missed. Best latency run was 6.32s p95 with 4 agent workers
and 4096 context, but it had many context-window HTTP errors. Best reliability
run was compact schema + 8192 context with 2990/3000 OK and 6.66s p95.
Aggressive schema compaction preserved 18/30 eval quality but is rejected for the
final serving config because its 10 RPS load p95 regressed to 21.06s. Execution
render capping fixed a real prompt blow-up, but the full combined run with
larger batching regressed to 35.06s p95, so the compact-schema run remains the
best final candidate. The conclusion is not that schema compaction is harmful;
it is that this implementation needs another iteration before its quality gain
can be used without making serving latency the limiting factor.

## What I Would Do With More Time

- Replace full-schema prompting with conservative table retrieval plus mandatory
  foreign-key bridge tables, then re-run the 30-question eval before load.
- Continue schema compaction work specifically because it preserved the best
  quality result. The next version should use a token-aware schema budget,
  relation-preserving table selection, and a hard prompt guard so schema context
  stops being a latency risk.
- Run a full 300s confirmation load for execution render capping with compact
  schema and the default `--max-num-batched-tokens=8192`.
- Add a hard prompt-token guard before LLM calls so near-8192 prompts are
  truncated or failed with a controlled agent error instead of becoming vLLM 400s.
- Add request-level timeout/error categories to the agent response instead of
  surfacing vLLM context errors as HTTP 500s.

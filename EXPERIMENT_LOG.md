# Experiment Log

Raw working memory for agent and serving iterations. Distill only final,
validated evidence into `REPORT.md`.

## Environment

- Staging backend: Nebius OpenAI-compatible hosted API.
- Final backend: local vLLM on 1x H100 80GB.
- Final model: `Qwen/Qwen3-30B-A3B-Instruct-2507`.
- Final metrics rule: only local H100 vLLM runs count for reported latency,
  throughput, eval accuracy, and screenshots.
- Hosted/staging runs are iteration evidence only.

## Setup Commands

```bash
uv sync
uv run python scripts/load_data.py
docker compose up -d
uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001
```

## Caching Notes

- DB schema cache/prewarm is intentional and acceptable for evals: it caches
  static schema metadata and rendered column descriptions, not answers.
- Schema prewarm runs at FastAPI startup. `/health` reports `schemasReady`
  once rendered schemas are ready.
- Hosted prompt/backend caching is external to this repo. One hosted latency
  run was suspiciously fast and is not valid SLO evidence.
- `AGENT_LLM_CACHE_BUST=1` is a diagnostic countermeasure for hosted prompt
  caching. It appends a unique marker to LLM prompts so repeated eval prompts
  are less likely to hit an external prompt/response cache.
- Cache busting is not a final agent feature and should not be used to claim
  final H100 assignment metrics unless local serving shows the same cache risk.

## Current Architecture

Main LangGraph path:

```text
/answer
  -> attach_schema
  -> generate_sql
  -> execute
  -> deterministic_verify
      -> deterministic fail and attempts left: revise -> execute -> deterministic_verify
      -> deterministic pass in fast mode: return
      -> deterministic pass in full mode: LLM verify -> maybe revise
```

Key configuration:

- `AGENT_VERIFY_MODE=full|fast`: full runs the LLM verifier after deterministic
  pass; fast returns after deterministic pass.
- `AGENT_MAX_ITERATIONS`: total SQL attempts, including the first generation.
- `AGENT_PROMPT_PROFILE=normal|short`: normal is the quality default; short is
  an SLO experiment.
- `AGENT_SCHEMA_CACHE_DIR`: optional file-backed rendered-schema cache.
- `AGENT_LLM_CACHE_BUST=1`: diagnostic hosted-cache countermeasure.

Deterministic verifier checks include:

- read-only single `SELECT` shape;
- literal existence/case/date variants against relevant text/date columns;
- output-column hints such as IDs, school identifiers, full names, and address
  fields;
- duplicate rows when the question implies unique/list output;
- known time-conversion and domain traps from failed cases.

## Eval Runs

| Date | Backend | Run | Agent / Prompt | Accuracy | First pass | P50 latency | P95 latency | Result file | Notes |
|---|---|---|---|---:|---:|---:|---:|---|---|
| 2026-06-17 | Nebius hosted API | `baseline_tagged_20260617` | `a0_baseline` / `p0_baseline` | 11/30 | 11/30 | 0.124s | 0.419s | `results/eval_baseline_tagged.json` | Baseline. All 19 failures were `wrong_rows`; revise did not recover errors. |
| 2026-06-17 | Nebius hosted API | `iter1_strict_verifier_20260617` | `a0_baseline` / `p1_strict_verifier` | 12/30 | 11/30 | 0.447s | 4.023s | `results/eval_iter1_strict_verifier.json` | Prompt-only verifier/reviser caught one duplicate-row case, but most semantic failures remained. |
| 2026-06-17 | Nebius hosted API | `iter2_deterministic_20260617` | `a1_deterministic_verifier` / `p2_deterministic_grounding` | 19/30 | 9/30 | 0.300s | 11.509s | `results/eval_iter2_deterministic_verifier.json` | Best hosted accuracy. Deterministic checks and richer schema grounding improved correctness but increased revision loops and tail latency. |
| 2026-06-17 | Nebius hosted API | `agno_iter1_20260617` | `a1_agno_experiment` / `p2_deterministic_grounding` | 18/30 | 9/30 | 2.519s | 12.795s | `results/eval_agno_iter1.json` | Agno variant reused the same execution and verifier pieces. It did not beat LangGraph on accuracy or latency. |
| 2026-06-17 | Nebius hosted API | `iter3_fast_mode_20260617` | `a2_fast_deterministic` / `p3_short_fast` | 18/30 | 11/30 | 1.372s | 12.404s | `results/eval_iter3_fast_mode.json` | Fast mode skipped LLM verification after deterministic pass and capped attempts at 2. Accuracy stayed near iter2, but hosted p95 did not improve. |
| 2026-06-17 | Nebius hosted API | `agno_iter2_fast_mode_20260617` | `a2_agno_fast_deterministic` / `p3_short_fast` | 16/30 | 11/30 | 0.964s | 7.874s | `results/eval_agno_iter2_fast_mode.json` | Faster than Agno iter1 but accuracy regressed. Not a primary path. |
| 2026-06-17 | Nebius hosted API | `iter4_fast_normal_prewarm_20260617` | `a3_langgraph_prewarm` / `p4_normal_prewarm` | 17/30 | 9/30 | 0.129s | 0.699s | `results/eval_iter4_fast_normal_prewarm.json` | Suspicious hosted-cache artifact. Too fast for fresh generation; invalid for SLO claims. |
| 2026-06-17 | Nebius hosted API | `iter4_fast_normal_prewarm_cachebust_timed_20260617` | `a3_langgraph_prewarm_cachebust` / `p4_normal_prewarm_cachebust` | 18/30 | 8/30 | 1.641s | 3.636s | `results/eval_iter4_fast_normal_prewarm_cachebust_timed.json` | Valid hosted diagnostic run with cache busting and node timing. Still staging evidence only. |
| TBD | H100 vLLM | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Required before final reporting. |

## Iteration Audit Trail

Each row follows: saw X -> hypothesized Y -> changed Z -> result W.

| Iteration | Run | Saw | Hypothesis | Change | Result |
|---:|---|---|---|---|---|
| 0 | `baseline_tagged_20260617` | Executable SQL often returned wrong rows: missing `DISTINCT`, wrong output columns, literal/date/case mismatches, bad time parsing, and wrong aggregation target. | The verifier was too permissive and accepted plausible non-empty outputs. | Baseline prompts and graph. | 11/30 correct; 36.7% accuracy; every failure was `wrong_rows`. |
| 1 | `iter1_strict_verifier_20260617` | More questions entered revise, but only Australian Grand Prix coordinates improved. | Prompt-only verification can catch simple duplicate/cardinality issues but lacks concrete DB value evidence. | Tightened verifier/reviser prompts. | 12/30 correct; first-pass unchanged at 11/30; p95 rose to 4.023s. |
| 2 | `iter2_deterministic_20260617` | Failures clustered around deterministic patterns: literals, selected columns, duplicates, time conversion, and domain traps. | Cheap checks plus better schema grounding would fix mistakes the LLM verifier missed. | Added deterministic verifier after execution, schema descriptions, and compact value samples. | 19/30 correct; best hosted accuracy; p95 11.509s due to more revisions and LLM calls. |
| 3 | `agno_iter1_20260617` | Agno might reduce orchestration overhead or simplify the agent path. | A separate agent runtime could be faster if it reused the same SQL/verifier primitives. | Added Agno experiment with same `/answer` contract and Langfuse session metadata. | 18/30 correct; p95 12.795s; no measured advantage over LangGraph. |
| 4 | `iter3_fast_mode_20260617` | Full verifier mode spent extra LLM calls and only recovered limited extra accuracy. | Skipping LLM verification after deterministic pass should reduce latency with a small accuracy cost. | Added `AGENT_VERIFY_MODE=fast`, `AGENT_MAX_ITERATIONS=2`, and short prompts. | 18/30 correct; p95 12.404s on hosted backend; no clear hosted latency win. |
| 5 | `agno_iter2_fast_mode_20260617` | Agno plus fast deterministic mode might combine lower overhead with fewer LLM calls. | Same fast-mode logic could improve Agno latency without large accuracy loss. | Applied fast attempt cap and deterministic routing to Agno variant. | 16/30 correct; p95 7.874s; worse accuracy, so not primary. |
| 6 | `iter4_fast_normal_prewarm_20260617` | Short prompts hurt quality, and schema rendering should not happen on first request. | Normal prompts plus schema prewarm should recover accuracy while reducing cold overhead. | Restored normal prompt profile, added startup schema prewarm and optional rendered-schema cache. | 17/30 correct; p95 0.699s, but latency was suspiciously low and likely a hosted prompt-cache artifact. Invalid for SLO. |
| 7 | `iter4_fast_normal_prewarm_cachebust_timed_20260617` | Hosted endpoint appeared to cache repeated eval prompts. | Unique prompt markers would avoid exact hosted cache hits and show real uncached hosted latency. | Added diagnostic `AGENT_LLM_CACHE_BUST=1` and per-node `duration_ms` history. | 18/30 correct; p95 3.636s; valid hosted diagnostic. LLM calls dominate latency. |

## Latency Diagnosis

Latest timed hosted diagnostic:

| Node | Count | Avg | P50 | P95 | Max |
|---|---:|---:|---:|---:|---:|
| `attach_schema` | 30 | 0.134ms | 0.095ms | 0.619ms | 0.630ms |
| `generate_sql` | 30 | 1112.432ms | 1096.704ms | 1756.882ms | 1916.811ms |
| `execute` | 49 | 19.119ms | 2.108ms | 122.386ms | 201.919ms |
| `deterministic_verify` | 49 | 0.496ms | 0.249ms | 0.970ms | 9.062ms |
| `revise` | 19 | 1242.045ms | 1408.431ms | 2029.846ms | 2111.795ms |

Conclusion: verifier/schema optimizations reduce local overhead, but hosted
tail latency is dominated by LLM generation/revision and by how many LLM calls
each request needs. Final SLO claims still require local H100 vLLM eval plus
load testing.

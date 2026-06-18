# AGENTS

## Project Shape
- This is an MLOps assignment repo for local LLM serving, observability, agent evaluation, and load testing.
- Python packaging is managed by `pyproject.toml` and `uv.lock`.
- The observability stack runs through `docker-compose.yml`.
- vLLM runs on the host, not inside Compose. Prometheus scrapes host vLLM on port `8000`.
- The FastAPI agent server listens on port `8001`.

## Setup Commands
- Default Python environment: `uv sync`
- Include local vLLM serving dependencies: `uv sync --extra serve`
- Include optional Agno experiment dependencies: `uv sync --extra experiments`
- Load the BIRD subset: `uv run python scripts/load_data.py`
- Start observability stack: `docker compose up -d`
- Start vLLM: `scripts/start_vllm.sh`

If `mise` is available:
- Trust config once: `mise trust .mise.toml`
- Install pinned tools: `mise install`
- Use task aliases such as `mise run sync`, `mise run load-data`, `mise run o11y`, and `mise run vllm`.

## Environment
- `.env` contains local secrets and runtime endpoints. Do not print or commit it.
- `.env.example` documents required variables.
- `VLLM_MODEL` should match the model served by `scripts/start_vllm.sh`.
- Hosted OpenAI-compatible APIs can be used for development, but final vLLM metrics and screenshots should use the local vLLM endpoint.

## Validation
- Run focused tests with `uv run pytest tests/<file>.py` when changing agent behavior.
- For setup-only changes, validate config syntax and show the exact diff.
- Do not claim deployment is working unless the relevant command was run and its output checked.
- If Docker, model downloads, or package installs fail because of network or daemon state, report that explicitly.

## Editing Guidance
- Keep changes surgical. Preserve the existing `uv` + Docker Compose workflow.
- Do not add Nix or alternate package managers unless explicitly requested.
- Do not move vLLM into Compose without updating Prometheus, README setup steps, and port-forwarding assumptions.
- Avoid committing generated data, screenshots, `.env`, `.venv`, or local caches.

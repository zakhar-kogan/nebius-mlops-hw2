"""FastAPI wrapper exposing the agent over HTTP.

Run:
    uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001

The /answer endpoint accepts {question, db, tags?} and returns the
agent's final SQL, the result rows, and per-iteration history.
"""
from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

if os.environ.get("LANGFUSE_BASE_URL"):
    os.environ["LANGFUSE_HOST"] = os.environ["LANGFUSE_BASE_URL"]
elif os.environ.get("LANGFUSE_HOST"):
    os.environ["LANGFUSE_BASE_URL"] = os.environ["LANGFUSE_HOST"]

from agent.graph import AgentState, VLLM_BASE_URL, VLLM_MODEL, graph  # noqa: E402

# Langfuse callback handler. If keys are set we initialize it; failures
# are NOT swallowed - a misconfigured Langfuse should not silently
# produce zero traces.
_lf_handler: Any = None
_propagate_attributes: Any = None
if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
    from langfuse import propagate_attributes
    from langfuse.langchain import CallbackHandler

    _lf_handler = CallbackHandler()
    _propagate_attributes = propagate_attributes


def _camel_metadata(db_id: str, tags: dict[str, str]) -> dict[str, str]:
    metadata = {
        "dbId": db_id,
        "model": VLLM_MODEL,
        "backendBaseUrl": VLLM_BASE_URL,
    }
    key_map = {
        "environment": "environment",
        "inference_backend": "inferenceBackend",
        "prompt_version": "promptVersion",
        "agent_version": "agentVersion",
        "eval_run_id": "evalRunId",
        "load_run_id": "loadRunId",
        "session_id": "sessionId",
        "run_type": "runType",
    }
    for source, target in key_map.items():
        value = tags.get(source)
        if value:
            metadata[target] = value
    return metadata


def _session_id(tags: dict[str, str]) -> str | None:
    return tags.get("session_id") or tags.get("eval_run_id") or tags.get("load_run_id")


def _trace_tags(tags: dict[str, str]) -> list[str]:
    labels = [
        tags.get("run_type"),
        tags.get("environment"),
        tags.get("inference_backend"),
        tags.get("prompt_version"),
        tags.get("agent_version"),
    ]
    return [label for label in labels if label]


def _trace_name(tags: dict[str, str]) -> str:
    run_type = tags.get("run_type") or "request"
    return f"agent.answer.{run_type}"


app = FastAPI()


class AnswerRequest(BaseModel):
    question: str
    db: str
    tags: dict[str, str] = Field(default_factory=dict)


class AnswerResponse(BaseModel):
    sql: str
    rows: list[list[Any]] | None
    iterations: int
    ok: bool
    error: str | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest) -> AnswerResponse:
    state = AgentState(question=req.question, db_id=req.db)
    metadata = _camel_metadata(req.db, req.tags)
    config: dict[str, Any] = {
        "callbacks": [_lf_handler] if _lf_handler is not None else [],
        "metadata": metadata,
    }
    try:
        if _propagate_attributes is None:
            final = graph.invoke(state, config=config)
        else:
            with _propagate_attributes(
                session_id=_session_id(req.tags),
                metadata=metadata,
                version=req.tags.get("agent_version"),
                tags=_trace_tags(req.tags),
                trace_name=_trace_name(req.tags),
            ):
                final = graph.invoke(state, config=config)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    sql = final.get("sql", "")
    iteration = final.get("iteration", 0)
    history = final.get("history", [])
    execution = final.get("execution")

    if execution is None:
        return AnswerResponse(
            sql=sql,
            rows=None,
            iterations=iteration,
            ok=False,
            error="agent produced no execution result",
            history=history,
        )
    if not execution.ok:
        return AnswerResponse(
            sql=sql,
            rows=None,
            iterations=iteration,
            ok=False,
            error=execution.error,
            history=history,
        )

    return AnswerResponse(
        sql=sql,
        rows=[list(r) for r in (execution.rows or [])],
        iterations=iteration,
        ok=True,
        history=history,
    )

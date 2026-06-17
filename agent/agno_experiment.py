"""Optional Agno-based experiment with the same /answer contract.

Run after installing the optional extra:

    uv sync --extra experiments
    uv run uvicorn agent.agno_experiment:app --host 0.0.0.0 --port 8002

This is intentionally separate from agent.server so LangGraph remains the
baseline implementation until this variant is measured.
"""
from __future__ import annotations

import os
import re
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langfuse import observe, propagate_attributes
from pydantic import BaseModel, Field

from agent import prompts
from agent.deterministic_verifier import verify_deterministic
from agent.execution import ExecutionResult, execute_sql
from agent.graph import _extract_sql, get_max_iterations, get_verify_mode
from agent.schema import render_schema

load_dotenv()

if os.environ.get("LANGFUSE_BASE_URL"):
    os.environ["LANGFUSE_HOST"] = os.environ["LANGFUSE_BASE_URL"]
elif os.environ.get("LANGFUSE_HOST"):
    os.environ["LANGFUSE_BASE_URL"] = os.environ["LANGFUSE_HOST"]

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")

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


def _camel_metadata(db_id: str, tags: dict[str, str]) -> dict[str, str]:
    metadata = {
        "dbId": db_id,
        "model": VLLM_MODEL,
        "backendBaseUrl": VLLM_BASE_URL,
        "agentKind": "agno",
        "verifyMode": get_verify_mode(),
        "maxIterations": str(get_max_iterations()),
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
    metadata.setdefault("agentVersion", "a1_agno_experiment")
    return metadata


def _session_id(tags: dict[str, str]) -> str | None:
    return tags.get("session_id") or tags.get("eval_run_id") or tags.get("load_run_id")


def _trace_tags(tags: dict[str, str]) -> list[str]:
    labels = [
        tags.get("run_type"),
        tags.get("environment"),
        tags.get("inference_backend"),
        tags.get("prompt_version"),
        tags.get("agent_version") or "a1_agno_experiment",
        "agno",
    ]
    return [label for label in labels if label]


def _trace_name(tags: dict[str, str]) -> str:
    run_type = tags.get("run_type") or "request"
    return f"agent.answer.agno.{run_type}"


def _load_agno() -> tuple[type[Any], type[Any]]:
    try:
        from agno.agent import Agent
        from agno.models.openai import OpenAIChat
    except ImportError as exc:
        raise RuntimeError("install optional dependency with: uv sync --extra experiments") from exc
    return Agent, OpenAIChat


@observe(name="agno.llm", as_type="generation")
def _run_agno(system: str, user: str) -> str:
    Agent, OpenAIChat = _load_agno()
    model = OpenAIChat(
        id=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0.0,
    )
    agent = Agent(model=model, instructions=[system], markdown=False)
    response = agent.run(user, stream=False)
    return str(getattr(response, "content", response))


@observe(name="agno.generate_sql", as_type="chain")
def _generate_sql(question: str, schema: str) -> str:
    return _extract_sql(
        _run_agno(
            prompts.GENERATE_SQL_SYSTEM,
            prompts.GENERATE_SQL_USER.format(schema=schema, question=question),
        )
    )


@observe(name="agno.revise_sql", as_type="chain")
def _revise_sql(
    question: str,
    schema: str,
    sql: str,
    execution: ExecutionResult | None,
    issue: str,
) -> str:
    execution_text = execution.render() if execution is not None else "ERROR: SQL was not executed."
    return _extract_sql(
        _run_agno(
            prompts.REVISE_SYSTEM,
            prompts.REVISE_USER.format(
                question=question,
                schema=schema,
                sql=sql,
                execution=execution_text,
                issue=issue,
            ),
        )
    )


def _fallback_issue(execution: ExecutionResult | None) -> str:
    if execution is None:
        return "SQL was not executed"
    if not execution.ok:
        return f"SQL execution failed: {execution.error}"
    if execution.row_count == 0:
        return "SQL returned zero rows; verify exact literals and joins"
    return "deterministic checks passed; Agno experiment stops after executable SQL"


@observe(name="agent.answer.agno", as_type="agent")
def _answer_core(req: AnswerRequest) -> AnswerResponse:
    schema = render_schema(req.db)
    sql = _generate_sql(req.question, schema)
    history: list[dict[str, Any]] = []
    execution: ExecutionResult | None = None
    iteration = 0

    max_iterations = get_max_iterations()
    for iteration in range(1, max_iterations + 1):
        history.append({"node": "generate_sql" if iteration == 1 else "revise", "iteration": iteration, "sql": sql})
        execution = execute_sql(req.db, sql)
        history.append({
            "node": "execute",
            "iteration": iteration,
            "sql": sql,
            "ok": execution.ok,
            "row_count": execution.row_count,
            "columns": execution.columns or [],
            "error": execution.error,
        })
        issue = verify_deterministic(req.db, req.question, sql, execution)
        history.append({
            "node": "deterministic_verify",
            "iteration": iteration,
            "ok": issue is None,
            "issue": issue or "",
        })
        if issue is None or iteration >= max_iterations:
            break
        sql = _revise_sql(req.question, schema, sql, execution, issue)

    if execution is None:
        return AnswerResponse(sql=sql, rows=None, iterations=0, ok=False, error="agent produced no execution result", history=history)
    if not execution.ok:
        return AnswerResponse(sql=sql, rows=None, iterations=iteration, ok=False, error=execution.error, history=history)
    return AnswerResponse(
        sql=sql,
        rows=[list(row) for row in (execution.rows or [])],
        iterations=iteration,
        ok=True,
        error=None if re.match(r"^\s*(select|with)\b", sql, re.IGNORECASE) else _fallback_issue(execution),
        history=history,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "agent": "a1_agno_experiment"}


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest) -> AnswerResponse:
    metadata = _camel_metadata(req.db, req.tags)
    try:
        with propagate_attributes(
            session_id=_session_id(req.tags),
            metadata=metadata,
            version=req.tags.get("agent_version") or "a1_agno_experiment",
            tags=_trace_tags(req.tags),
            trace_name=_trace_name(req.tags),
        ):
            return _answer_core(req)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")

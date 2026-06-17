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
from pydantic import BaseModel, Field

from agent import prompts
from agent.deterministic_verifier import verify_deterministic
from agent.execution import ExecutionResult, execute_sql
from agent.graph import MAX_ITERATIONS, _extract_sql
from agent.schema import render_schema

load_dotenv()

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


def _load_agno() -> tuple[type[Any], type[Any]]:
    try:
        from agno.agent import Agent
        from agno.models.openai import OpenAIChat
    except ImportError as exc:
        raise RuntimeError("install optional dependency with: uv sync --extra experiments") from exc
    return Agent, OpenAIChat


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


def _generate_sql(question: str, schema: str) -> str:
    return _extract_sql(
        _run_agno(
            prompts.GENERATE_SQL_SYSTEM,
            prompts.GENERATE_SQL_USER.format(schema=schema, question=question),
        )
    )


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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "agent": "a1_agno_experiment"}


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest) -> AnswerResponse:
    try:
        schema = render_schema(req.db)
        sql = _generate_sql(req.question, schema)
        history: list[dict[str, Any]] = []
        execution: ExecutionResult | None = None

        for iteration in range(1, MAX_ITERATIONS + 1):
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
            if issue is None or iteration >= MAX_ITERATIONS:
                break
            sql = _revise_sql(req.question, schema, sql, execution, issue)

    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")

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

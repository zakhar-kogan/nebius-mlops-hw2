"""LangGraph agent: text-to-SQL with verify+revise loop.

Graph shape:

    START -> attach_schema -> generate_sql -> execute -> verify
                                                          |
                                              ok=true ----+----> END
                                                          |
                                              ok=false ---+----> revise -> execute -> verify (loop)

Loop is capped at MAX_ITERATIONS total generate/revise calls.

The execute node and the graph wiring are provided. `generate_sql_node` is
filled in as a worked example; you implement `verify`, `revise`, and the
conditional router following the same shape.
"""
from __future__ import annotations

import os
import re
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agent import prompts
from agent.deterministic_verifier import verify_deterministic
from agent.execution import ExecutionResult, execute_sql
from agent.schema import render_schema

DEFAULT_MAX_ITERATIONS = 3
VERIFY_MODE_FULL = "full"
VERIFY_MODE_FAST = "fast"
VERIFY_MODE_LLM_ONLY = "llm_only"
VERIFY_MODES = {VERIFY_MODE_FULL, VERIFY_MODE_FAST, VERIFY_MODE_LLM_ONLY}

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
# vLLM ignores the key, but a hosted OpenAI-compatible provider needs a real one.
# Lets you point the agent at e.g. OpenAI while iterating without a running vLLM.
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")


def get_max_iterations() -> int:
    raw = os.environ.get("AGENT_MAX_ITERATIONS", str(DEFAULT_MAX_ITERATIONS))
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MAX_ITERATIONS


def get_verify_mode() -> str:
    mode = os.environ.get("AGENT_VERIFY_MODE", VERIFY_MODE_FULL).strip().lower()
    if mode not in VERIFY_MODES:
        return VERIFY_MODE_FULL
    return mode


def llm_cache_bust_enabled() -> bool:
    return os.environ.get("AGENT_LLM_CACHE_BUST", "").strip().lower() in {"1", "true", "yes", "on"}


def _cache_bust_suffix() -> str:
    if not llm_cache_bust_enabled():
        return ""
    marker = uuid.uuid4().hex
    return f"\n\nRequest marker: {marker}. Ignore this marker; it is not part of the question."


def _duration_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


# Backward-compatible export for code that imports the previous constant.
MAX_ITERATIONS = DEFAULT_MAX_ITERATIONS


@dataclass
class AgentState:
    """State threaded through the graph. Extend with fields you need."""

    question: str
    db_id: str
    schema: str = ""
    sql: str = ""
    execution: ExecutionResult | None = None
    verify_ok: bool = False
    verify_issue: str = ""
    deterministic_issue: str = ""
    iteration: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


def llm() -> ChatOpenAI:
    """Chat client pointed at VLLM_BASE_URL (your local vLLM by default)."""
    return ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0.0,
    )


# ---- Nodes ------------------------------------------------------------

def _attach_schema(state: AgentState) -> dict:
    """Provided. Render the DB schema once at the start of the run."""
    start = time.perf_counter()
    schema = render_schema(state.db_id)
    return {
        "schema": schema,
        "history": state.history + [{
            "node": "attach_schema",
            "duration_ms": _duration_ms(start),
        }],
    }


def _extract_sql(text: str) -> str:
    """Pull a SQL statement out of an LLM reply, stripping markdown fences/prose.

    Intentionally simple: take the first ```sql ... ``` block if there is one,
    otherwise the whole reply. You may need to harden this for your prompts.
    """
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (fenced.group(1) if fenced else text).strip()


def _extract_json(text: str) -> dict[str, Any]:
    """Parse the first JSON object from an LLM reply."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidate = (fenced.group(1) if fenced else text).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if not match:
            return {"ok": False, "issue": "verifier returned no JSON"}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"ok": False, "issue": "verifier returned invalid JSON"}
    if isinstance(parsed, dict):
        return parsed
    return {"ok": False, "issue": "verifier JSON was not an object"}


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "ok"}
    return bool(value)


def generate_sql_node(state: AgentState) -> dict:
    """Worked example - the other LLM nodes follow this same shape.

    Build messages from the prompts, call the shared llm(), extract the SQL,
    and return only the state fields you changed. `iteration` is bumped here
    (and in revise) so route_after_verify can enforce AGENT_MAX_ITERATIONS.

    This node is wired and ready; fill in GENERATE_SQL_SYSTEM / GENERATE_SQL_USER
    in prompts.py to make it produce real queries.
    """
    start = time.perf_counter()
    response = llm().invoke([
        ("system", prompts.generate_sql_system()),
        ("user", prompts.generate_sql_user().format(
            schema=state.schema,
            question=state.question,
        ) + _cache_bust_suffix()),
    ])
    duration_ms = _duration_ms(start)
    sql = _extract_sql(response.content)
    iteration = state.iteration + 1
    return {
        "sql": sql,
        "iteration": iteration,
        "history": state.history + [{
            "node": "generate_sql",
            "iteration": iteration,
            "sql": sql,
            "duration_ms": duration_ms,
        }],
    }


def execute_node(state: AgentState) -> dict:
    """Provided. Runs the SQL and stores the result."""
    start = time.perf_counter()
    execution = execute_sql(state.db_id, state.sql)
    entry = {
        "node": "execute",
        "iteration": state.iteration,
        "sql": state.sql,
        "ok": execution.ok,
        "row_count": execution.row_count,
        "columns": execution.columns or [],
        "error": execution.error,
        "duration_ms": _duration_ms(start),
    }
    return {"execution": execution, "history": state.history + [entry]}


def deterministic_verify_node(state: AgentState) -> dict:
    """Run cheap, concrete SQL checks before spending an LLM verifier call."""
    start = time.perf_counter()
    issue = verify_deterministic(
        db_id=state.db_id,
        question=state.question,
        sql=state.sql,
        execution=state.execution,
    )
    entry = {
        "node": "deterministic_verify",
        "iteration": state.iteration,
        "ok": issue is None,
        "issue": issue or "",
        "duration_ms": _duration_ms(start),
    }
    if issue:
        return {
            "verify_ok": False,
            "verify_issue": issue,
            "deterministic_issue": issue,
            "history": state.history + [entry],
        }
    return {
        "deterministic_issue": "",
        "history": state.history + [entry],
    }


def verify_node(state: AgentState) -> dict:
    """Decide whether state.execution plausibly answers state.question.

    Follow the generate_sql_node pattern: build messages from the VERIFY_*
    prompts, call llm(), parse the reply. Ask the model for a small JSON object
    like {"ok": bool, "issue": str} and parse it defensively - the model may
    wrap it in prose or fences. state.execution.render() gives you a compact
    view of the rows or error to feed into the prompt.

    Return: {"verify_ok": <bool>, "verify_issue": <str>}.
    What counts as "not plausible" is yours to define - see the Phase 3 targets
    in the README.
    """
    execution_text = (
        state.execution.render()
        if state.execution is not None
        else "ERROR: SQL was not executed."
    )
    start = time.perf_counter()
    response = llm().invoke([
        ("system", prompts.verify_system()),
        ("user", prompts.verify_user().format(
            question=state.question,
            schema=state.schema,
            sql=state.sql,
            execution=execution_text,
        ) + _cache_bust_suffix()),
    ])
    duration_ms = _duration_ms(start)
    parsed = _extract_json(response.content)
    ok = _coerce_bool(parsed.get("ok", False))
    issue = str(parsed.get("issue") or "")
    if not ok and not issue:
        issue = "verifier marked the result implausible"
    entry = {
        "node": "verify",
        "iteration": state.iteration,
        "ok": ok,
        "issue": issue,
        "duration_ms": duration_ms,
    }
    return {"verify_ok": ok, "verify_issue": issue, "history": state.history + [entry]}


def revise_node(state: AgentState) -> dict:
    """Produce a revised SQL query given state.verify_issue and the prior attempt.

    Same shape as generate_sql_node, but the prompt should include the failing
    SQL, its execution result, and the verifier's complaint so the model can fix
    it. Bump the iteration counter the same way generate_sql_node does so the
    loop terminates.

    Return: {"sql": <str>, "iteration": state.iteration + 1, ...}.
    """
    execution_text = (
        state.execution.render()
        if state.execution is not None
        else "ERROR: SQL was not executed."
    )
    start = time.perf_counter()
    response = llm().invoke([
        ("system", prompts.revise_system()),
        ("user", prompts.revise_user().format(
            question=state.question,
            schema=state.schema,
            sql=state.sql,
            execution=execution_text,
            issue=state.verify_issue,
        ) + _cache_bust_suffix()),
    ])
    duration_ms = _duration_ms(start)
    sql = _extract_sql(response.content)
    iteration = state.iteration + 1
    return {
        "sql": sql,
        "iteration": iteration,
        "history": state.history + [{
            "node": "revise",
            "iteration": iteration,
            "sql": sql,
            "issue": state.verify_issue,
            "duration_ms": duration_ms,
        }],
    }


def route_after_verify(state: AgentState) -> str:
    """Conditional router: return "revise" to loop, "end" to terminate.

    Two reasons to end: the verifier was happy (state.verify_ok), or you've hit
    the iteration cap (state.iteration >= AGENT_MAX_ITERATIONS). Otherwise, revise.
    """
    if state.verify_ok or state.iteration >= get_max_iterations():
        return "end"
    return "revise"


def route_after_deterministic_verify(state: AgentState) -> str:
    if state.deterministic_issue:
        if state.iteration >= get_max_iterations():
            return "end"
        return "revise"
    if get_verify_mode() == VERIFY_MODE_FAST:
        return "end"
    return "verify"


def route_after_execute(state: AgentState) -> str:
    """Route baseline mode around deterministic checks."""
    if get_verify_mode() == VERIFY_MODE_LLM_ONLY:
        return "verify"
    return "deterministic_verify"


# ---- Graph wiring -----------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("attach_schema", _attach_schema)
    g.add_node("generate_sql", generate_sql_node)
    g.add_node("execute", execute_node)
    g.add_node("deterministic_verify", deterministic_verify_node)
    g.add_node("verify", verify_node)
    g.add_node("revise", revise_node)

    g.add_edge(START, "attach_schema")
    g.add_edge("attach_schema", "generate_sql")
    g.add_edge("generate_sql", "execute")
    g.add_conditional_edges(
        "execute",
        route_after_execute,
        {"deterministic_verify": "deterministic_verify", "verify": "verify"},
    )
    g.add_conditional_edges(
        "deterministic_verify",
        route_after_deterministic_verify,
        {"verify": "verify", "revise": "revise", "end": END},
    )
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"revise": "revise", "end": END},
    )
    g.add_edge("revise", "execute")
    return g.compile()


graph = build_graph()

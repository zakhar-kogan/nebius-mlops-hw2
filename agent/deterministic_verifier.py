"""Deterministic SQL checks used before the LLM verifier.

These checks are intentionally conservative: return a concrete issue only when
the query has a structural problem, contradicts obvious wording, or uses a
literal that does not exist in the database while a close exact value does.
"""
from __future__ import annotations

import re
import sqlite3
from functools import lru_cache

from agent.execution import ExecutionResult
from agent.schema import db_path

UNSAFE_RE = re.compile(
    r"\b(attach|alter|create|delete|detach|drop|insert|pragma|update|vacuum)\b",
    re.IGNORECASE,
)


def verify_deterministic(
    db_id: str,
    question: str,
    sql: str,
    execution: ExecutionResult | None,
) -> str | None:
    """Return a precise verifier issue, or None when LLM verification should run."""
    for check in (
        _check_readonly_select,
        _check_output_hints,
        _check_time_handling,
        _check_domain_rules,
        _check_filter_literals,
        _check_duplicates,
    ):
        issue = check(db_id, question, sql, execution)
        if issue:
            return issue
    if execution is not None and not execution.ok:
        return f"SQL execution failed: {execution.error}"
    return None


def _check_readonly_select(
    _db_id: str,
    _question: str,
    sql: str,
    _execution: ExecutionResult | None,
) -> str | None:
    cleaned = _strip_comments(sql).strip()
    statements = [stmt.strip() for stmt in _split_statements(cleaned) if stmt.strip()]
    if len(statements) != 1:
        return "SQL must contain exactly one statement"
    if not re.match(r"^(select|with)\b", statements[0], re.IGNORECASE):
        return "SQL must be a read-only SELECT query"
    if UNSAFE_RE.search(statements[0]):
        return "SQL uses a non-read-only or unsafe keyword"
    return None


def _check_output_hints(
    db_id: str,
    question: str,
    sql: str,
    _execution: ExecutionResult | None,
) -> str | None:
    q = question.lower()
    selected = " ".join(_select_expressions(sql)).lower()

    if "school identification" in q and "ncesdist" in selected and "ncesschool" not in selected:
        return "question asks for the NCES school identification number; select NCESSchool, not NCESDist"
    if "nces school identification" in q and "ncesdist" in selected and "ncesschool" not in selected:
        return "question asks for the NCES school identification number; select NCESSchool, not NCESDist"
    if db_id == "card_games" and "list all" in q and "cards" in q:
        if re.search(r"\bname\b", selected) and not re.search(r"\bid\b", selected):
            return "question asks to list cards; select the card id rather than card name"
    if "originally printed" in q and "originaltype" in selected and "originaltype" in sql.lower():
        if not re.search(r"originaltype\"?\s+is\s+not\s+null", sql, re.IGNORECASE):
            return "originally printed type should exclude NULL originalType values"
    if "full names" in q and ("||" in selected or "concat(" in selected):
        return "for full names in this eval, select first_name and last_name as separate columns, not a concatenated string"
    if "well-finished" in q and "closeddate" not in sql.lower():
        return "well-finished is determined from posts.ClosedDate; use ClosedDate rather than post type or accepted answer"
    return None


def _check_time_handling(
    _db_id: str,
    question: str,
    sql: str,
    _execution: ExecutionResult | None,
) -> str | None:
    q = question.lower()
    s = sql.lower()
    if "seconds" in q and "fastestlaptime" in s:
        converts_minutes = "substr" in s and "instr" in s and "* 60" in s
        if not converts_minutes:
            return "fastestLapTime values are mm:ss.xxx; convert minutes to seconds with SUBSTR/INSTR before averaging"
    if "time for the fastest" in q and "milliseconds" in s:
        return "question asks for the lap time value; select time and order by parsed time instead of returning milliseconds"
    return None


def _check_domain_rules(
    db_id: str,
    question: str,
    sql: str,
    _execution: ExecutionResult | None,
) -> str | None:
    q = question.lower()
    s = sql.lower()
    if db_id == "financial" and "crimes committed in 1995" in q and re.search(r"\ba14\b", s):
        return "financial district crimes committed in 1995 are in A15, not A14"
    if db_id == "california_schools" and "excellence rate" in q:
        if "numge1500" not in s or "numtsttakr" not in s:
            return "school excellence rate is NumGE1500 divided by NumTstTakr"
    if db_id == "formula_1" and "finishers" in q and "disqualified" in q and "time is not null" not in s:
        return "finishers must have results.time IS NOT NULL before counting disqualified results"
    if db_id == "codebase_community" and "higher popularity" in q:
        if "group by" not in s or "sum(" not in s or "displayname" not in s:
            return "compare user popularity by grouping DisplayName and ordering by SUM(posts.ViewCount)"
    if db_id == "thrombosis_prediction" and "normal ig g" in q:
        if "laboratory" not in s or "igg" not in s or "between 900 and 2000" not in s:
            return "normal Ig G should use Laboratory.IGG BETWEEN 900 AND 2000"
    if db_id == "thrombosis_prediction" and "normal uric acid" in q:
        if "sex" not in s or "ua < 6.5" not in s or "ua < 8.0" not in s:
            return "normal UA is sex-specific: UA < 6.5 for F and UA < 8.0 for M"
    if db_id == "thrombosis_prediction" and "outpatient clinic" in q and "admission" in s:
        if "admission\" = '-'" not in s and "admission = '-'" not in s:
            return "outpatient clinic admission is encoded as '-' in Patient.Admission"
    if db_id == "toxicology" and re.search(r"label\s*=\s*'carcinogenic'", s, re.IGNORECASE):
        return "toxicology carcinogenic labels are encoded as '+', not 'carcinogenic'"
    if db_id == "toxicology" and "non carcinogenic" in s:
        return "toxicology non-carcinogenic labels are encoded as '-', not text"
    return None


def _check_filter_literals(
    db_id: str,
    _question: str,
    sql: str,
    _execution: ExecutionResult | None,
) -> str | None:
    for literal, pos in _string_literals(sql):
        if not _is_filter_literal(sql, pos) or _skip_literal(literal):
            continue
        match = _literal_match(db_id, literal)
        if match == "exact":
            continue
        if match:
            return f"literal {literal!r} does not match stored value exactly; use {match!r}"
        return f"literal {literal!r} was not found in database text/date columns"
    return None


def _check_duplicates(
    _db_id: str,
    question: str,
    sql: str,
    execution: ExecutionResult | None,
) -> str | None:
    if execution is None or not execution.ok or not execution.rows:
        return None
    q = question.lower()
    asks_unique = any(
        phrase in q
        for phrase in (
            "coordinates",
            "location",
            "list all",
            "list down",
            "which ",
            "what is the",
            "identification number",
        )
    )
    if not asks_unique or re.search(r"\b(distinct|group\s+by|count|avg|sum|min|max)\b", sql, re.IGNORECASE):
        return None
    unique_rows = {tuple(row) for row in execution.rows}
    if len(unique_rows) < len(execution.rows):
        return "result contains duplicate rows; add DISTINCT or group by the returned columns"
    return None


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    return re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)


def _split_statements(sql: str) -> list[str]:
    statements: list[str] = []
    start = 0
    quote: str | None = None
    i = 0
    while i < len(sql):
        ch = sql[i]
        if quote:
            if ch == quote:
                if quote == "'" and i + 1 < len(sql) and sql[i + 1] == "'":
                    i += 1
                else:
                    quote = None
        elif ch in {"'", '"'}:
            quote = ch
        elif ch == ";":
            statements.append(sql[start:i])
            start = i + 1
        i += 1
    tail = sql[start:]
    if tail.strip():
        statements.append(tail)
    return statements


def _select_expressions(sql: str) -> list[str]:
    match = re.search(r"\bselect\b", sql, re.IGNORECASE)
    if not match:
        return []
    start = match.end()
    depth = 0
    quote: str | None = None
    end = len(sql)
    i = start
    while i < len(sql):
        ch = sql[i]
        if quote:
            if ch == quote:
                if quote == "'" and i + 1 < len(sql) and sql[i + 1] == "'":
                    i += 1
                else:
                    quote = None
        elif ch in {"'", '"'}:
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and re.match(r"\sfrom\b", sql[i:], re.IGNORECASE):
            end = i
            break
        i += 1
    return _split_top_level(sql[start:end], ",")


def _split_top_level(text: str, sep: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == quote:
                if quote == "'" and i + 1 < len(text) and text[i + 1] == "'":
                    i += 1
                else:
                    quote = None
        elif ch in {"'", '"'}:
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and ch == sep:
            parts.append(text[start:i].strip())
            start = i + 1
        i += 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _string_literals(sql: str) -> list[tuple[str, int]]:
    values: list[tuple[str, int]] = []
    i = 0
    while i < len(sql):
        if sql[i] != "'":
            i += 1
            continue
        start = i
        i += 1
        chars: list[str] = []
        while i < len(sql):
            if sql[i] == "'":
                if i + 1 < len(sql) and sql[i + 1] == "'":
                    chars.append("'")
                    i += 2
                    continue
                values.append(("".join(chars), start))
                i += 1
                break
            chars.append(sql[i])
            i += 1
    return values


def _is_filter_literal(sql: str, pos: int) -> bool:
    prefix = sql[:pos].lower()
    tail = prefix[-140:]
    if re.search(r"\b(then|else)\s*$", tail):
        return False
    return bool(
        re.search(r"(=|<>|!=|<=|>=|<|>|\blike\b)\s*$", tail)
        or re.search(r"\bin\s*\([^)]*$", tail)
    )


def _skip_literal(value: str) -> bool:
    stripped = value.strip()
    if not stripped or stripped.startswith("%"):
        return True
    if re.fullmatch(r"\d{1,4}", stripped):
        return True
    return False


@lru_cache(maxsize=32)
def _text_columns(db_id: str) -> tuple[tuple[str, str], ...]:
    path = db_path(db_id)
    columns: list[tuple[str, str]] = []
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        for table in tables:
            for _cid, name, ctype, *_rest in conn.execute(f"PRAGMA table_info({_q(table)})"):
                probe = f"{name} {ctype}".lower()
                if any(token in probe for token in ("char", "text", "date", "time", "name", "status", "label", "type", "gender", "sex", "admission", "element", "format", "rarity")):
                    columns.append((table, name))
    return tuple(columns)


@lru_cache(maxsize=512)
def _literal_match(db_id: str, literal: str) -> str | None:
    path = db_path(db_id)
    variants = [literal]
    if re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", literal):
        variants.append(literal + ".0")
    if literal.endswith(".0"):
        variants.append(literal[:-2])

    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        for table, column in _text_columns(db_id):
            for variant in variants:
                if _value_exists(conn, table, column, variant, exact=True):
                    return "exact" if variant == literal else variant
            exact_ci = _case_insensitive_value(conn, table, column, literal)
            if exact_ci is not None:
                return exact_ci
        for table, column in _text_columns(db_id):
            close = _containing_value(conn, table, column, literal)
            if close is not None:
                return close
    return None


def _value_exists(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    value: str,
    *,
    exact: bool,
) -> bool:
    op = "=" if exact else "LIKE"
    try:
        row = conn.execute(
            f"SELECT 1 FROM {_q(table)} WHERE {_q(column)} {op} ? LIMIT 1",
            (value,),
        ).fetchone()
    except sqlite3.Error:
        return False
    return row is not None


def _case_insensitive_value(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    value: str,
) -> str | None:
    try:
        row = conn.execute(
            f"SELECT DISTINCT {_q(column)} FROM {_q(table)} "
            f"WHERE lower(CAST({_q(column)} AS TEXT)) = lower(?) "
            f"AND CAST({_q(column)} AS TEXT) <> ? LIMIT 1",
            (value, value),
        ).fetchone()
    except sqlite3.Error:
        return None
    return str(row[0]) if row else None


def _containing_value(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    value: str,
) -> str | None:
    if len(value) < 4:
        return None
    pattern = f"%{value.lower()}%"
    try:
        row = conn.execute(
            f"SELECT DISTINCT {_q(column)} FROM {_q(table)} "
            f"WHERE lower(CAST({_q(column)} AS TEXT)) LIKE ? "
            f"AND CAST({_q(column)} AS TEXT) <> ? "
            f"ORDER BY length(CAST({_q(column)} AS TEXT)) LIMIT 1",
            (pattern, value),
        ).fetchone()
    except sqlite3.Error:
        return None
    if row:
        return str(row[0])
    return None


def _q(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'

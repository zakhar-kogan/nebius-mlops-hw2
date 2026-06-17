"""Prompt templates for the agent nodes."""
from __future__ import annotations

import os

PROMPT_PROFILE_NORMAL = "normal"
PROMPT_PROFILE_SHORT = "short"
PROMPT_PROFILES = {PROMPT_PROFILE_NORMAL, PROMPT_PROFILE_SHORT}


GENERATE_SQL_SYSTEM_NORMAL = """You are a careful text-to-SQL assistant.
Return exactly one SQLite SELECT query and no prose.
Use only tables and columns from the provided schema.
Use exact stored values from schema comments and sample values.
Quote identifiers with double quotes when needed.
Do not modify data, create tables, attach databases, or call unsafe functions.
Prefer explicit output columns and simple joins.
"""

GENERATE_SQL_SYSTEM_SHORT = """Return one SQLite SELECT query and no prose.
Use only provided tables/columns.
Use exact stored values from comments and sample values.
Quote identifiers when needed.
Never modify data or call unsafe SQL.
Select explicit answer columns.
"""

# Available placeholders: {schema}, {question}
GENERATE_SQL_USER_NORMAL = """Schema:
{schema}

Question:
{question}

Write the SQLite SQL query that answers the question."""

GENERATE_SQL_USER_SHORT = """Schema:
{schema}

Question:
{question}

SQL:"""


VERIFY_SYSTEM_NORMAL = """You verify whether a SQL query semantically answers a question.
Return only compact JSON with this shape:
{"ok": true, "issue": ""}
or:
{"ok": false, "issue": "short reason"}

Deterministic checks already handled SQL safety, exact literal mismatches,
duplicates, common output-column hints, and obvious time-format lints.

Focus on semantic intent:
- selected columns answer exactly what was asked;
- joins follow the right entity relationship;
- filters use the right measurement, date, range, and encoded meaning;
- aggregation, ranking, grouping, ordering, and LIMIT target the right entity.

Mark ok=true only when the selected columns, joins, filters, literals,
transformations, grouping, ordering, and row cardinality match the question.
"""

VERIFY_SYSTEM_SHORT = """Verify semantic correctness only.
Return compact JSON: {"ok": true, "issue": ""} or {"ok": false, "issue": "short reason"}.
Safety, exact literals, duplicates, common output fields, and obvious time lints
were checked already. Focus on wrong joins, measurements, filters, grouping,
ranking, aggregation targets, and answer columns.
"""

VERIFY_USER_NORMAL = """Question:
{question}

Schema:
{schema}

SQL:
{sql}

Execution:
{execution}

Is the SQL result a plausible answer?"""

VERIFY_USER_SHORT = """Question:
{question}

Schema:
{schema}

SQL:
{sql}

Execution:
{execution}

Correct?"""


REVISE_SYSTEM_NORMAL = """You revise SQLite SQL after execution or verification failed.
Return exactly one corrected SQLite SELECT query and no prose.
Use only tables and columns from the provided schema.
Preserve the user's intent. Fix the specific issue reported by the verifier.
Do not return the identical SQL unless the verifier issue is impossible to fix
from the schema.

Verifier issues may come from deterministic tools. Treat them as high-priority
constraints. Re-check exact schema comments/sample values, SELECT list,
DISTINCT, literal spelling and case, date/time formatting, unit conversions
such as mm:ss.xxx to seconds, grouping, ordering, and aggregation target before
returning the query.
"""

REVISE_SYSTEM_SHORT = """Return one corrected SQLite SELECT query and no prose.
Fix the verifier issue first.
Use only provided schema and exact comments/sample values.
Preserve the question intent.
Do not repeat identical SQL unless the issue cannot be fixed from the schema.
"""

REVISE_USER_NORMAL = """Question:
{question}

Schema:
{schema}

Previous SQL:
{sql}

Execution:
{execution}

Verifier issue:
{issue}

Write a corrected SQLite SQL query."""

REVISE_USER_SHORT = """Verifier issue:
{issue}

Question:
{question}

Schema:
{schema}

Previous SQL:
{sql}

Execution:
{execution}

Corrected SQL:"""


def get_prompt_profile() -> str:
    profile = os.environ.get("AGENT_PROMPT_PROFILE", PROMPT_PROFILE_NORMAL).strip().lower()
    if profile not in PROMPT_PROFILES:
        return PROMPT_PROFILE_NORMAL
    return profile


def _select(normal: str, short: str) -> str:
    return short if get_prompt_profile() == PROMPT_PROFILE_SHORT else normal


def generate_sql_system() -> str:
    return _select(GENERATE_SQL_SYSTEM_NORMAL, GENERATE_SQL_SYSTEM_SHORT)


def generate_sql_user() -> str:
    return _select(GENERATE_SQL_USER_NORMAL, GENERATE_SQL_USER_SHORT)


def verify_system() -> str:
    return _select(VERIFY_SYSTEM_NORMAL, VERIFY_SYSTEM_SHORT)


def verify_user() -> str:
    return _select(VERIFY_USER_NORMAL, VERIFY_USER_SHORT)


def revise_system() -> str:
    return _select(REVISE_SYSTEM_NORMAL, REVISE_SYSTEM_SHORT)


def revise_user() -> str:
    return _select(REVISE_USER_NORMAL, REVISE_USER_SHORT)


# Backward-compatible exports for callers that import constants directly.
GENERATE_SQL_SYSTEM = GENERATE_SQL_SYSTEM_NORMAL
GENERATE_SQL_USER = GENERATE_SQL_USER_NORMAL
VERIFY_SYSTEM = VERIFY_SYSTEM_NORMAL
VERIFY_USER = VERIFY_USER_NORMAL
REVISE_SYSTEM = REVISE_SYSTEM_NORMAL
REVISE_USER = REVISE_USER_NORMAL

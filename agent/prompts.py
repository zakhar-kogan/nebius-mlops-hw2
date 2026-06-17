"""Prompt templates for the agent nodes.

The GENERATE_SQL_* prompts are consumed by the worked-example
`generate_sql_node` in graph.py via `.format(schema=..., question=...)`, so
keep those placeholders intact. The VERIFY_* and REVISE_* prompts are yours to
design alongside their nodes - pick whatever placeholders your nodes pass in.

Filling these in is part of Phase 3.
"""

GENERATE_SQL_SYSTEM = """You are a careful text-to-SQL assistant.
Return exactly one SQLite SELECT query and no prose.
Use only tables and columns from the provided schema.
Quote identifiers with double quotes when needed.
Do not modify data, create tables, attach databases, or call unsafe functions.
Prefer simple joins and explicit column names.
"""

# Available placeholders: {schema}, {question}
GENERATE_SQL_USER = """Schema:
{schema}

Question:
{question}

Write the SQLite SQL query that answers the question."""


VERIFY_SYSTEM = """You verify whether a SQL query plausibly answers a question.
Return only compact JSON with this shape:
{"ok": true, "issue": ""}
or:
{"ok": false, "issue": "short reason"}

Mark ok=false when the SQL errored, returned zero rows despite a question that
expects concrete rows, uses columns unrelated to the question, or clearly
answers a different question. Mark ok=true for plausible non-empty results and
for valid zero-row results when the question could naturally have no matches.
"""

VERIFY_USER = """Question:
{question}

Schema:
{schema}

SQL:
{sql}

Execution:
{execution}

Is the SQL result a plausible answer?"""


REVISE_SYSTEM = """You revise SQLite SQL after execution or verification failed.
Return exactly one corrected SQLite SELECT query and no prose.
Use only tables and columns from the provided schema.
Preserve the user's intent. Fix the specific issue reported by the verifier.
"""

REVISE_USER = """Question:
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

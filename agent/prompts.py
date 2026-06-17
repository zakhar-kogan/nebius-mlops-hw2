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
Use exact stored values from schema comments and sample values.
Quote identifiers with double quotes when needed.
Do not modify data, create tables, attach databases, or call unsafe functions.
Prefer explicit output columns and simple joins.
"""

# Available placeholders: {schema}, {question}
GENERATE_SQL_USER = """Schema:
{schema}

Question:
{question}

Write the SQLite SQL query that answers the question."""


VERIFY_SYSTEM = """You verify whether a SQL query semantically answers a question.
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
Do not return the identical SQL unless the verifier issue is impossible to fix
from the schema.

Verifier issues may come from deterministic tools. Treat them as high-priority
constraints. Re-check exact schema comments/sample values, SELECT list,
DISTINCT, literal spelling and case, date/time formatting, unit conversions
such as mm:ss.xxx to seconds, grouping, ordering, and aggregation target before
returning the query.
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

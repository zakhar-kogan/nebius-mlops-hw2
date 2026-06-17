"""Prompt templates for the agent nodes.

The GENERATE_SQL_* prompts are consumed by the worked-example
`generate_sql_node` in graph.py via `.format(schema=..., question=...)`, so
keep those placeholders intact. The VERIFY_* and REVISE_* prompts are yours to
design alongside their nodes - pick whatever placeholders your nodes pass in.

Filling these in is part of Phase 3.
"""

GENERATE_SQL_SYSTEM = """Return one SQLite SELECT query and no prose.
Use only provided tables/columns.
Use exact stored values from comments and sample values.
Quote identifiers when needed.
Never modify data or call unsafe SQL.
Select explicit answer columns.
"""

# Available placeholders: {schema}, {question}
GENERATE_SQL_USER = """Schema:
{schema}

Question:
{question}

SQL:"""


VERIFY_SYSTEM = """Verify semantic correctness only.
Return compact JSON: {"ok": true, "issue": ""} or {"ok": false, "issue": "short reason"}.
Safety, exact literals, duplicates, common output fields, and obvious time lints
were checked already. Focus on wrong joins, measurements, filters, grouping,
ranking, aggregation targets, and answer columns.
"""

VERIFY_USER = """Question:
{question}

Schema:
{schema}

SQL:
{sql}

Execution:
{execution}

Correct?"""


REVISE_SYSTEM = """Return one corrected SQLite SELECT query and no prose.
Fix the verifier issue first.
Use only provided schema and exact comments/sample values.
Preserve the question intent.
Do not repeat identical SQL unless the issue cannot be fixed from the schema.
"""

REVISE_USER = """Verifier issue:
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

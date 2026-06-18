"""Schema-rendering helper.

Loads the schema directly from sqlite and renders quoted CREATE TABLE
text suitable for prompt context. Identifiers are always double-quoted
so reserved-word table/column names (e.g. `order`) don't break either
the PRAGMA introspection here or the SQL the model emits later.

The BIRD release also ships compact column descriptions. We include them
inline because many failures come from encoded values or ambiguous columns
that are not recoverable from SQLite types alone.
"""
from __future__ import annotations

import csv
import os
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "data" / "bird"
DESCRIPTION_DIR = DB_DIR / "dev_20240627" / "dev_databases"
COMMENT_MAX_LEN = 100
SAMPLE_VALUE_MAX_LEN = 32
SAMPLE_VALUE_LIMIT = 3
SAMPLE_DISTINCT_LIMIT = 20
SCHEMA_PROFILE_COMPACT = "compact"
SCHEMA_PROFILE_BUDGET = "budget"
SCHEMA_PROFILE_AGGRESSIVE = "aggressive"
SCHEMA_PROFILES = {
    SCHEMA_PROFILE_COMPACT,
    SCHEMA_PROFILE_BUDGET,
    SCHEMA_PROFILE_AGGRESSIVE,
}
CATEGORICAL_HINTS = (
    "admission",
    "answer",
    "brand",
    "category",
    "class",
    "code",
    "color",
    "department",
    "element",
    "format",
    "gender",
    "grade",
    "label",
    "level",
    "method",
    "rank",
    "rarity",
    "result",
    "schooltype",
    "sex",
    "status",
    "type",
)
IMPORTANT_DESCRIPTION_HINTS = (
    "encoded",
    "format",
    "indicates",
    "meaning",
    "percentage",
    "second",
    "unit",
    "value",
    "values",
)


@dataclass(frozen=True)
class SchemaPolicy:
    comment_max_len: int
    sample_value_max_len: int
    sample_value_limit: int
    sample_distinct_limit: int
    include_samples: bool
    only_categorical_samples: bool
    aggressive_descriptions: bool
    max_chars: int | None


def db_path(db_id: str) -> Path:
    return DB_DIR / f"{db_id}.sqlite"


def _q(ident: str) -> str:
    """Double-quote a SQL identifier, escaping any embedded quotes."""
    return '"' + ident.replace('"', '""') + '"'


def _comment(text: str, max_len: int = COMMENT_MAX_LEN) -> str:
    compact = " ".join(text.replace("*/", "").split())
    if len(compact) > max_len:
        return compact[: max_len - 3].rstrip() + "..."
    return compact


def get_schema_profile() -> str:
    profile = os.environ.get("AGENT_SCHEMA_PROFILE", SCHEMA_PROFILE_COMPACT).strip().lower()
    if profile not in SCHEMA_PROFILES:
        return SCHEMA_PROFILE_COMPACT
    return profile


def _schema_complexity(conn: sqlite3.Connection) -> tuple[int, int]:
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
    ]
    columns = 0
    for table in tables:
        columns += len(conn.execute(f"PRAGMA table_info({_q(table)})").fetchall())
    return len(tables), columns


def _schema_policy(conn: sqlite3.Connection, profile: str) -> SchemaPolicy:
    tables, columns = _schema_complexity(conn)
    if profile == SCHEMA_PROFILE_COMPACT:
        return SchemaPolicy(
            comment_max_len=COMMENT_MAX_LEN,
            sample_value_max_len=SAMPLE_VALUE_MAX_LEN,
            sample_value_limit=SAMPLE_VALUE_LIMIT,
            sample_distinct_limit=SAMPLE_DISTINCT_LIMIT,
            include_samples=True,
            only_categorical_samples=True,
            aggressive_descriptions=False,
            max_chars=None,
        )

    is_huge = columns >= 100 or tables >= 20
    is_medium = columns >= 70 or tables >= 12
    if profile == SCHEMA_PROFILE_BUDGET:
        return SchemaPolicy(
            comment_max_len=60 if is_huge else 80 if is_medium else COMMENT_MAX_LEN,
            sample_value_max_len=24 if is_medium or is_huge else SAMPLE_VALUE_MAX_LEN,
            sample_value_limit=1 if is_huge else 2 if is_medium else SAMPLE_VALUE_LIMIT,
            sample_distinct_limit=SAMPLE_DISTINCT_LIMIT,
            include_samples=True,
            only_categorical_samples=True,
            aggressive_descriptions=False,
            max_chars=9000 if is_huge else 8000 if is_medium else None,
        )

    return SchemaPolicy(
        comment_max_len=40 if is_huge else 60 if is_medium else 70,
        sample_value_max_len=20,
        sample_value_limit=0 if is_huge else 1,
        sample_distinct_limit=12,
        include_samples=not is_huge,
        only_categorical_samples=True,
        aggressive_descriptions=True,
        max_chars=7000 if is_huge else 6000 if is_medium else None,
    )


def _schema_cache_dir() -> Path | None:
    raw = os.environ.get("AGENT_SCHEMA_CACHE_DIR", "").strip()
    return Path(raw) if raw else None


def _schema_cache_path(cache_dir: Path, db_id: str, profile: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in db_id)
    if profile != SCHEMA_PROFILE_COMPACT:
        return cache_dir / f"{safe}.{profile}.schema.txt"
    return cache_dir / f"{safe}.schema.txt"


@lru_cache(maxsize=128)
def _column_descriptions(db_id: str, table: str) -> dict[str, str]:
    path = DESCRIPTION_DIR / db_id / "database_description" / f"{table}.csv"
    if not path.exists():
        return {}

    descriptions: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = (row.get("original_column_name") or "").strip()
            if not name:
                continue
            bits = [
                (row.get("column_name") or "").strip(),
                (row.get("column_description") or "").strip(),
                (row.get("value_description") or "").strip(),
            ]
            text = "; ".join(bit for bit in bits if bit)
            if text:
                descriptions[name] = " ".join(text.replace("*/", "").split())
    return descriptions


def _is_text_like(name: str, ctype: str) -> bool:
    probe = f"{name} {ctype}".lower()
    return any(
        token in probe
        for token in (
            "char",
            "clob",
            "text",
            "date",
            "time",
            "name",
            "status",
            "label",
            "type",
            "gender",
            "sex",
            "admission",
            "element",
            "department",
            "format",
            "rarity",
        )
    )


def _is_categorical_candidate(column: str, ctype: str, description: str) -> bool:
    probe = f"{column} {ctype} {description}".lower()
    return any(token in probe for token in CATEGORICAL_HINTS)


def _sample_values(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    ctype: str,
    policy: SchemaPolicy,
) -> list[str]:
    if not policy.include_samples or policy.sample_value_limit <= 0:
        return []
    if not _is_text_like(column, ctype):
        return []

    try:
        count = conn.execute(
            f"SELECT COUNT(DISTINCT {_q(column)}) FROM {_q(table)} "
            f"WHERE {_q(column)} IS NOT NULL AND CAST({_q(column)} AS TEXT) <> ''"
        ).fetchone()[0]
    except sqlite3.Error:
        return []
    if not count or count > policy.sample_distinct_limit:
        return []

    try:
        rows = conn.execute(
            f"SELECT DISTINCT {_q(column)} FROM {_q(table)} "
            f"WHERE {_q(column)} IS NOT NULL AND CAST({_q(column)} AS TEXT) <> '' "
            f"ORDER BY {_q(column)} LIMIT {policy.sample_value_limit}"
        ).fetchall()
    except sqlite3.Error:
        return []
    return [_comment(str(row[0]), policy.sample_value_max_len) for row in rows]


def _include_description(
    column: str,
    ctype: str,
    description: str,
    policy: SchemaPolicy,
) -> bool:
    if not description:
        return False
    if not policy.aggressive_descriptions:
        return True
    probe = f"{column} {ctype} {description}".lower()
    return (
        _is_categorical_candidate(column, ctype, description)
        or any(hint in probe for hint in IMPORTANT_DESCRIPTION_HINTS)
    )


def render_schema(db_id: str) -> str:
    cache_dir = _schema_cache_dir()
    profile = get_schema_profile()
    cache_key = str(cache_dir.resolve()) if cache_dir is not None else ""
    return _render_schema_cached(db_id, cache_key, profile)


@lru_cache(maxsize=64)
def _render_schema_cached(db_id: str, cache_key: str, profile: str) -> str:
    if cache_key:
        cache_dir = Path(cache_key)
        cache_path = _schema_cache_path(cache_dir, db_id, profile)
        if cache_path.exists():
            return cache_path.read_text()

        rendered = _render_schema_live(db_id, profile)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(rendered)
        return rendered

    return _render_schema_live(db_id, profile)


def _render_schema_live(db_id: str, profile: str) -> str:
    path = db_path(db_id)
    if not path.exists():
        raise FileNotFoundError(f"DB {db_id} not found at {path}. Did you run scripts/load_data.py?")

    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        policy = _schema_policy(conn, profile)
        rendered = _render_schema_with_policy(db_id, conn, policy)
        if policy.max_chars is None or len(rendered) <= policy.max_chars:
            return rendered
        no_samples = SchemaPolicy(
            **{**policy.__dict__, "include_samples": False}
        )
        rendered = _render_schema_with_policy(db_id, conn, no_samples)
        if len(rendered) <= policy.max_chars:
            return rendered
        no_annotations = SchemaPolicy(
            **{
                **policy.__dict__,
                "include_samples": False,
                "aggressive_descriptions": True,
                "comment_max_len": min(policy.comment_max_len, 32),
            }
        )
        rendered = _render_schema_with_policy(db_id, conn, no_annotations)
        if len(rendered) <= policy.max_chars:
            return rendered
        return _trim_schema_blocks(rendered, policy.max_chars)


def _render_schema_with_policy(
    db_id: str,
    conn: sqlite3.Connection,
    policy: SchemaPolicy,
) -> str:
    parts: list[str] = [f"-- Database: {db_id}"]
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
    ]
    for t in tables:
        parts.append(f"\nCREATE TABLE {_q(t)} (")
        col_lines: list[str] = []
        descriptions = _column_descriptions(db_id, t)
        for _cid, name, ctype, notnull, _dflt, pk in conn.execute(f"PRAGMA table_info({_q(t)})"):
            line = f"  {_q(name)} {ctype}"
            if pk:
                line += " PRIMARY KEY"
            if notnull and not pk:
                line += " NOT NULL"
            annotations: list[str] = []
            description = descriptions.get(name, "")
            if _include_description(name, ctype, description, policy):
                annotations.append(_comment(description, policy.comment_max_len))
            samples = (
                _sample_values(conn, t, name, ctype, policy)
                if (
                    not policy.only_categorical_samples
                    or _is_categorical_candidate(name, ctype, description)
                )
                else []
            )
            if samples:
                annotations.append("sample values: " + ", ".join(repr(v) for v in samples))
            if annotations:
                line += f" /* {' | '.join(annotations)} */"
            col_lines.append(line)
        for fk in conn.execute(f"PRAGMA foreign_key_list({_q(t)})"):
            # (id, seq, ref_table, from, to, on_update, on_delete, match)
            if not fk[2] or not fk[3] or not fk[4]:
                continue
            col_lines.append(
                f"  FOREIGN KEY ({_q(fk[3])}) REFERENCES {_q(fk[2])}({_q(fk[4])})"
            )
        parts.append(",\n".join(col_lines))
        parts.append(");")
    return "\n".join(parts)


def _trim_schema_blocks(rendered: str, max_chars: int) -> str:
    blocks = rendered.split("\n\nCREATE TABLE ")
    kept = [blocks[0]]
    current = len(kept[0])
    for block in blocks[1:]:
        candidate = "\n\nCREATE TABLE " + block
        if current + len(candidate) > max_chars:
            continue
        kept.append(candidate)
        current += len(candidate)
    omitted = len(blocks) - len(kept)
    if omitted > 0:
        kept.append(f"\n\n-- {omitted} table blocks omitted by schema budget")
    return "".join(kept)


def prewarm_schemas(db_ids: list[str] | None = None) -> list[str]:
    warmed = db_ids or available_dbs()
    for db_id in warmed:
        render_schema(db_id)
    return warmed


def clear_schema_caches() -> None:
    _column_descriptions.cache_clear()
    _render_schema_cached.cache_clear()


def available_dbs() -> list[str]:
    if not DB_DIR.exists():
        return []
    return sorted(p.stem for p in DB_DIR.glob("*.sqlite"))

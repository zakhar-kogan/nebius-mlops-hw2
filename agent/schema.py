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
import sqlite3
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "data" / "bird"
DESCRIPTION_DIR = DB_DIR / "dev_20240627" / "dev_databases"


def db_path(db_id: str) -> Path:
    return DB_DIR / f"{db_id}.sqlite"


def _q(ident: str) -> str:
    """Double-quote a SQL identifier, escaping any embedded quotes."""
    return '"' + ident.replace('"', '""') + '"'


def _comment(text: str, max_len: int = 220) -> str:
    compact = " ".join(text.replace("*/", "").split())
    if len(compact) > max_len:
        return compact[: max_len - 3].rstrip() + "..."
    return compact


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
                descriptions[name] = _comment(text)
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


def _sample_values(conn: sqlite3.Connection, table: str, column: str, ctype: str) -> list[str]:
    if not _is_text_like(column, ctype):
        return []

    try:
        count = conn.execute(
            f"SELECT COUNT(DISTINCT {_q(column)}) FROM {_q(table)} "
            f"WHERE {_q(column)} IS NOT NULL AND CAST({_q(column)} AS TEXT) <> ''"
        ).fetchone()[0]
    except sqlite3.Error:
        return []
    if not count or count > 20:
        return []

    try:
        rows = conn.execute(
            f"SELECT DISTINCT {_q(column)} FROM {_q(table)} "
            f"WHERE {_q(column)} IS NOT NULL AND CAST({_q(column)} AS TEXT) <> '' "
            f"ORDER BY {_q(column)} LIMIT 8"
        ).fetchall()
    except sqlite3.Error:
        return []
    return [_comment(str(row[0]), 40) for row in rows]


@lru_cache(maxsize=32)
def render_schema(db_id: str) -> str:
    path = db_path(db_id)
    if not path.exists():
        raise FileNotFoundError(f"DB {db_id} not found at {path}. Did you run scripts/load_data.py?")

    parts: list[str] = [f"-- Database: {db_id}"]
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
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
                if descriptions.get(name):
                    annotations.append(descriptions[name])
                samples = _sample_values(conn, t, name, ctype)
                if samples:
                    annotations.append("sample values: " + ", ".join(repr(v) for v in samples))
                if annotations:
                    line += f" /* {' | '.join(annotations)} */"
                col_lines.append(line)
            for fk in conn.execute(f"PRAGMA foreign_key_list({_q(t)})"):
                # (id, seq, ref_table, from, to, on_update, on_delete, match)
                col_lines.append(
                    f"  FOREIGN KEY ({_q(fk[3])}) REFERENCES {_q(fk[2])}({_q(fk[4])})"
                )
            parts.append(",\n".join(col_lines))
            parts.append(");")
    return "\n".join(parts)


def available_dbs() -> list[str]:
    if not DB_DIR.exists():
        return []
    return sorted(p.stem for p in DB_DIR.glob("*.sqlite"))

"""Allowed values per collection and a WHERE-clause builder that accepts only typed, validated values."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

ENUMS_PATH = Path(__file__).with_name("enums.json")

# Columns find_traces can filter on, and the Python type each value must have.
FILTER_COLUMNS: dict[str, type] = {
    "model": str,
    "correct_strict": bool,
    "subject": str,
    "level": str,
    "token_budget": int,
    "finish_reason": str,
}

ENUM_COLUMNS = ("model", "subject", "level", "token_budget", "finish_reason", "dataset")


class InvalidArgument(ValueError):
    """An argument is outside the allowed values; the message lists them."""


@lru_cache(maxsize=1)
def load_enums() -> dict[str, Any]:
    with ENUMS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def collection_entry(collection: str) -> dict[str, Any]:
    entries = load_enums()["collections"]
    if collection not in entries:
        raise InvalidArgument(f"unknown collection {collection!r}; allowed: {sorted(entries)}")
    return entries[collection]


def allowed_values(collection: str, column: str) -> list[Any]:
    return list(collection_entry(collection)["values"].get(column, []))


def validate_value(collection: str, column: str, value: Any) -> None:
    """Raise InvalidArgument unless value is one of the known values of column in collection."""
    allowed = allowed_values(collection, column)
    if any(v == value and type(v) is type(value) for v in allowed):
        return
    if not allowed:
        raise InvalidArgument(f"{column} has no values in {collection}; omit this filter")
    raise InvalidArgument(f"{column}={value!r} is not in {collection}; allowed values: {allowed}")


def quote_ident(name: str) -> str:
    if name not in FILTER_COLUMNS and name not in ENUM_COLUMNS:
        raise InvalidArgument(f"column {name!r} cannot be used in a filter")
    return '"' + name.replace('"', '""') + '"'


def quote_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(int(value))
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    raise InvalidArgument(f"unsupported filter value type {type(value).__name__}")


def build_where(filters: dict[str, Any]) -> str | None:
    """Build `"col"=literal AND ...` from typed values; None values are skipped."""
    parts = []
    for column, value in filters.items():
        if value is None:
            continue
        expected = FILTER_COLUMNS.get(column)
        if expected is None:
            raise InvalidArgument(f"column {column!r} cannot be used in a filter")
        if expected is int and (isinstance(value, bool) or not isinstance(value, int)):
            raise InvalidArgument(f"{column} must be an integer")
        if expected is not int and not isinstance(value, expected):
            raise InvalidArgument(f"{column} must be {expected.__name__}")
        parts.append(f"{quote_ident(column)}={quote_literal(value)}")
    return " AND ".join(parts) if parts else None

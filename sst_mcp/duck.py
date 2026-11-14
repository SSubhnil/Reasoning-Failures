"""DuckDB queries over the dataset's auto-converted parquet (accuracy breakdowns and enum building)."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from .models import DATASET

GROUP_COLUMNS = ("model", "subject", "level", "token_budget", "finish_reason", "dataset")


class QueryTimeout(RuntimeError):
    """A DuckDB query ran past its time limit and was interrupted."""


def parquet_url(collection: str) -> str:
    return f"hf://datasets/{DATASET}@~parquet/{collection}/train/*.parquet"


def default_cache_dir() -> Path:
    root = os.environ.get("SST_MCP_CACHE_DIR") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "sst_mcp"
    )
    return Path(root)


class DuckRunner:
    """Runs read-only queries in a fresh DuckDB connection with a hard time limit."""

    def __init__(self, source: str | None = None, timeout_s: float = 20.0) -> None:
        # `source` overrides the parquet location (a format string with {collection}); tests use local files.
        self.source = source
        self.timeout_s = timeout_s
        self._setup_lock = threading.Lock()
        self._installed = False

    def url(self, collection: str) -> str:
        return self.source.format(collection=collection) if self.source else parquet_url(collection)

    def _connect(self):
        import duckdb

        con = duckdb.connect(":memory:")
        con.execute("SET enable_progress_bar=false")
        if self.source is None:
            with self._setup_lock:
                if not self._installed:
                    con.execute("INSTALL httpfs")
                    self._installed = True
            con.execute("LOAD httpfs")
            token = os.environ.get("HF_TOKEN")
            if token:
                con.execute("CREATE SECRET hf_token (TYPE huggingface, TOKEN '" + token.replace("'", "''") + "')")
        return con

    def query(self, sql: str, params: list[Any] | None = None) -> list[tuple]:
        con = self._connect()
        timed_out = threading.Event()

        def _interrupt() -> None:
            timed_out.set()
            con.interrupt()

        timer = threading.Timer(self.timeout_s, _interrupt)
        timer.start()
        try:
            return con.execute(sql, params or []).fetchall()
        except Exception:
            if timed_out.is_set():
                raise QueryTimeout(f"query exceeded {self.timeout_s:g} s") from None
            raise
        finally:
            timer.cancel()
            con.close()

    def breakdown(
        self, collection: str, group_by: str, model: str | None = None, token_budget: int | None = None
    ) -> list[dict[str, Any]]:
        if group_by not in GROUP_COLUMNS:
            raise ValueError(f"group_by must be one of {GROUP_COLUMNS}")
        where, params = [], []
        if model is not None:
            where.append('"model" = ?')
            params.append(model)
        if token_budget is not None:
            where.append('"token_budget" = ?')
            params.append(int(token_budget))
        sql = (
            f'SELECT "{group_by}" AS g, count(*) AS n, '
            "avg(CASE WHEN correct_strict THEN 1.0 WHEN correct_strict IS NULL THEN NULL ELSE 0.0 END), "
            "avg(CASE WHEN correct_answer_forced THEN 1.0 WHEN correct_answer_forced IS NULL THEN NULL ELSE 0.0 END) "
            f"FROM read_parquet('{self.url(collection)}')"
            + (" WHERE " + " AND ".join(where) if where else "")
            + " GROUP BY 1 ORDER BY 1 NULLS LAST"
        )
        return [
            {
                "group": g,
                "n": int(n),
                "accuracy_strict": None if a is None else round(float(a), 4),
                "accuracy_answer_forced": None if b is None else round(float(b), 4),
            }
            for g, n, a, b in self.query(sql, params)
        ]

    def distinct_values(self, collection: str, columns: tuple[str, ...] = GROUP_COLUMNS) -> dict[str, list[Any]]:
        """Distinct non-null, non-empty values per column, in one scan."""
        select = ", ".join(f'list(DISTINCT "{c}") FILTER (WHERE "{c}" IS NOT NULL)' for c in columns)
        row = self.query(f"SELECT {select} FROM read_parquet('{self.url(collection)}')")[0]
        out = {}
        for column, values in zip(columns, row, strict=True):
            vals = [v for v in (values or []) if not (isinstance(v, str) and not v.strip())]
            out[column] = sorted(vals)
        return out


class DiskCache:
    """JSON results on disk, keyed by a hash of the request, with a TTL."""

    def __init__(self, directory: Path | None = None, ttl_s: float = 86400.0) -> None:
        self.directory = Path(directory) if directory else default_cache_dir()
        self.ttl_s = ttl_s

    def _path(self, key: dict[str, Any]) -> Path:
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:32]
        return self.directory / f"{digest}.json"

    def get(self, key: dict[str, Any]) -> Any | None:
        path = self._path(key)
        try:
            if time.time() - path.stat().st_mtime > self.ttl_s:
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def set(self, key: dict[str, Any], value: Any) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self._path(key)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(value), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            pass

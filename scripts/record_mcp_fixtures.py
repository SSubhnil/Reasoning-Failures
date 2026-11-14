"""Record dataset-viewer responses for the offline MCP tests (tests/mcp_server/fixtures/).

    python scripts/record_mcp_fixtures.py
    python scripts/record_mcp_fixtures.py --only filter_aime search_math500 --attempts 10

Long text cells are trimmed so every fixture stays small. A failed request is not recorded. If /filter fails,
filter_aime is derived from real /rows data filtered locally, and the fixture says so in "derived_from".
The "Authentication check" 500 is transient and cannot be triggered on demand, so error_auth_check.json holds
the body observed live on 14 Nov 2026.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "mcp_server" / "fixtures"
BASE = "https://datasets-server.huggingface.co"
DATASET = "ssubhnil/Reasoning-Failures"
TRACE_KEEP = 6000
TEXT_KEEP = 600
AUTH_BODY = {
    "error": "Authentication check on the Hugging Face Hub failed or timed out. "
    "Please try again later, it's a temporary internal issue."
}

REQUESTS = {
    "rows_math500": ("rows", {"config": "eval_math500", "offset": 0, "length": 3}),
    "filter_aime": (
        "filter",
        {"config": "eval_aime", "where": '"model"=\'qwen3-8b\' AND "correct_strict"=false', "offset": 0, "length": 3},
    ),
    "search_math500": ("search", {"config": "eval_math500", "query": "hexagon", "offset": 0, "length": 2}),
    "rows_deepmath_scan_limit": ("rows", {"config": "eval_deepmath", "offset": 5, "length": 1}),
}


def trim_trace(text: str) -> str:
    """Keep the start and the end of a trace, both cut at paragraph breaks, so steps still exist."""
    if len(text) <= TRACE_KEEP:
        return text
    head, tail = text[: TRACE_KEEP // 2], text[-TRACE_KEEP // 2 :]
    head = head[: head.rfind("\n\n")] if "\n\n" in head else head
    tail = tail[tail.find("\n\n") + 2 :] if "\n\n" in tail else tail
    return head + "\n\n" + tail


def trim(data: dict) -> dict:
    for item in data.get("rows", []):
        row = item["row"]
        if isinstance(row.get("trace"), str):
            row["trace"] = trim_trace(row["trace"])
        for key in ("question", "gt_answer", "predicted_answer"):
            if isinstance(row.get(key), str) and len(row[key]) > TEXT_KEEP:
                row[key] = row[key][:TEXT_KEEP]
    data.pop("features", None)
    return data


def fetch(client: httpx.Client, endpoint: str, params: dict, attempts: int) -> tuple[int, dict]:
    full = {"dataset": DATASET, "split": "train", **params}
    status, body = 0, {}
    for attempt in range(attempts):
        if attempt:
            time.sleep(min(30, 4 * attempt))
        try:
            resp = client.get(f"{BASE}/{endpoint}", params=full)
        except httpx.HTTPError as exc:
            status, body = 0, {"error": type(exc).__name__}
            continue
        status = resp.status_code
        try:
            body = resp.json()
        except ValueError:
            body = {"error": resp.text[:300]}
        if status == 200 or "Scan size limit" in str(body):
            break
    return status, body


def derive_filter_aime(client: httpx.Client) -> None:
    """Build a /filter-shaped fixture from /rows when /filter is unavailable (same response format)."""
    path = OUT / "filter_aime.json"
    if path.exists() and is_good(json.loads(path.read_text())):
        return
    hits, body = [], {}
    for offset in range(0, 1200, 100):
        status, body = fetch(client, "rows", {"config": "eval_aime", "offset": offset, "length": 100}, 4)
        if status != 200:
            return
        hits += [r for r in body["rows"] if r["row"]["model"] == "qwen3-8b" and r["row"]["correct_strict"] is False]
        if len(hits) >= 3:
            break
    sys.path.insert(0, str(ROOT))
    from sst_mcp.duck import DuckRunner, parquet_url

    source = parquet_url("eval_aime")
    sql = f"SELECT count(*) FROM read_parquet('{source}') WHERE model = 'qwen3-8b' AND NOT correct_strict"
    total = DuckRunner().query(sql)[0][0]
    derived = {k: v for k, v in body.items() if k != "rows"}
    derived.update(rows=hits[:3], num_rows_total=int(total), num_rows_per_page=3)
    record = {
        "endpoint": "filter",
        "params": REQUESTS["filter_aime"][1],
        "status": 200,
        "derived_from": "/rows eval_aime pages filtered locally, num_rows_total counted with DuckDB: /filter "
        "returned the 'Authentication check' 500 on every attempt when this was recorded",
        "body": trim(derived),
    }
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"filter_aime: derived from /rows ({len(hits)} matching rows)", file=sys.stderr)


def is_good(record: dict) -> bool:
    return record.get("status") == 200 or "Scan size limit" in json.dumps(record.get("body", {}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=sorted(REQUESTS), help="record only these fixtures")
    ap.add_argument("--attempts", type=int, default=4, help="attempts per request")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with httpx.Client(timeout=httpx.Timeout(90, connect=10), headers=headers) as client:
        for name in args.only or REQUESTS:
            endpoint, params = REQUESTS[name]
            status, body = fetch(client, endpoint, params, args.attempts)
            record = {"endpoint": endpoint, "params": params, "status": status, "body": trim(body)}
            path = OUT / f"{name}.json"
            if not is_good(record):
                print(f"{name}: HTTP {status}; not recorded", file=sys.stderr)
                continue
            path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"{name}: HTTP {status}, {path.stat().st_size / 1024:.0f} KB", file=sys.stderr)
        derive_filter_aime(client)
    auth = {"endpoint": "filter", "params": {}, "status": 500, "body": AUTH_BODY}
    (OUT / "error_auth_check.json").write_text(json.dumps(auth, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

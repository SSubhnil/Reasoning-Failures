"""Build sst_mcp/enums.json: allowed filter values per collection and which viewer endpoints work.

Distinct values come from DuckDB over the parquet conversion (one scan per collection). Endpoint support
comes from probing /rows, /filter and /search on every collection. A permanent failure (for example
"Scan size limit exceeded") marks the endpoint unsupported; a transient one (429, 5xx, timeout) is
recorded in the notes and the endpoint stays supported.

    python scripts/build_mcp_enums.py                 # probe and rebuild everything
    python scripts/build_mcp_enums.py --skip-probe    # rebuild values, keep the stored support flags
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sst_mcp.duck import DuckRunner  # noqa: E402
from sst_mcp.hub import BASE_URL, SCAN_LIMIT_MARKER, HubClient  # noqa: E402
from sst_mcp.models import COLLECTIONS, DATASET  # noqa: E402
from sst_mcp.query import ENUMS_PATH  # noqa: E402

PROBES = {
    "rows": {"offset": 0, "length": 1},
    "filter": {"where": '"model"=\'qwen3-8b\' AND "correct_strict"=false', "offset": 0, "length": 1},
    "search": {"query": "triangle", "offset": 0, "length": 1},
}


async def probe(client: httpx.AsyncClient, endpoint: str, config: str, attempts: int) -> dict:
    params = {"dataset": DATASET, "config": config, "split": "train", **PROBES[endpoint]}
    outcomes = []
    for attempt in range(attempts):
        if attempt:
            await asyncio.sleep(3 * attempt)
        start = time.perf_counter()
        try:
            resp = await client.get(f"{BASE_URL}/{endpoint}", params=params)
            status, body = resp.status_code, " ".join(resp.text.split())[:160]
        except httpx.HTTPError as exc:
            status, body = "timeout" if isinstance(exc, httpx.TimeoutException) else "error", type(exc).__name__
        latency = round(time.perf_counter() - start, 1)
        outcomes.append({"status": status, "latency_s": latency, "body": body if status != 200 else ""})
        if status == 200 or (status == 500 and SCAN_LIMIT_MARKER in body):
            break
    last = outcomes[-1]
    permanent = last["status"] == 500 and SCAN_LIMIT_MARKER in last["body"]
    if isinstance(last["status"], int) and 400 <= last["status"] < 500 and last["status"] != 429:
        permanent = True
    return {"ok": last["status"] == 200, "permanent_failure": permanent, "attempts": outcomes}


def _note(endpoint: str, result: dict) -> str | None:
    if result["ok"]:
        lat = result["attempts"][-1]["latency_s"]
        failed = len(result["attempts"]) - 1
        return f"/{endpoint}: ok in {lat} s" + (f" after {failed} failed attempt(s)" if failed else "")
    last = result["attempts"][-1]
    kind = "unsupported" if result["permanent_failure"] else "failed during probing (transient)"
    return f"/{endpoint}: {kind}: HTTP {last['status']} {last['body'][:100]}".rstrip()


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-probe", action="store_true", help="keep support flags and notes from the existing file")
    ap.add_argument("--attempts", type=int, default=3, help="attempts per endpoint probe")
    ap.add_argument("--only", nargs="*", choices=COLLECTIONS, help="rebuild only these collections")
    args = ap.parse_args()

    old = json.loads(ENUMS_PATH.read_text()) if ENUMS_PATH.exists() else {"collections": {}}
    collections = dict(old.get("collections", {}))
    hub = HubClient()
    info = await hub.get("info", {})
    duck = DuckRunner(timeout_s=600)
    headers = {"Authorization": f"Bearer {hub.token}"} if hub.token else {}
    async with httpx.AsyncClient(timeout=httpx.Timeout(90, connect=10), headers=headers) as client:
        for name in args.only or COLLECTIONS:
            meta = info["dataset_info"][name]
            entry = {
                "rows": int(meta["splits"]["train"]["num_examples"]),
                "columns": list(meta["features"]),
            }
            start = time.perf_counter()
            entry["values"] = duck.distinct_values(name)
            print(f"{name}: distinct values in {time.perf_counter() - start:.1f} s", file=sys.stderr)
            prev = collections.get(name, {})
            if args.skip_probe and prev:
                for k in ("filter_supported", "search_supported", "rows_supported", "notes", "probe"):
                    entry[k] = prev[k]
            else:
                results = {ep: await probe(client, ep, name, args.attempts) for ep in PROBES}
                entry["rows_supported"] = not results["rows"]["permanent_failure"]
                entry["filter_supported"] = not results["filter"]["permanent_failure"]
                entry["search_supported"] = not results["search"]["permanent_failure"]
                stats = await hub.get("statistics", {"config": name, "split": "train"})
                notes = [n for n in (_note(ep, r) for ep, r in results.items()) if n]
                if stats.get("partial"):
                    notes.append(f"/statistics is partial ({stats.get('num_examples')} of {entry['rows']} rows)")
                for column, values in entry["values"].items():
                    if not values:
                        notes.append(f"{column} is empty in this collection")
                entry["notes"] = notes
                entry["probe"] = {"date": datetime.now(UTC).strftime("%Y-%m-%d"), "results": results}
                print(f"{name}: " + "; ".join(notes), file=sys.stderr)
            collections[name] = entry
    out = {
        "dataset": DATASET,
        "built": datetime.now(UTC).strftime("%Y-%m-%d"),
        "collections": {n: collections[n] for n in COLLECTIONS if n in collections},
    }
    ENUMS_PATH.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {ENUMS_PATH}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())

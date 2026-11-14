"""Smoke-test a running Reasoning-Failures MCP server over Streamable HTTP.

    python scripts/smoke_mcp_http.py http://127.0.0.1:7860/mcp
    python scripts/smoke_mcp_http.py https://ssubhnil-reasoning-failures-mcp.hf.space/mcp

Lists the tools, calls list_collections and one find_traces, and sends an initialize request with the
Space's Host header to check that DNS-rebinding protection does not refuse it. Exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from urllib.parse import urlparse

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

SPACE_HOST = "ssubhnil-reasoning-failures-mcp.hf.space"
EXPECTED = {"list_collections", "find_traces", "get_trace", "check_answer", "accuracy_breakdown"}


FAILURES: list[str] = []


def check(ok: bool, what: str) -> None:
    print(("PASS " if ok else "FAIL ") + what)
    if not ok:
        FAILURES.append(what)


async def mcp_checks(url: str, find_args: dict) -> None:
    async with streamablehttp_client(url, timeout=30, sse_read_timeout=300) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = {t.name for t in (await session.list_tools()).tools}
            check(names == EXPECTED, f"list_tools returned {sorted(names)}")
            res = await session.call_tool("list_collections", {})
            collections = (res.structuredContent or {}).get("collections", [])
            check(not res.isError and len(collections) == 9, "list_collections")
            res = await session.call_tool("find_traces", find_args)
            if res.isError:
                print("  find_traces error: " + res.content[0].text)
            else:
                out = res.structuredContent
                print(f"  find_traces returned {len(out['items'])} item(s), total_matches={out['total_matches']}")
            check(not res.isError, f"find_traces {json.dumps(find_args)}")


def host_header_check(url: str) -> None:
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "smoke", "version": "0"},
        },
    }
    headers = {"Host": SPACE_HOST, "Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    resp = httpx.post(url, json=body, headers=headers, timeout=30)
    check(resp.status_code == 200, f"initialize with Host: {SPACE_HOST} -> HTTP {resp.status_code}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url", help="server URL ending in /mcp")
    ap.add_argument("--collection", default="eval_aime")
    ap.add_argument("--skip-host-check", action="store_true", help="skip the Host-header check")
    args = ap.parse_args()
    find_args = {"collection": args.collection, "model": "qwen3-8b", "correct": False, "limit": 2}
    asyncio.run(mcp_checks(args.url, find_args))
    host = urlparse(args.url).hostname or ""
    if not args.skip_host_check and host in {"127.0.0.1", "localhost", "0.0.0.0"}:
        host_header_check(args.url)
    if FAILURES:
        sys.exit(f"smoke test failed: {len(FAILURES)} check(s)")
    print("smoke test passed")


if __name__ == "__main__":
    main()

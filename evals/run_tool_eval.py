"""Tool-calling eval for the sst_mcp server.

A hand-written loop: an LLM (OpenRouter, or a scripted fake with --dry-run) is given the MCP tools and the
skill text as system prompt; the tool calls it makes are scored against evals/golden_tasks.jsonl.
"""

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_FILE_SYSTEM", "READ_ONLY")

from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent.parent
SKILL_PATH = ROOT / "skills" / "reasoning-failure-analysis" / "SKILL.md"
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
LLM_TIMEOUT_S = 60
LLM_RETRIES = 2
TOOL_RESULT_CHARS = 6000
# Refuse tasks: heavy trace reading fails the task.
MAX_GET_TRACE = 5
MAX_FIND_TRACES = 3


def load_system_prompt() -> str:
    text = SKILL_PATH.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            text = parts[2]
    return text.strip()


def load_tasks(path: Path, only: list[str] | None, limit: int | None) -> list[dict[str, Any]]:
    tasks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if only:
        wanted = set(only)
        tasks = [t for t in tasks if t["id"] in wanted]
        missing = wanted - {t["id"] for t in tasks}
        if missing:
            sys.exit(f"unknown task ids: {sorted(missing)}")
    if limit:
        tasks = tasks[:limit]
    return tasks


def to_openai_tools(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description or "", "parameters": t.inputSchema},
        }
        for t in mcp_tools
    ]


class OpenRouterLLM:
    def __init__(self, model: str):
        from openai import AsyncOpenAI

        self.model = model
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], max_retries=0
        )

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(LLM_RETRIES + 1):
            try:
                resp = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model, messages=messages, tools=tools, timeout=LLM_TIMEOUT_S
                    ),
                    timeout=LLM_TIMEOUT_S + 5,
                )
                msg = resp.choices[0].message
                calls = [
                    {"id": c.id, "name": c.function.name, "arguments": c.function.arguments or "{}"}
                    for c in (msg.tool_calls or [])
                ]
                return {"content": msg.content, "tool_calls": calls}
            except Exception as exc:
                last = exc
                await asyncio.sleep(2 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {LLM_RETRIES + 1} attempts: {last!r}")


class ScriptedLLM:
    """Dry-run stand-in: replays a task's golden calls, then answers. Not a model."""

    def __init__(self, tasks: list[dict[str, Any]]):
        self.tasks = {t["task"]: t for t in tasks}
        self.state: dict[str, int] = {}

    def _golden_calls(self, task: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        calls = []
        for name in task["expected_tools"]:
            args = dict(task["expected_args"].get(name, {}))
            if name == "check_answer":
                m = re.search(r"\\boxed\{[^\"]*?\}(?=[.\"])", task["task"]) or re.search(
                    r"\\boxed\{.*?\}", task["task"]
                )
                args.setdefault("response", m.group(0) if m else "")
            if name == "get_trace":
                args.setdefault("row_idx", 0)
            if name == "find_traces":
                args.setdefault("limit", 3)
            calls.append((name, args))
        return calls

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        user = next(m["content"] for m in messages if m["role"] == "user")
        task = self.tasks[user]
        calls = self._golden_calls(task)
        i = self.state.get(user, 0)
        self.state[user] = i + 1
        if i < len(calls):
            name, args = calls[i]
            return {"content": None, "tool_calls": [{"id": f"fake{i}", "name": name, "arguments": json.dumps(args)}]}
        return {"content": "(scripted answer)", "tool_calls": []}


def _result_text(result: Any) -> str:
    parts = [getattr(c, "text", None) or str(c) for c in result.content]
    return "\n".join(parts)


async def run_task(
    llm: Any, session: ClientSession, tools: list[dict[str, Any]], system: str, task: dict[str, Any], max_turns: int
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": task["task"]},
    ]
    calls: list[dict[str, Any]] = []
    final = ""
    error = None
    try:
        for _ in range(max_turns):
            reply = await llm.chat(messages, tools)
            assistant: dict[str, Any] = {"role": "assistant", "content": reply["content"] or ""}
            if reply["tool_calls"]:
                assistant["tool_calls"] = [
                    {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                    for c in reply["tool_calls"]
                ]
            messages.append(assistant)
            if not reply["tool_calls"]:
                final = reply["content"] or ""
                break
            for c in reply["tool_calls"]:
                try:
                    args = json.loads(c["arguments"] or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("arguments are not a JSON object")
                except ValueError as exc:
                    args = {}
                    text, is_error = f"could not parse tool arguments: {exc}", True
                else:
                    try:
                        res = await asyncio.wait_for(session.call_tool(c["name"], args), timeout=120)
                        text, is_error = _result_text(res), bool(res.isError)
                    except Exception as exc:
                        text, is_error = f"tool call failed: {exc!r}", True
                calls.append({"name": c["name"], "args": args, "is_error": is_error})
                if len(text) > TOOL_RESULT_CHARS:
                    text = text[:TOOL_RESULT_CHARS] + "\n[truncated]"
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": text})
        else:
            error = f"no final answer within {max_turns} turns"
    except Exception as exc:
        error = repr(exc)
    return {"calls": calls, "final": final, "error": error, "transcript": messages[1:]}


def _eq(a: Any, b: Any) -> bool:
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().casefold() == b.strip().casefold()
    return a == b


def args_match(expected_args: dict[str, dict[str, Any]], calls: list[dict[str, Any]]) -> dict[str, bool]:
    out = {}
    for tool, want in expected_args.items():
        out[tool] = any(
            c["name"] == tool and all(k in c["args"] and _eq(c["args"][k], v) for k, v in want.items()) for c in calls
        )
    return out


def tool_selection(task: dict[str, Any], calls: list[dict[str, Any]], final: str) -> tuple[float, str]:
    metric = ToolCorrectnessMetric(threshold=1.0, should_consider_ordering=True, include_reason=True)
    case = LLMTestCase(
        input=task["task"],
        actual_output=final or "(no answer)",
        tools_called=[ToolCall(name=c["name"], input_parameters=c["args"]) for c in calls],
        expected_tools=[ToolCall(name=n) for n in task["expected_tools"]],
    )
    score = metric.measure(case, _show_indicator=False)
    return float(score), str(metric.reason)


def score_task(task: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    calls = run["calls"]
    res: dict[str, Any] = {"id": task["id"], "kind": task["kind"], "task": task["task"]}
    n_get = sum(c["name"] == "get_trace" for c in calls)
    n_find = sum(c["name"] == "find_traces" for c in calls)
    if task["kind"] == "refuse":
        ok = n_get <= MAX_GET_TRACE and n_find <= MAX_FIND_TRACES
        res.update(
            tool_score=None,
            tool_pass=ok,
            args_pass=True,
            arg_detail={},
            reason=f"find_traces={n_find}, get_trace={n_get}",
        )
    else:
        score, reason = tool_selection(task, calls, run["final"])
        detail = args_match(task["expected_args"], calls)
        res.update(
            tool_score=score, tool_pass=score >= 1.0, args_pass=all(detail.values()), arg_detail=detail, reason=reason
        )
    if run["error"]:
        res.update(tool_pass=False, args_pass=False)
    res["passed"] = res["tool_pass"] and res["args_pass"]
    res["error"] = run["error"]
    res["tool_calls"] = calls
    res["final_answer"] = run["final"]
    res["transcript"] = run["transcript"]
    return res


def write_reports(results: list[dict[str, Any]], model: str, dry: bool, out_dir: Path) -> tuple[Path, Path]:
    now = datetime.now(UTC)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", "scripted-fake" if dry else model).strip("-")
    stem = f"{'dryrun_' if dry else ''}{now:%Y-%m-%d}_{slug}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tool_tasks = [r for r in results if r["kind"] == "tool"]
    sel = sum(r["tool_pass"] for r in tool_tasks)
    arg = sum(r["args_pass"] for r in tool_tasks)
    ref = [r for r in results if r["kind"] == "refuse"]
    ref_ok = sum(r["passed"] for r in ref)
    total = sum(r["passed"] for r in results)
    summary = {
        "tool_selection": f"{sel}/{len(tool_tasks)}",
        "arguments": f"{arg}/{len(tool_tasks)}",
        "refuse": f"{ref_ok}/{len(ref)}",
        "passed": f"{total}/{len(results)}",
    }
    payload = {
        "dry_run": dry,
        "model": "scripted fake LLM" if dry else model,
        "date_utc": now.isoformat(timespec="seconds"),
        "summary": summary,
        "tasks": results,
    }
    jpath, mpath = out_dir / f"{stem}.json", out_dir / f"{stem}.md"
    jpath.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = []
    if dry:
        lines += ["DRY RUN — scripted fake LLM, not a model score", ""]
    lines += [
        "# Tool-calling eval",
        "",
        f"- Model: {payload['model']}",
        f"- Date (UTC): {payload['date_utc']}",
        f"- Tasks run: {len(results)}",
        f"- Tool-selection accuracy (tool tasks): {summary['tool_selection']}",
        f"- Argument accuracy (tool tasks): {summary['arguments']}",
        f"- Refuse tasks passed: {summary['refuse']}",
        f"- Tasks passed overall: {summary['passed']}",
        "",
        "| id | kind | tools called | tool selection | args | result |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        called = ", ".join(c["name"] for c in r["tool_calls"]) or "-"
        verdict = "PASS" if r["passed"] else ("ERROR" if r["error"] else "FAIL")
        lines.append(
            f"| {r['id']} | {r['kind']} | {called} | {'pass' if r['tool_pass'] else 'fail'} "
            f"| {'pass' if r['args_pass'] else 'fail'} | {verdict} |"
        )
    mpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jpath, mpath


async def amain(args: argparse.Namespace) -> int:
    tasks = load_tasks(Path(args.tasks), args.only, args.limit)
    if not tasks:
        sys.exit("no tasks selected")
    llm: Any = ScriptedLLM(tasks) if args.dry_run else OpenRouterLLM(args.model)
    system = load_system_prompt()
    server_env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}  # the server needs no LLM key
    params = StdioServerParameters(command=sys.executable, args=["-m", "sst_mcp"], env=server_env, cwd=str(ROOT))
    results = []
    with open(os.devnull, "w") as errlog:
        async with stdio_client(params, errlog=errlog) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            tools = to_openai_tools((await session.list_tools()).tools)
            for task in tasks:
                run = await run_task(llm, session, tools, system, task, args.max_turns)
                res = score_task(task, run)
                results.append(res)
                print(
                    f"{task['id']} {'PASS' if res['passed'] else 'FAIL'} "
                    f"calls={[c['name'] for c in res['tool_calls']]} {res['error'] or ''}"
                )
    jpath, mpath = write_reports(results, args.model, args.dry_run, Path(args.out_dir))
    print(mpath.read_text(encoding="utf-8"))
    print(f"wrote {jpath} and {mpath}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model id")
    p.add_argument("--tasks", default=str(ROOT / "evals" / "golden_tasks.jsonl"))
    p.add_argument("--only", nargs="+", help="task ids to run")
    p.add_argument("--limit", type=int, help="run only the first N selected tasks")
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--out-dir", default=str(ROOT / "evals" / "results"))
    p.add_argument("--dry-run", action="store_true", help="use a scripted fake LLM instead of OpenRouter")
    args = p.parse_args()
    if not args.dry_run and not os.environ.get("OPENROUTER_API_KEY"):
        print(
            "OPENROUTER_API_KEY is not set: the eval was NOT run and no scores were produced. "
            "Set the key, or use --dry-run for a scripted check of the harness.",
            file=sys.stderr,
        )
        sys.exit(2)
    sys.exit(asyncio.run(amain(args)))


if __name__ == "__main__":
    main()

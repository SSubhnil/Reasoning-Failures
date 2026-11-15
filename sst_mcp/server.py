"""MCP server for the Reasoning-Failures dataset: tools, resources and the analysis prompt."""

import functools
import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import Field

from . import steps as steps_mod
from .checker import AnswerChecker, normalise_gold
from .duck import DiskCache, DuckRunner, QueryTimeout
from .hub import HubClient, UnsupportedForCollection, UpstreamError
from .models import (
    BIBTEX,
    CITATION,
    COLLECTIONS,
    AccuracyBreakdownOutput,
    BreakdownRow,
    CheckAnswerOutput,
    Collection,
    CollectionInfo,
    FindTracesOutput,
    GetTraceOutput,
    GroupBy,
    ListCollectionsOutput,
    Step,
    TraceMeta,
    TraceView,
)
from .query import InvalidArgument, build_where, collection_entry, load_enums, validate_value

log = logging.getLogger("sst_mcp")

ModelName = Literal["qwen3-1.7b", "qwen3-4b", "qwen3-8b", "qwen3-14b"]

STEP_DEFINITION = (
    "From the repository README (github.com/SSubhnil/Reasoning-Failures):\n\n"
    '> "Steps are made by splitting each trace at blank lines and merging adjacent blocks until every step has '
    'at least 400 tokens (`merged_step_char_spans` in `sst/shared/extractors.py`, Qwen3 tokenizer)."\n\n'
    "get_trace(view='steps') uses the same function and the Qwen3 tokenizer (Qwen/Qwen3-4B; all Qwen3 sizes "
    "share it). step_tokens=400 reproduces the paper's steps; other values give coarser or finer steps."
)

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True)
LOCAL_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)


class NotFound(LookupError):
    pass


@dataclass
class Services:
    hub: HubClient = field(default_factory=HubClient)
    duck: DuckRunner = field(default_factory=DuckRunner)
    cache: DiskCache = field(default_factory=DiskCache)
    checker: AnswerChecker = field(default_factory=AnswerChecker)
    tokenizer: Callable | None = None  # injected in tests; default loads the Qwen3 tokenizer
    viewer_deadline_s: float = 20.0  # total time find_traces waits for the viewer, retries included
    viewer_cooldown_s: float = 300.0  # after a viewer endpoint fails, go straight to parquet for this long
    fallback_timeout_s: float = 45.0
    viewer_down_until: dict[str, float] = field(default_factory=dict)


def _error_text(exc: Exception) -> str | None:
    if isinstance(exc, InvalidArgument):
        return f"invalid_argument: {exc}"
    if isinstance(exc, UnsupportedForCollection):
        return f"unsupported_for_collection: {exc}"
    if isinstance(exc, (UpstreamError, QueryTimeout)):
        return f"upstream_error: {exc}"
    if isinstance(exc, NotFound):
        return f"not_found: {exc}"
    return None


def _logged(fn):
    """Log one line per tool call to stderr and map typed errors to clear MCP errors."""

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        status = "ok"
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            text = _error_text(exc)
            status = text.split(":", 1)[0] if text else f"error:{type(exc).__name__}"
            if text is not None:
                raise ToolError(text) from exc
            raise
        finally:
            log.info("tool=%s latency_ms=%.0f status=%s", fn.__name__, (time.perf_counter() - start) * 1000, status)

    return wrapper


def _one_line(exc: Exception, limit: int = 300) -> str:
    return " ".join(str(exc).split())[:limit]


async def _viewer_or_parquet(
    svc: Services,
    endpoint: str,
    params: dict[str, Any],
    collection: str,
    filters: dict[str, Any],
    text_query: str | None,
) -> dict[str, Any]:
    """Ask the dataset viewer; if it keeps failing, run the same query on the parquet files with DuckDB."""
    if time.monotonic() >= svc.viewer_down_until.get(endpoint, 0.0):
        try:
            with anyio.fail_after(svc.viewer_deadline_s):
                return {**await svc.hub.get(endpoint, params), "source": "dataset_viewer"}
        except (UpstreamError, TimeoutError) as exc:
            if isinstance(exc, UpstreamError) and not exc.retryable:
                raise
            viewer_error = str(exc) or f"/{endpoint} gave no answer within {svc.viewer_deadline_s:g} s"
            svc.viewer_down_until[endpoint] = time.monotonic() + svc.viewer_cooldown_s
            log.warning("/%s failing; using parquet for %.0f s", endpoint, svc.viewer_cooldown_s)
    else:
        viewer_error = f"/{endpoint} failed recently"
    find = functools.partial(
        svc.duck.find_rows,
        collection,
        filters,
        params["offset"],
        params["length"],
        text_query,
        timeout_s=svc.fallback_timeout_s,
    )
    try:
        rows, total = await anyio.to_thread.run_sync(find)
    except Exception as exc:
        raise UpstreamError(f"{viewer_error.rstrip('.')}; the parquet fallback also failed: {_one_line(exc)}") from exc
    return {"rows": rows, "num_rows_total": total, "partial": False, "source": "parquet"}


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text.strip() else None


def _meta(row_idx: int, row: dict[str, Any], truncated_cells: list[str]) -> TraceMeta:
    trace = row.get("trace") or ""
    budget = row.get("token_budget")
    difficulty = row.get("difficulty")
    return TraceMeta(
        row_idx=row_idx,
        problem_id=_str_or_none(row.get("problem_id")),
        model=_str_or_none(row.get("model")),
        dataset=_str_or_none(row.get("dataset")),
        subject=_str_or_none(row.get("subject")),
        level=_str_or_none(row.get("level")),
        difficulty=None if difficulty is None else float(difficulty),
        token_budget=None if budget is None else int(budget),
        correct_strict=row.get("correct_strict"),
        finish_reason=_str_or_none(row.get("finish_reason")),
        predicted_answer=_str_or_none(row.get("predicted_answer")),
        gt_answer=_str_or_none(row.get("gt_answer")),
        question_preview=(row.get("question") or "")[:300],
        trace_chars=None if "trace" in truncated_cells else len(trace),
    )


def _fit_steps(trace: str, spans: list[tuple[int, int]], max_chars: int) -> tuple[list[Step], list[int], bool]:
    """Keep the first two and the last step; drop whole middle steps (earliest first) until the text fits."""
    all_steps = [Step(index=i, start_char=s, end_char=e, text=trace[s:e]) for i, (s, e) in enumerate(spans)]
    n = len(all_steps)
    keep = list(range(n))
    total = sum(len(s.text) for s in all_steps)
    dropped: list[int] = []
    middle = list(range(2, n - 1))
    while total > max_chars and middle:
        i = middle.pop(0)
        keep.remove(i)
        dropped.append(i)
        total -= len(all_steps[i].text)
    kept = [all_steps[i] for i in keep]
    truncated = bool(dropped)
    if total > max_chars and kept:
        share = max_chars // len(kept)
        clipped = []
        for st in kept:
            if len(st.text) > share:
                is_last = st.index == n - 1
                body = st.text[-share:] if is_last else st.text[:share]
                st = st.model_copy(update={"text": ("[...]" + body) if is_last else (body + "[...]")})
            clipped.append(st)
        kept = clipped
        truncated = True
    return kept, dropped, truncated


def build_server(
    host: str = "127.0.0.1", port: int = 8000, http: bool = False, services: Services | None = None
) -> FastMCP:
    svc = services or Services()
    mcp = FastMCP(
        "reasoning-failures",
        instructions=(
            "Read-only tools over the Reasoning-Failures dataset (Qwen3 reasoning traces on math benchmarks). "
            "Start with list_collections, then accuracy_breakdown, find_traces, get_trace and check_answer. "
            "Cite the dataset when you report results."
        ),
        host=host,
        port=port,
        stateless_http=True,
        json_response=True,
        streamable_http_path="/mcp",
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False) if http else None,
    )

    @mcp.tool(annotations=READ_ONLY)
    @_logged
    async def list_collections() -> ListCollectionsOutput:
        """List the nine dataset collections (configs) with row counts, columns and which tools work on each.

        Call this first. Collection names are the `collection` argument of every other tool."""
        enums = load_enums()["collections"]
        return ListCollectionsOutput(
            collections=[
                CollectionInfo(
                    name=name,
                    rows=enums[name]["rows"],
                    columns=enums[name]["columns"],
                    filter_supported=enums[name]["filter_supported"],
                    search_supported=enums[name]["search_supported"],
                    rows_supported=enums[name]["rows_supported"],
                    notes=enums[name]["notes"],
                )
                for name in COLLECTIONS
            ],
            citation=CITATION,
        )

    @mcp.tool(annotations=READ_ONLY)
    @_logged
    async def find_traces(
        collection: Annotated[Collection, Field(description="Collection name from list_collections.")],
        model: Annotated[
            ModelName | None, Field(description="Model that wrote the trace (exact lowercase id).")
        ] = None,
        correct: Annotated[
            bool | None,
            Field(description="True for correct traces, false for failed ones (the correct_strict column)."),
        ] = None,
        subject: Annotated[
            str | None,
            Field(
                max_length=200,
                description="Exact subject value, e.g. 'Algebra'. Invalid values list the allowed ones.",
            ),
        ] = None,
        level: Annotated[
            str | None,
            Field(max_length=200, description="Exact level value, e.g. 'Level 5'. Some collections have no levels."),
        ] = None,
        token_budget: Annotated[
            int | None, Field(ge=1, le=10_000_000, description="Exact generation token budget, e.g. 4096.")
        ] = None,
        finish_reason: Annotated[
            str | None,
            Field(max_length=200, description="Exact finish_reason value. Invalid values list the allowed ones."),
        ] = None,
        text_query: Annotated[
            str | None,
            Field(
                min_length=1,
                max_length=200,
                description="Full-text search over the text columns. Cannot be combined with the other filters.",
            ),
        ] = None,
        limit: Annotated[int, Field(ge=1, le=20, description="Rows per page (1-20). Keep it small.")] = 10,
        offset: Annotated[int, Field(ge=0, le=10_000_000, description="Rows to skip; use next_offset to page.")] = 0,
    ) -> FindTracesOutput:
        """Find reasoning traces by model, correctness, subject, level, token budget or finish reason.

        Returns metadata and a question preview per trace, not the trace itself: pass row_idx to get_trace
        to read one. Use correct=false to find failures."""
        entry = collection_entry(collection)
        filters = {
            "model": model,
            "correct_strict": correct,
            "subject": subject,
            "level": level,
            "token_budget": token_budget,
            "finish_reason": finish_reason,
        }
        for column in ("model", "subject", "level", "token_budget", "finish_reason"):
            if filters[column] is not None:
                validate_value(collection, column, filters[column])
        where = build_where(filters)
        base = {"config": collection, "split": "train", "offset": offset, "length": limit}
        if text_query is not None:
            if where is not None:
                raise InvalidArgument(
                    "text_query cannot be combined with model, correct, subject, level, token_budget or "
                    "finish_reason (the dataset viewer's search endpoint takes no filters). Use one or the other."
                )
            if not entry["search_supported"]:
                raise UnsupportedForCollection(f"text search is not available for {collection}")
            data = await _viewer_or_parquet(svc, "search", {**base, "query": text_query}, collection, {}, text_query)
        elif where is not None:
            if not entry["filter_supported"]:
                raise UnsupportedForCollection(f"filtering is not available for {collection}")
            data = await _viewer_or_parquet(svc, "filter", {**base, "where": where}, collection, filters, None)
        elif entry["rows_supported"]:
            data = {**await svc.hub.get("rows", base), "source": "dataset_viewer"}
        else:
            raise InvalidArgument(f"{collection} is too large to list without a filter; pass at least one filter")
        rows = data.get("rows") or []
        svc.hub.remember_rows(collection, rows)
        items = [_meta(int(r["row_idx"]), r["row"], list(r.get("truncated_cells") or [])) for r in rows]
        total = data.get("num_rows_total")
        if total is not None:
            next_offset = offset + len(items) if offset + len(items) < total and items else None
        else:
            next_offset = offset + len(items) if len(items) == limit else None
        notes = []
        if data["source"] == "parquet":
            notes.append("the dataset viewer was failing, so this page was read from the parquet files with DuckDB")
            if text_query is not None:
                notes.append(
                    "text_query was matched as a case-insensitive substring of the question and the trace, "
                    "not with the viewer's ranked full-text search"
                )
        return FindTracesOutput(
            collection=collection,
            total_matches=total,
            offset=offset,
            items=items,
            next_offset=next_offset,
            partial=bool(data.get("partial", False)),
            source=data["source"],
            notes=notes,
            citation=CITATION,
        )

    @mcp.tool(annotations=READ_ONLY)
    @_logged
    async def get_trace(
        collection: Annotated[Collection, Field(description="Collection name from list_collections.")],
        row_idx: Annotated[int, Field(ge=0, description="row_idx of the trace, as returned by find_traces.")],
        view: Annotated[
            TraceView,
            Field(
                description="'tail' (default): the last max_chars characters, where the answer is. "
                "'steps': the trace split into the paper's steps. 'full': the trace from the start, cut at max_chars."
            ),
        ] = "tail",
        max_chars: Annotated[
            int, Field(ge=500, le=40000, description="Maximum characters of trace text to return (500-40000).")
        ] = 6000,
        step_tokens: Annotated[
            int, Field(ge=100, le=2000, description="Minimum tokens per step for view='steps'; the paper uses 400.")
        ] = 400,
    ) -> GetTraceOutput:
        """Read one reasoning trace: its metadata plus the tail, the steps or the start of the text.

        Prefer view='tail'. Use view='steps' when you need to locate the step where the reasoning goes wrong;
        middle steps are dropped to fit max_chars and listed in dropped_step_indices."""
        entry = collection_entry(collection)
        if row_idx >= entry["rows"]:
            n = entry["rows"]
            raise NotFound(f"row_idx {row_idx} is out of range: {collection} has {n} rows (0-{n - 1})")
        notes: list[str] = []
        cached = svc.hub.cached_row(collection, row_idx)
        if cached is not None and "trace" not in cached["truncated_cells"]:
            row, truncated_cells = cached["row"], cached["truncated_cells"]
        elif entry["rows_supported"]:
            data = await svc.hub.get("rows", {"config": collection, "split": "train", "offset": row_idx, "length": 1})
            rows = data.get("rows") or []
            if not rows:
                raise NotFound(f"row_idx {row_idx} was not returned by the dataset viewer for {collection}")
            svc.hub.remember_rows(collection, rows[:1])
            row, truncated_cells = rows[0]["row"], list(rows[0].get("truncated_cells") or [])
        elif cached is not None:
            row, truncated_cells = cached["row"], cached["truncated_cells"]
        else:
            raise UnsupportedForCollection(
                f"{collection} rows cannot be fetched by index; call find_traces first; "
                "get_trace serves rows it returned"
            )
        trace = row.get("trace") or ""
        upstream_cut = "trace" in truncated_cells
        if upstream_cut:
            notes.append("the dataset viewer truncated this trace; the text is incomplete")
        meta = _meta(row_idx, row, truncated_cells)
        if view == "steps":
            spans = await anyio.to_thread.run_sync(steps_mod.step_spans, trace, step_tokens, svc.tokenizer)
            kept, dropped, cut = _fit_steps(trace, spans, max_chars)
            return GetTraceOutput(
                meta=meta,
                view=view,
                text=None,
                steps=kept,
                total_steps=len(spans),
                truncated=cut or upstream_cut,
                dropped_step_indices=dropped,
                notes=notes,
                citation=CITATION,
            )
        cut = len(trace) > max_chars
        text = trace[-max_chars:] if view == "tail" else trace[:max_chars]
        if cut:
            notes.append(f"showing {'last' if view == 'tail' else 'first'} {max_chars} of {len(trace)} characters")
        return GetTraceOutput(
            meta=meta,
            view=view,
            text=text,
            steps=None,
            truncated=cut or upstream_cut,
            dropped_step_indices=[],
            notes=notes,
            citation=CITATION,
        )

    @mcp.tool(annotations=LOCAL_ONLY)
    @_logged
    async def check_answer(
        response: Annotated[
            str,
            Field(
                min_length=1,
                max_length=200_000,
                description="Text containing the final answer in \\boxed{...} (a whole trace works; the last "
                "boxed answer is used). Without \\boxed{...} the result is reason='no_boxed_answer'.",
            ),
        ],
        gt_answer: Annotated[
            str,
            Field(
                min_length=1,
                max_length=10_000,
                description="Gold answer exactly as returned in gt_answer (e.g. '[106]'), or a LaTeX answer.",
            ),
        ],
        timeout_s: Annotated[float, Field(ge=1, le=10, description="Hard time limit in seconds (1-10).")] = 10,
    ) -> CheckAnswerOutput:
        """Check an answer with the paper's deterministic math checker (math_verify, symbolic equivalence).

        Use it to confirm a suspected mis-grade or to test an edited answer. Correct if the boxed answer
        matches any element of the gold list."""
        golds = normalise_gold(gt_answer)
        if not golds:
            raise InvalidArgument("gt_answer has no answer in it")
        result = await anyio.to_thread.run_sync(svc.checker.check_sync, response, golds, timeout_s)
        return CheckAnswerOutput(gt_used=golds, **result)

    @mcp.tool(annotations=READ_ONLY)
    @_logged
    async def accuracy_breakdown(
        collection: Annotated[Collection, Field(description="Collection name from list_collections.")],
        group_by: Annotated[GroupBy, Field(description="Column to group by.")],
        model: Annotated[ModelName | None, Field(description="Only rows from this model.")] = None,
        token_budget: Annotated[
            int | None, Field(ge=1, le=10_000_000, description="Only rows with this token budget.")
        ] = None,
    ) -> AccuracyBreakdownOutput:
        """Accuracy per group (model, subject, level, token_budget, finish_reason or dataset) over a whole collection.

        Use it to find where accuracy drops before looking at individual traces. accuracy_strict uses
        correct_strict; accuracy_answer_forced uses correct_answer_forced."""
        collection_entry(collection)
        if model is not None:
            validate_value(collection, "model", model)
        if token_budget is not None:
            validate_value(collection, "token_budget", token_budget)
        key = {"v": 1, "collection": collection, "group_by": group_by, "model": model, "token_budget": token_budget}
        rows = svc.cache.get(key)
        if rows is None:
            try:
                rows = await anyio.to_thread.run_sync(svc.duck.breakdown, collection, group_by, model, token_budget)
            except QueryTimeout as exc:
                if collection == "eval_deepmath":
                    raise UnsupportedForCollection(f"breakdown on eval_deepmath timed out ({exc})") from exc
                raise
            except Exception as exc:
                raise UpstreamError(f"parquet query failed: {_one_line(exc)}") from exc
            svc.cache.set(key, rows)
        notes = [
            "computed over every row of the collection's parquet files",
            "accuracy is the share of rows with the flag true; rows where it is null are left out",
        ]
        if not rows:
            notes.append("no rows match these filters")
        return AccuracyBreakdownOutput(
            collection=collection,
            group_by=group_by,
            rows=[BreakdownRow(**r) for r in rows],
            notes=notes,
            citation=CITATION,
        )

    @mcp.resource("rf://dataset-card", mime_type="text/markdown")
    def dataset_card() -> str:
        """Dataset description, columns, collections, licence and citation."""
        enums = load_enums()["collections"]
        lines = [
            "# Reasoning-Failures dataset",
            "",
            "Reasoning traces of Qwen3 models (1.7B, 4B, 8B, 14B) on math benchmarks, with the gold answer, the "
            "predicted answer and correctness flags. Hugging Face: `ssubhnil/Reasoning-Failures`. Code: "
            "github.com/SSubhnil/Reasoning-Failures.",
            "",
            "## Collections",
            "",
            "| collection | rows |",
            "|---|---|",
            *[f"| {n} | {enums[n]['rows']:,} |" for n in COLLECTIONS],
            "",
            "## Columns",
            "",
            ", ".join(enums[COLLECTIONS[0]]["columns"]) + ". eval_deepmath adds source_shard.",
            "",
            "- `correct_strict`: the final boxed answer verifies against the gold answer.",
            "- `correct_answer_forced`: correctness after the answer was forced at the end of the budget.",
            "- `gt_answer` and `predicted_answer` are lists stored as strings, e.g. `'[106]'`.",
            "",
            "## Licence",
            "",
            "Data: CC BY 4.0. Code: Apache-2.0. Both require attribution.",
            "",
            "## Citation",
            "",
            "```bibtex",
            BIBTEX,
            "```",
        ]
        return "\n".join(lines)

    @mcp.resource("rf://step-definition", mime_type="text/markdown")
    def step_definition() -> str:
        """How traces are split into steps (the paper's definition)."""
        return STEP_DEFINITION

    @mcp.prompt()
    def analyse_failures(collection: str, model: str) -> str:
        """Analyse where and how a model fails on one collection, following the reasoning-failure-analysis skill."""
        return (
            f"Analyse the reasoning failures of {model} on {collection} with the Reasoning-Failures tools.\n"
            f"1. Call list_collections and check what works on {collection}.\n"
            f"2. Call accuracy_breakdown(collection='{collection}', group_by=...) with model='{model}' to find "
            "where accuracy drops (try token_budget, subject or level).\n"
            f"3. Call find_traces(collection='{collection}', model='{model}', correct=False, limit=5) in the "
            "weakest group.\n"
            "4. Read at most 5 traces with get_trace(view='tail'); use view='steps' only when you need the step "
            "where the reasoning goes wrong.\n"
            "5. Use check_answer, passing gt_answer exactly as returned, to confirm any suspected mis-grade.\n"
            "6. For each failure report: the question, the model's answer, the correct answer, the step where it "
            "goes wrong (quoted, with its index), a one-line diagnosis, and the citation. Report only what the "
            "tools returned and say when results are partial. The dataset has no reliable failure-type labels; "
            "do not claim any."
        )

    return mcp


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False

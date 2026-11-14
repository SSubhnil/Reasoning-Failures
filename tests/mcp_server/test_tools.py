import copy

import httpx
import pytest
import respx
from fakes import load_fixture

from sst_mcp.duck import DuckRunner, QueryTimeout
from sst_mcp.hub import BASE_URL
from sst_mcp.query import allowed_values

pytestmark = pytest.mark.anyio

INJECTIONS = ["x' OR '1'='1", "'; DROP TABLE train; --", 'Algebra" OR "model"="b']


def body(name, **changes):
    data = copy.deepcopy(load_fixture(name)["body"])
    data.update(changes)
    return data


def error_text(result):
    assert result.isError, result
    return result.content[0].text


def auth_500():
    return httpx.Response(500, json=load_fixture("error_auth_check")["body"])


def scan_500():
    fx = load_fixture("rows_deepmath_scan_limit")
    return httpx.Response(fx["status"], json=fx["body"])


def deepmath_rows():
    rows = copy.deepcopy(load_fixture("rows_math500")["body"]["rows"])
    for i, item in enumerate(rows):
        item["row_idx"] = 900_000 + i
        item["row"]["dataset"] = "deepmath"
        item["row"]["model"] = "qwen3-4b"
    return rows


async def test_list_collections(call):
    res = await call("list_collections")
    assert not res.isError
    out = res.structuredContent
    names = [c["name"] for c in out["collections"]]
    assert len(names) == 9 and "eval_deepmath" in names
    deep = next(c for c in out["collections"] if c["name"] == "eval_deepmath")
    assert deep["rows"] == 1_235_860 and deep["rows_supported"] is False
    assert "CC BY 4.0" in out["citation"]


async def test_find_traces_filter_happy_path(call, services):
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.get("/filter").mock(return_value=httpx.Response(200, json=body("filter_aime")))
        res = await call("find_traces", collection="eval_aime", model="qwen3-8b", correct=False, limit=3)
    assert not res.isError, res.content
    params = route.calls[0].request.url.params
    assert params["where"] == '"model"=\'qwen3-8b\' AND "correct_strict"=false'
    assert params["length"] == "3" and params["offset"] == "0" and params["config"] == "eval_aime"
    out = res.structuredContent
    assert out["items"] and all(i["model"] == "qwen3-8b" and i["correct_strict"] is False for i in out["items"])
    first = out["items"][0]
    assert len(first["question_preview"]) <= 300 and first["trace_chars"] > 0
    assert out["total_matches"] == load_fixture("filter_aime")["body"]["num_rows_total"]
    assert out["next_offset"] == len(out["items"])
    assert services.hub.cached_row("eval_aime", first["row_idx"]) is not None
    assert "citation" in out


async def test_find_traces_last_page_and_partial(call):
    data = body("filter_aime", partial=True)
    data["num_rows_total"] = len(data["rows"])
    with respx.mock(base_url=BASE_URL) as mock:
        mock.get("/filter").mock(return_value=httpx.Response(200, json=data))
        res = await call("find_traces", collection="eval_aime", correct=False, limit=3)
    out = res.structuredContent
    assert out["next_offset"] is None and out["partial"] is True


async def test_find_traces_without_total_pages_by_limit(call):
    data = body("rows_math500")
    data.pop("num_rows_total")
    with respx.mock(base_url=BASE_URL) as mock:
        mock.get("/rows").mock(return_value=httpx.Response(200, json=data))
        res = await call("find_traces", collection="eval_math500", limit=len(data["rows"]))
    assert res.structuredContent["next_offset"] == len(data["rows"])


async def test_find_traces_no_filter_uses_rows(call):
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.get("/rows").mock(return_value=httpx.Response(200, json=body("rows_math500")))
        res = await call("find_traces", collection="eval_math500", limit=3, offset=0)
    assert not res.isError and route.called


async def test_find_traces_deepmath_needs_a_filter(call):
    res = await call("find_traces", collection="eval_deepmath")
    assert "invalid_argument" in error_text(res)


async def test_find_traces_search(call):
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.get("/search").mock(return_value=httpx.Response(200, json=body("rows_math500")))
        res = await call("find_traces", collection="eval_math500", text_query="x' OR '1'='1", limit=2)
    assert not res.isError
    assert route.calls[0].request.url.params["query"] == "x' OR '1'='1"


async def test_find_traces_search_with_filters_rejected(call):
    res = await call("find_traces", collection="eval_math500", text_query="hexagon", model="qwen3-8b")
    assert "cannot be combined" in error_text(res)


async def test_find_traces_bad_enum_lists_allowed(call):
    res = await call("find_traces", collection="eval_math500", subject="Astrology")
    text = error_text(res)
    assert "allowed values" in text
    assert allowed_values("eval_math500", "subject")[0] in text


async def test_find_traces_bad_model_and_collection(call):
    text = error_text(await call("find_traces", collection="eval_math500", model="QWEN3-8B"))
    assert "qwen3-8b" in text
    text = error_text(await call("find_traces", collection="eval_nope"))
    assert "eval_aime" in text


async def test_find_traces_empty_enum_column(call):
    if allowed_values("eval_aime", "level"):
        pytest.skip("eval_aime has levels")
    text = error_text(await call("find_traces", collection="eval_aime", level="Level 1"))
    assert "omit this filter" in text


@pytest.mark.parametrize("args", [{"limit": 0}, {"limit": 21}, {"offset": -1}, {"token_budget": 0}])
async def test_find_traces_bounds(call, args):
    res = await call("find_traces", collection="eval_math500", **args)
    assert res.isError


@pytest.mark.parametrize("payload", INJECTIONS)
@pytest.mark.parametrize("param", ["subject", "level", "finish_reason"])
async def test_find_traces_injection_rejected_before_any_request(call, param, payload):
    with respx.mock(base_url=BASE_URL, assert_all_called=False) as mock:
        route = mock.get("/filter")
        res = await call("find_traces", collection="eval_math500", **{param: payload})
    assert "invalid_argument" in error_text(res)
    assert not route.called


async def test_find_traces_upstream_retry_then_ok(call):
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.get("/filter").mock(
            side_effect=[auth_500(), auth_500(), httpx.Response(200, json=body("filter_aime"))]
        )
        res = await call("find_traces", collection="eval_aime", correct=False)
    assert not res.isError and route.call_count == 3


async def test_find_traces_upstream_error(call):
    with respx.mock(base_url=BASE_URL) as mock:
        mock.get("/filter").mock(side_effect=[auth_500()] * 3)
        res = await call("find_traces", collection="eval_aime", correct=False)
    text = error_text(res)
    assert "upstream_error" in text and "try again" in text


async def test_find_traces_timeout(call):
    with respx.mock(base_url=BASE_URL) as mock:
        mock.get("/filter").mock(side_effect=httpx.ReadTimeout("slow"))
        res = await call("find_traces", collection="eval_aime", correct=True)
    assert "timed out" in error_text(res)


async def test_find_traces_scan_limit(call):
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.get("/filter").mock(return_value=scan_500())
        res = await call("find_traces", collection="eval_deepmath", correct=False)
    assert "unsupported_for_collection" in error_text(res) and route.call_count == 1


async def test_find_traces_truncated_cell(call):
    data = body("rows_math500")
    data["rows"][0]["truncated_cells"] = ["trace"]
    with respx.mock(base_url=BASE_URL) as mock:
        mock.get("/rows").mock(return_value=httpx.Response(200, json=data))
        res = await call("find_traces", collection="eval_math500", limit=3)
    assert res.structuredContent["items"][0]["trace_chars"] is None


async def test_get_trace_tail_from_rows(call):
    row = load_fixture("rows_math500")["body"]["rows"][0]
    one = {**body("rows_math500"), "rows": [row]}
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.get("/rows").mock(return_value=httpx.Response(200, json=one))
        res = await call("get_trace", collection="eval_math500", row_idx=row["row_idx"], max_chars=500)
    assert not res.isError, res.content
    params = route.calls[0].request.url.params
    assert params["offset"] == str(row["row_idx"]) and params["length"] == "1"
    out = res.structuredContent
    trace = row["row"]["trace"]
    assert out["text"] == trace[-500:] and out["truncated"] == (len(trace) > 500)
    assert out["meta"]["row_idx"] == row["row_idx"] and out["steps"] is None


async def test_get_trace_full_view(call, services):
    rows = load_fixture("rows_math500")["body"]["rows"]
    services.hub.remember_rows("eval_math500", rows)
    res = await call("get_trace", collection="eval_math500", row_idx=rows[0]["row_idx"], view="full", max_chars=700)
    out = res.structuredContent
    assert out["text"] == rows[0]["row"]["trace"][:700]


async def test_get_trace_served_from_cache_after_find(call):
    with respx.mock(base_url=BASE_URL, assert_all_called=False) as mock:
        mock.get("/filter").mock(return_value=httpx.Response(200, json=body("filter_aime")))
        rows_route = mock.get("/rows")
        found = await call("find_traces", collection="eval_aime", correct=False, limit=3)
        idx = found.structuredContent["items"][0]["row_idx"]
        res = await call("get_trace", collection="eval_aime", row_idx=idx)
    assert not res.isError and not rows_route.called


async def test_get_trace_steps_drop_middle(call, services):
    rows = load_fixture("rows_math500")["body"]["rows"]
    services.hub.remember_rows("eval_math500", rows)
    longest = max(rows, key=lambda r: len(r["row"]["trace"]))
    res = await call(
        "get_trace", collection="eval_math500", row_idx=longest["row_idx"], view="steps", max_chars=500, step_tokens=100
    )
    out = res.structuredContent
    assert out["text"] is None and out["truncated"] is True
    assert out["total_steps"] >= 4 and out["dropped_step_indices"]
    idx = [s["index"] for s in out["steps"]]
    assert idx[:2] == [0, 1] and idx[-1] == out["total_steps"] - 1
    trace = longest["row"]["trace"]
    for s in out["steps"]:
        if "[...]" not in s["text"]:
            assert trace[s["start_char"] : s["end_char"]] == s["text"]


async def test_get_trace_steps_fit_without_drops(call, services):
    rows = load_fixture("rows_math500")["body"]["rows"]
    services.hub.remember_rows("eval_math500", rows)
    res = await call("get_trace", collection="eval_math500", row_idx=rows[0]["row_idx"], view="steps", max_chars=40000)
    out = res.structuredContent
    assert out["dropped_step_indices"] == [] and out["truncated"] is False
    assert "".join(s["text"] for s in out["steps"]).replace("\n", "") != ""


@pytest.mark.parametrize("args", [{"max_chars": 499}, {"max_chars": 40001}, {"step_tokens": 99}, {"row_idx": -1}])
async def test_get_trace_bounds(call, args):
    params = {"collection": "eval_math500", "row_idx": 0, **args}
    assert (await call("get_trace", **params)).isError


async def test_get_trace_row_idx_out_of_range(call):
    with respx.mock(base_url=BASE_URL, assert_all_called=False) as mock:
        route = mock.get("/rows")
        res = await call("get_trace", collection="eval_math500", row_idx=8000)
    assert "not_found" in error_text(res) and "8000 rows" in error_text(res)
    assert not route.called


async def test_get_trace_empty_rows_response(call):
    with respx.mock(base_url=BASE_URL) as mock:
        mock.get("/rows").mock(return_value=httpx.Response(200, json={**body("rows_math500"), "rows": []}))
        res = await call("get_trace", collection="eval_math500", row_idx=7)
    assert "not_found" in error_text(res)


async def test_get_trace_upstream_timeout(call):
    with respx.mock(base_url=BASE_URL) as mock:
        mock.get("/rows").mock(side_effect=httpx.ReadTimeout("slow"))
        res = await call("get_trace", collection="eval_math500", row_idx=3)
    assert "upstream_error" in error_text(res)


async def test_get_trace_deepmath_refused_without_cache(call):
    with respx.mock(base_url=BASE_URL, assert_all_called=False) as mock:
        route = mock.get("/rows")
        res = await call("get_trace", collection="eval_deepmath", row_idx=900_000)
    text = error_text(res)
    assert "unsupported_for_collection" in text and "call find_traces first" in text
    assert not route.called


async def test_get_trace_deepmath_from_row_cache(call):
    data = {**body("rows_math500"), "rows": deepmath_rows(), "num_rows_total": 412_000}
    with respx.mock(base_url=BASE_URL) as mock:
        mock.get("/filter").mock(return_value=httpx.Response(200, json=data))
        found = await call("find_traces", collection="eval_deepmath", model="qwen3-4b", correct=False, limit=3)
        assert not found.isError, found.content
        res = await call("get_trace", collection="eval_deepmath", row_idx=900_000, view="steps", step_tokens=100)
    assert not res.isError, res.content
    assert res.structuredContent["meta"]["row_idx"] == 900_000


async def test_get_trace_upstream_truncated_cell(call, services):
    rows = deepmath_rows()
    rows[0]["truncated_cells"] = ["trace"]
    services.hub.remember_rows("eval_deepmath", rows)
    res = await call("get_trace", collection="eval_deepmath", row_idx=900_000)
    out = res.structuredContent
    assert out["truncated"] is True and any("truncated" in n for n in out["notes"])


async def test_get_trace_truncated_cell_refetched_when_rows_work(call, services):
    rows = copy.deepcopy(load_fixture("rows_math500")["body"]["rows"])
    full = copy.deepcopy(rows[0])
    rows[0]["truncated_cells"] = ["trace"]
    rows[0]["row"]["trace"] = rows[0]["row"]["trace"][:100]
    services.hub.remember_rows("eval_math500", rows)
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.get("/rows").mock(return_value=httpx.Response(200, json={**body("rows_math500"), "rows": [full]}))
        res = await call("get_trace", collection="eval_math500", row_idx=full["row_idx"], max_chars=40000)
    assert route.called and res.structuredContent["text"] == full["row"]["trace"]


async def test_check_answer_tool(call):
    res = await call("check_answer", response="so \\boxed{106}", gt_answer="[106]")
    assert res.structuredContent == {
        "correct": True,
        "extracted_answer": "106",
        "gt_used": ["\\boxed{106}"],
        "reason": "ok",
    }
    res = await call("check_answer", response="no box", gt_answer="[106]", timeout_s=2)
    assert res.structuredContent["reason"] == "no_boxed_answer"


@pytest.mark.parametrize(
    "args",
    [
        {"response": "", "gt_answer": "1"},
        {"response": "x" * 200_001, "gt_answer": "1"},
        {"response": "\\boxed{1}", "gt_answer": "1", "timeout_s": 0},
        {"response": "\\boxed{1}", "gt_answer": "1", "timeout_s": 11},
    ],
)
async def test_check_answer_bounds(call, args):
    assert (await call("check_answer", **args)).isError


async def test_check_answer_empty_gold(call):
    assert "invalid_argument" in error_text(await call("check_answer", response="\\boxed{1}", gt_answer="[]"))


def expected_by_model(parquet_dir, model=None):
    import duckdb

    where = f"WHERE model = '{model}'" if model else ""
    return duckdb.sql(
        "SELECT model, count(*), round(avg(correct_strict::INT), 4), round(avg(correct_answer_forced::INT), 4) "
        f"FROM '{parquet_dir}/eval_math500/*.parquet' {where} GROUP BY 1 ORDER BY 1"
    ).fetchall()


async def test_accuracy_breakdown_by_model(call, parquet_dir):
    res = await call("accuracy_breakdown", collection="eval_math500", group_by="model")
    assert not res.isError, res.content
    out = res.structuredContent
    got = [(r["group"], r["n"], r["accuracy_strict"], r["accuracy_answer_forced"]) for r in out["rows"]]
    assert got == [tuple(x) for x in expected_by_model(parquet_dir)]
    assert out["source"] == "duckdb" and out["citation"]


async def test_accuracy_breakdown_filtered_and_cached(call, services, monkeypatch):
    budgets = allowed_values("eval_math500", "token_budget")
    first = await call("accuracy_breakdown", collection="eval_math500", group_by="subject", model="qwen3-8b")
    assert not first.isError, first.content
    groups = [r["group"] for r in first.structuredContent["rows"]]
    assert groups[-1] is None and "Algebra" in groups

    def boom(*a, **k):
        raise AssertionError("should be served from the disk cache")

    monkeypatch.setattr(services.duck, "breakdown", boom)
    again = await call("accuracy_breakdown", collection="eval_math500", group_by="subject", model="qwen3-8b")
    assert again.structuredContent == first.structuredContent
    if 2048 in budgets:
        monkeypatch.undo()
        res = await call("accuracy_breakdown", collection="eval_math500", group_by="token_budget", token_budget=2048)
        assert [r["group"] for r in res.structuredContent["rows"]] == [2048]


async def test_accuracy_breakdown_no_rows_note(call, services, monkeypatch):
    monkeypatch.setattr(services.duck, "breakdown", lambda *a: [])
    res = await call("accuracy_breakdown", collection="eval_aime", group_by="model")
    assert "no rows match these filters" in res.structuredContent["notes"]


async def test_accuracy_breakdown_bad_values(call):
    assert error_text(await call("accuracy_breakdown", collection="eval_math500", group_by="trace"))
    text = error_text(await call("accuracy_breakdown", collection="eval_math500", group_by="model", token_budget=7))
    assert "allowed values" in text


async def test_accuracy_breakdown_deepmath_timeout(call, services, monkeypatch):
    def slow(*a):
        raise QueryTimeout("query exceeded 20 s")

    monkeypatch.setattr(services.duck, "breakdown", slow)
    assert "unsupported_for_collection" in error_text(
        await call("accuracy_breakdown", collection="eval_deepmath", group_by="model")
    )
    assert "upstream_error" in error_text(await call("accuracy_breakdown", collection="eval_aime", group_by="model"))


async def test_accuracy_breakdown_query_failure(call, services, monkeypatch):
    def fail(*a):
        raise RuntimeError("HTTP 429 Too Many Requests")

    monkeypatch.setattr(services.duck, "breakdown", fail)
    text = error_text(await call("accuracy_breakdown", collection="eval_aime", group_by="model"))
    assert "upstream_error" in text and "429" in text


def test_duck_query_timeout():
    runner = DuckRunner(source="unused/{collection}", timeout_s=0.2)
    with pytest.raises(QueryTimeout):
        runner.query("SELECT count(*) FROM range(100000000000) a")


def test_duck_distinct_values(parquet_dir):
    runner = DuckRunner(source=str(parquet_dir) + "/{collection}/*.parquet")
    values = runner.distinct_values("eval_math500")
    assert values["model"] == ["qwen3-4b", "qwen3-8b"]
    assert values["token_budget"] == [2048, 4096]
    assert values["subject"] == ["Algebra", "Geometry"]
    with pytest.raises(ValueError):
        runner.breakdown("eval_math500", "trace")


def test_duck_default_url():
    assert (
        DuckRunner().url("eval_aime") == "hf://datasets/ssubhnil/Reasoning-Failures@~parquet/eval_aime/train/*.parquet"
    )


def test_disk_cache_ttl(tmp_path):
    from sst_mcp.duck import DiskCache

    cache = DiskCache(tmp_path, ttl_s=-1)
    cache.set({"a": 1}, [1])
    assert cache.get({"a": 1}) is None
    assert DiskCache(tmp_path).get({"a": 1}) == [1]


async def test_resources_and_prompt(server):
    from mcp.shared.memory import create_connected_server_and_client_session

    async with create_connected_server_and_client_session(server._mcp_server) as client:
        uris = {str(r.uri) for r in (await client.list_resources()).resources}
        assert uris == {"rf://dataset-card", "rf://step-definition"}
        card = (await client.read_resource("rf://dataset-card")).contents[0].text
        assert "@inproceedings{" in card and "CC BY 4.0" in card and "1,235,860" in card
        steps = (await client.read_resource("rf://step-definition")).contents[0].text
        assert "at least 400 tokens" in steps
        prompt = await client.get_prompt("analyse_failures", {"collection": "eval_aime", "model": "qwen3-8b"})
        text = prompt.messages[0].content.text
        assert "find_traces(collection='eval_aime', model='qwen3-8b', correct=False" in text

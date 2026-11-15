import importlib.util

import pytest

# The base environment (torch, no MCP SDK) must still run `pytest tests` unchanged.
if importlib.util.find_spec("mcp") is None or importlib.util.find_spec("respx") is None:
    collect_ignore_glob = ["test_*.py"]
else:
    from fakes import WordTokenizer
    from mcp.shared.memory import create_connected_server_and_client_session

    from sst_mcp.checker import AnswerChecker
    from sst_mcp.duck import DiskCache, DuckRunner
    from sst_mcp.hub import HubClient
    from sst_mcp.server import Services, build_server


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
def parquet_dir(tmp_path_factory):
    """Tiny local parquet files standing in for the dataset's parquet conversion."""
    import duckdb

    root = tmp_path_factory.mktemp("parquet")
    rows = []
    for i in range(40):
        model = ["qwen3-4b", "qwen3-8b"][i % 2]
        budget = [2048, 4096][(i // 2) % 2]
        subject = ["Algebra", "Geometry", None][i % 3]
        strict = (i % 5) != 0 if model == "qwen3-8b" else (i % 3) == 0
        forced = strict or i % 7 == 0
        question = "What is 50% of 20?" if i == 7 else f"Question {i}: find x."
        trace = f"Start of trace {i}.\n\nMiddle step.\n\nSo the answer is \\boxed{{{i}}}."
        rows.append(
            (f"p{i}", model, "math500", budget, subject, f"Level {1 + i % 3}", i / 40, "end_think", strict, forced)
            + (str(i), f"[{i}]", question, trace)
        )
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE t (problem_id VARCHAR, model VARCHAR, dataset VARCHAR, token_budget BIGINT, subject VARCHAR, "
        "level VARCHAR, difficulty DOUBLE, finish_reason VARCHAR, correct_strict BOOLEAN, "
        "correct_answer_forced BOOLEAN, predicted_answer VARCHAR, gt_answer VARCHAR, question VARCHAR, "
        "trace VARCHAR, i INTEGER)"
    )
    con.executemany(
        "INSERT INTO t VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [r + (n,) for n, r in enumerate(rows)]
    )
    for name in ("eval_math500", "eval_aime", "eval_deepmath"):
        (root / name).mkdir()
        for part, (lo, hi) in enumerate([(0, 25), (25, 40)]):
            path = root / name / f"{part:04d}.parquet"
            part_rows = f"SELECT * EXCLUDE (i) FROM t WHERE i >= {lo} AND i < {hi} ORDER BY i"
            con.execute(f"COPY ({part_rows}) TO '{path}' (FORMAT parquet)")
    con.close()
    return root


@pytest.fixture
def services(tmp_path, parquet_dir):
    return Services(
        hub=HubClient(backoff_s=0, token=""),
        duck=DuckRunner(source=str(parquet_dir) + "/{collection}/*.parquet"),
        cache=DiskCache(tmp_path / "cache"),
        checker=AnswerChecker(),
        tokenizer=WordTokenizer(),
    )


@pytest.fixture
def server(services):
    return build_server(services=services)


@pytest.fixture
def call(server):
    """Call a tool through a real in-memory MCP session; returns the CallToolResult."""

    async def _call(name, **arguments):
        async with create_connected_server_and_client_session(server._mcp_server) as client:
            return await client.call_tool(name, arguments)

    return _call

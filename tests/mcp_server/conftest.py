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
        rows.append((model, "math500", budget, subject, f"Level {1 + i % 3}", "end_think", strict, forced))
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE t (model VARCHAR, dataset VARCHAR, token_budget BIGINT, subject VARCHAR, level VARCHAR, "
        "finish_reason VARCHAR, correct_strict BOOLEAN, correct_answer_forced BOOLEAN)"
    )
    con.executemany("INSERT INTO t VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    for name in ("eval_math500", "eval_aime"):
        (root / name).mkdir()
        con.execute(f"COPY t TO '{root / name / '0000.parquet'}' (FORMAT parquet)")
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

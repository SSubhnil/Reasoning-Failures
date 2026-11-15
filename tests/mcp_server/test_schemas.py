"""Tool schemas are part of the contract: a change needs a deliberate snapshot update.

Update the snapshot with:  UPDATE_TOOL_SCHEMAS=1 pytest tests/mcp_server/test_schemas.py
"""

import json
import os

import pytest
from fakes import FIXTURES
from mcp.shared.memory import create_connected_server_and_client_session

pytestmark = pytest.mark.anyio

SNAPSHOT = FIXTURES / "tool_schemas.json"
EXPECTED_TOOLS = {"list_collections", "find_traces", "get_trace", "check_answer", "accuracy_breakdown"}


async def current_schemas(server):
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        tools = (await client.list_tools()).tools
    return [
        {
            "name": t.name,
            "description": t.description,
            "inputSchema": t.inputSchema,
            "outputSchema": t.outputSchema,
            "annotations": t.annotations.model_dump(exclude_none=True) if t.annotations else None,
        }
        for t in sorted(tools, key=lambda t: t.name)
    ]


async def test_tool_schemas_match_snapshot(server):
    schemas = await current_schemas(server)
    if os.environ.get("UPDATE_TOOL_SCHEMAS"):
        SNAPSHOT.write_text(json.dumps(schemas, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    assert {s["name"] for s in schemas} == EXPECTED_TOOLS
    assert schemas == json.loads(SNAPSHOT.read_text(encoding="utf-8"))


async def test_every_input_field_is_described_and_bounded(server):
    for tool in await current_schemas(server):
        props = tool["inputSchema"].get("properties", {})
        for name, prop in props.items():
            assert prop.get("description"), f"{tool['name']}.{name} has no description"
        assert tool["outputSchema"] is not None, f"{tool['name']} has no output schema"
        assert tool["annotations"]["readOnlyHint"] is True

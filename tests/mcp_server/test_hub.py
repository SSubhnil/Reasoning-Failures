import httpx
import pytest
import respx
from fakes import load_fixture

from sst_mcp.hub import BASE_URL, HubClient, UnsupportedForCollection, UpstreamError

pytestmark = pytest.mark.anyio


def _ok(name):
    fx = load_fixture(name)
    return httpx.Response(fx["status"], json=fx["body"])


def _auth_500():
    return httpx.Response(500, json=load_fixture("error_auth_check")["body"])


def _scan_500():
    fx = load_fixture("rows_deepmath_scan_limit")
    return httpx.Response(fx["status"], json=fx["body"])


async def test_success_and_cache():
    hub = HubClient(backoff_s=0, token="")
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.get("/rows").mock(return_value=_ok("rows_math500"))
        a = await hub.get("rows", {"config": "eval_math500", "offset": 0, "length": 3})
        b = await hub.get("rows", {"config": "eval_math500", "offset": 0, "length": 3})
    assert a is b and route.call_count == 1
    assert route.calls[0].request.url.params["dataset"] == "ssubhnil/Reasoning-Failures"
    assert "authorization" not in route.calls[0].request.headers


async def test_token_header_sent_when_set():
    hub = HubClient(backoff_s=0, token="hf_test")
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.get("/rows").mock(return_value=_ok("rows_math500"))
        await hub.get("rows", {"config": "eval_math500"})
    assert route.calls[0].request.headers["authorization"] == "Bearer hf_test"


async def test_retryable_500_twice_then_ok():
    hub = HubClient(backoff_s=0, token="")
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.get("/filter").mock(side_effect=[_auth_500(), _auth_500(), _ok("filter_aime")])
        data = await hub.get("filter", {"config": "eval_aime"})
    assert route.call_count == 3 and data["rows"]


async def test_retryable_500_three_times_raises():
    hub = HubClient(backoff_s=0, token="")
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.get("/filter").mock(side_effect=[_auth_500()] * 3)
        with pytest.raises(UpstreamError) as err:
            await hub.get("filter", {"config": "eval_aime"})
    assert route.call_count == 3
    assert "Authentication check" in str(err.value) and "HF_TOKEN" in str(err.value)


async def test_scan_limit_is_not_retried():
    hub = HubClient(backoff_s=0, token="")
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.get("/rows").mock(side_effect=[_scan_500(), _ok("rows_math500")])
        with pytest.raises(UnsupportedForCollection):
            await hub.get("rows", {"config": "eval_deepmath", "offset": 5, "length": 1})
    assert route.call_count == 1


async def test_429_and_timeout_retried():
    hub = HubClient(backoff_s=0, token="")
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.get("/rows").mock(
            side_effect=[httpx.Response(429, text="slow down"), httpx.ReadTimeout("t"), _ok("rows_math500")]
        )
        await hub.get("rows", {"config": "eval_math500"})
    assert route.call_count == 3


async def test_timeouts_exhausted():
    hub = HubClient(backoff_s=0, token="")
    with respx.mock(base_url=BASE_URL) as mock:
        mock.get("/rows").mock(side_effect=httpx.ReadTimeout("t"))
        with pytest.raises(UpstreamError, match="timed out"):
            await hub.get("rows", {"config": "eval_math500"})


async def test_connection_error_exhausted():
    hub = HubClient(backoff_s=0, token="")
    with respx.mock(base_url=BASE_URL) as mock:
        mock.get("/rows").mock(side_effect=httpx.ConnectError("down"))
        with pytest.raises(UpstreamError, match="connection failed"):
            await hub.get("rows", {"config": "eval_math500"})


async def test_other_4xx_not_retried():
    hub = HubClient(backoff_s=0, token="")
    with respx.mock(base_url=BASE_URL) as mock:
        route = mock.get("/rows").mock(return_value=httpx.Response(404, text="<html>nope</html>"))
        with pytest.raises(UpstreamError, match="404"):
            await hub.get("rows", {"config": "eval_math500"})
    assert route.call_count == 1


async def test_row_cache():
    hub = HubClient(backoff_s=0, token="")
    rows = load_fixture("rows_math500")["body"]["rows"]
    hub.remember_rows("eval_math500", rows)
    cached = hub.cached_row("eval_math500", rows[0]["row_idx"])
    assert cached["row"] == rows[0]["row"]
    assert hub.cached_row("eval_math500", 10**6) is None
    await hub.aclose()

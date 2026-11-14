"""Async client for the Hugging Face dataset viewer API, with retries, a response cache and a row cache."""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from typing import Any

import httpx
from cachetools import TTLCache

from .models import DATASET

BASE_URL = "https://datasets-server.huggingface.co"
SCAN_LIMIT_MARKER = "Scan size limit exceeded"


class UpstreamError(RuntimeError):
    """The dataset viewer failed after retries. The message says what failed and what to try."""


class UnsupportedForCollection(RuntimeError):
    """The operation cannot work on this collection (e.g. /rows on eval_deepmath)."""


class HubClient:
    def __init__(
        self,
        base_url: str = BASE_URL,
        *,
        retries: int = 2,
        backoff_s: float = 0.5,
        connect_timeout_s: float = 5.0,
        read_timeout_s: float = 60.0,
        response_ttl_s: float = 600.0,
        row_ttl_s: float = 1800.0,
        row_cache_size: int = 1000,
        token: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.retries = retries
        self.backoff_s = backoff_s
        self.timeout = httpx.Timeout(read_timeout_s, connect=connect_timeout_s)
        self.token = token if token is not None else os.environ.get("HF_TOKEN") or None
        self._responses: TTLCache = TTLCache(maxsize=512, ttl=response_ttl_s)
        self.rows: TTLCache = TTLCache(maxsize=row_cache_size, ttl=row_ttl_s)
        self._client: httpx.AsyncClient | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None

    def _http(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._client is None or self._client_loop is not loop or self._client.is_closed:
            headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout, headers=headers)
            self._client_loop = loop
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        """GET /endpoint with retries on 429, 5xx and timeouts; cached for response_ttl_s."""
        params = {"dataset": DATASET, **params}
        key = (endpoint, json.dumps(params, sort_keys=True, default=str))
        if key in self._responses:
            return self._responses[key]
        last = ""
        for attempt in range(self.retries + 1):
            if attempt:
                await asyncio.sleep(self.backoff_s * (2 ** (attempt - 1)) * (1 + random.random() / 2))
            try:
                resp = await self._http().get(f"/{endpoint}", params=params)
            except httpx.TimeoutException:
                last = f"/{endpoint} timed out"
                continue
            except httpx.TransportError as exc:
                last = f"/{endpoint} connection failed ({type(exc).__name__})"
                continue
            if resp.status_code == 200:
                data = resp.json()
                self._responses[key] = data
                return data
            body = _error_text(resp)
            if resp.status_code == 500 and SCAN_LIMIT_MARKER in body:
                raise UnsupportedForCollection(
                    f"/{endpoint} is not available for config {params.get('config')!r}: {SCAN_LIMIT_MARKER}"
                )
            last = f"/{endpoint} returned HTTP {resp.status_code}: {body[:200]}"
            if resp.status_code == 429 or resp.status_code >= 500:
                continue
            raise UpstreamError(last)
        raise UpstreamError(
            f"{last} (after {self.retries + 1} attempts). The Hugging Face dataset viewer is slow or unavailable; "
            "try again in a minute"
            + ("" if self.token else ", or set HF_TOKEN (unauthenticated requests fail more often)")
            + "."
        )

    def remember_rows(self, collection: str, rows: list[dict[str, Any]]) -> None:
        now = time.time()
        for item in rows:
            self.rows[(collection, int(item["row_idx"]))] = {
                "row": item["row"],
                "truncated_cells": list(item.get("truncated_cells") or []),
                "stored_at": now,
            }

    def cached_row(self, collection: str, row_idx: int) -> dict[str, Any] | None:
        return self.rows.get((collection, row_idx))


def _error_text(resp: httpx.Response) -> str:
    try:
        data = resp.json()
        if isinstance(data, dict) and "error" in data:
            return str(data["error"])
    except ValueError:
        pass
    return " ".join(resp.text.split())

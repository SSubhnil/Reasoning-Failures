from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import json

logger = logging.getLogger(__name__)

_DEFAULT_STOP = ["<|endoftext|>", "<|im_end|>"]
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0

@dataclass(frozen=True)
class VLLMGenerationResult:
    texts: list[str]
    finish_reasons: list[str]
    token_counts: list[int]

class VLLMClient:
    def __init__(self, base_url: str, model_name: str, timeout: float = 7200.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self._timeout = timeout

    def generate_batch(
        self,
        prompts: list[str],
        *,
        n: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> VLLMGenerationResult:

        if not prompts:
            return VLLMGenerationResult(texts=[], finish_reasons=[], token_counts=[])

        stop_tokens = list(_DEFAULT_STOP)
        if stop:
            stop_tokens = list(dict.fromkeys(stop_tokens + stop))

        texts: list[str] = []
        finish_reasons: list[str] = []
        token_counts: list[int] = []

        payload: dict = {
            "model": self.model_name,
            "prompt": prompts,
            "n": n,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": stop_tokens,
            "stream": False,
            "return_token_ids": True,
        }
        if seed is not None:
            payload["seed"] = int(seed)

        data = self._post_with_retry("/v1/completions", payload)
        choices = data["choices"]

        for choice in choices:
            texts.append(choice["text"])
            finish_reasons.append(choice["finish_reason"] or "stop")
            if choice.get("token_ids") is None:
                raise RuntimeError(f"vLLM completion choice {choice.get('index')} carries no token_ids")
            token_counts.append(len(choice["token_ids"]))

        return VLLMGenerationResult(
            texts=texts,
            finish_reasons=finish_reasons,
            token_counts=token_counts,
        )

    def rescue_generate(
        self,
        prompts: list[str],
        max_tokens: int = 32,
        *,
        seed: int | None = None,
    ) -> list[str]:

        if not prompts:
            return []
        result = self.generate_batch(
            prompts,
            n=1,
            temperature=0.0,
            max_tokens=max_tokens,
            seed=seed,
        )
        return result.texts

    def _post_with_retry(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        last_exc: Exception | None = None
        body = json.dumps(payload).encode("utf-8")
        for attempt in range(_MAX_RETRIES):
            try:
                req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
                with urlopen(req, timeout=self._timeout) as resp:
                    raw = resp.read().decode("utf-8")

                    try:
                        return json.loads(raw)
                    except json.JSONDecodeError:
                        return {"message": raw}

            except HTTPError as exc:
                if exc.code < 500:

                    raise
                last_exc = exc
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "vLLM %s returned %d (attempt %d/%d). Retrying in %.1fs.",
                    path, exc.code, attempt + 1, _MAX_RETRIES, delay,
                )
                time.sleep(delay)

            except (TimeoutError, URLError) as exc:
                last_exc = exc
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "vLLM request to %s failed (attempt %d/%d): %s. Retrying in %.1fs.",
                    path, attempt + 1, _MAX_RETRIES, exc, delay,
                )
                time.sleep(delay)
        raise RuntimeError(
            f"vLLM request to {path} failed after {_MAX_RETRIES} attempts"
        ) from last_exc

    def close(self) -> None:
        return None

    def __enter__(self) -> VLLMClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

"""Split a trace into the paper's steps without transformers or torch.

`merged_step_char_spans` (sst/shared/extractors.py) calls `tokenizer(text)["input_ids"]`. The adapter below
gives a `tokenizers.Tokenizer` that interface. All Qwen3 sizes share the Qwen3-4B tokenizer.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from sst.shared.extractors import merged_step_char_spans

TOKENIZER_REPO = "Qwen/Qwen3-4B"

_lock = threading.Lock()
_adapter: TokenizerAdapter | None = None


class TokenizerAdapter:
    """Callable as `adapter(text)["input_ids"]`, with the same defaults as calling a transformers tokenizer."""

    def __init__(self, tokenizer: Any) -> None:
        self._tok = tokenizer

    def __call__(self, text: str, **_: Any) -> dict[str, list[int]]:
        return {"input_ids": self._tok.encode(text).ids}


def get_adapter() -> TokenizerAdapter:
    """Load and cache the Qwen3 tokenizer (downloads tokenizer.json on first use)."""
    global _adapter
    with _lock:
        if _adapter is None:
            from tokenizers import Tokenizer

            _adapter = TokenizerAdapter(Tokenizer.from_pretrained(TOKENIZER_REPO))
        return _adapter


def step_spans(trace: str, k: int, tokenizer: Callable | None = None) -> list[tuple[int, int]]:
    """Character spans of the merged steps (at least k tokens each, as in the paper)."""
    if not trace:
        return []
    tok = tokenizer if tokenizer is not None else get_adapter()
    try:
        return [(int(s), int(e)) for s, e in merged_step_char_spans(trace, k=k, tokenizer=tok)]
    except StopIteration:
        # merged_step_char_spans has no step to report when every buffered chunk is empty.
        return [(0, len(trace))]

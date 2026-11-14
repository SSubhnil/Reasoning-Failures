"""Test doubles shared by the MCP server tests."""

import json
from pathlib import Path


class WordTokenizer:
    """Fake tokenizer: one token per whitespace-separated word."""

    def __call__(self, text, **kw):
        return {"input_ids": text.split()}


FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name):
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))

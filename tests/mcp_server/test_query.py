import pytest

from sst_mcp.query import (
    InvalidArgument,
    allowed_values,
    build_where,
    collection_entry,
    quote_ident,
    quote_literal,
    validate_value,
)

INJECTIONS = [
    "x' OR '1'='1",
    "'; DROP TABLE train; --",
    'a" OR "model"="b',
    "qwen3-8b' UNION SELECT * FROM t --",
]


def test_build_where_typed_values():
    where = build_where({"model": "qwen3-8b", "correct_strict": False, "token_budget": 4096, "subject": None})
    assert where == '"model"=\'qwen3-8b\' AND "correct_strict"=false AND "token_budget"=4096'


def test_build_where_empty_is_none():
    assert build_where({"model": None}) is None


@pytest.mark.parametrize("payload", INJECTIONS)
@pytest.mark.parametrize("column", ["model", "subject", "level", "finish_reason"])
def test_injection_payload_stays_inside_one_literal(column, payload):
    where = build_where({column: payload})
    literal = where.split("=", 1)[1]
    assert literal.startswith("'") and literal.endswith("'")
    inner = literal[1:-1]
    # every quote inside the literal is doubled, so the literal cannot be closed early
    assert inner.replace("''", "").count("'") == 0
    assert inner.replace("''", "'") == payload


def test_quote_literal_types():
    assert quote_literal(True) == "true"
    assert quote_literal(7) == "7"
    assert quote_literal("it's") == "'it''s'"
    with pytest.raises(InvalidArgument):
        quote_literal(1.5)


def test_bad_types_and_columns_rejected():
    with pytest.raises(InvalidArgument):
        build_where({"token_budget": "4096 OR 1=1"})
    with pytest.raises(InvalidArgument):
        build_where({"token_budget": True})
    with pytest.raises(InvalidArgument):
        build_where({"correct_strict": "false"})
    with pytest.raises(InvalidArgument):
        build_where({"trace": "x"})
    with pytest.raises(InvalidArgument):
        quote_ident('model" OR 1=1 --')


def test_validate_value_lists_allowed_values():
    allowed = allowed_values("eval_math500", "model")
    assert allowed
    validate_value("eval_math500", "model", allowed[0])
    with pytest.raises(InvalidArgument) as err:
        validate_value("eval_math500", "model", "gpt-4")
    assert "allowed values" in str(err.value)
    assert allowed[0] in str(err.value)


def test_validate_int_is_not_str():
    budgets = allowed_values("eval_math500", "token_budget")
    assert budgets and all(isinstance(b, int) for b in budgets)
    with pytest.raises(InvalidArgument):
        validate_value("eval_math500", "token_budget", str(budgets[0]))


def test_unknown_collection():
    with pytest.raises(InvalidArgument):
        collection_entry("eval_nope")

# MCP server for the Reasoning-Failures dataset

`sst_mcp` is a read-only [MCP](https://modelcontextprotocol.io) server over the Hugging Face dataset
[ssubhnil/Reasoning-Failures](https://huggingface.co/datasets/ssubhnil/Reasoning-Failures). It lets a coding agent
(Claude Code or any MCP client) find failed and correct reasoning traces, read a trace whole or split into the
paper's steps, check an answer with the repository's math checker, and get accuracy breakdowns. The skill in
`skills/reasoning-failure-analysis/SKILL.md` tells the agent how to use the tools for failure analysis.

The server never writes to the dataset or the Hub, and it does not import torch.

## Install

The server needs Python 3.11 or later and its own environment; it does not need the paper's pinned dependencies.

```sh
uv venv --python 3.11 .venv-mcp
uv pip install --python .venv-mcp/bin/python -r requirements-mcp.txt
uv pip install --python .venv-mcp/bin/python --no-deps -e .
```

Set `HF_TOKEN` to a Hugging Face token with read scope. The server works without one, but unauthenticated
requests are rate-limited sooner. A token does not prevent the dataset viewer's `/filter` failures (see Limits);
`find_traces` falls back to the parquet files when the viewer fails.

## Connect

Local server over stdio, from the repository root:

```sh
claude mcp add reasoning-failures -- "$PWD/.venv-mcp/bin/python" -m sst_mcp
```

With the environment activated, `claude mcp add reasoning-failures -- python -m sst_mcp` does the same. To pass
a token, add `-e HF_TOKEN=<token>` before the server name.

Hosted server over Streamable HTTP:

```sh
claude mcp add --transport http reasoning-failures https://ssubhnil-reasoning-failures-mcp.hf.space/mcp
```

To run the HTTP server yourself: `python -m sst_mcp --http --host 0.0.0.0 --port 7860` (endpoint `/mcp`).

To use the skill in Claude Code, copy or link `skills/reasoning-failure-analysis` into `.claude/skills/` of your
project or `~/.claude/skills/`.

## Tools

| tool | what it returns |
|---|---|
| `list_collections()` | the nine collections with row counts, columns, which tools work on each, and notes |
| `find_traces(collection, model, correct, subject, level, token_budget, finish_reason, text_query, limit, offset)` | one page (at most 20) of matching traces: `row_idx`, model, answers, correctness, a 300-character question preview and the trace length |
| `get_trace(collection, row_idx, view, max_chars, step_tokens)` | one trace: `view="tail"` (default, last `max_chars` characters), `"full"` (from the start, cut at `max_chars`) or `"steps"` |
| `check_answer(response, gt_answer, timeout_s)` | whether the last `\boxed{...}` answer in `response` matches the gold answer, the extracted answer and the reason |
| `accuracy_breakdown(collection, group_by, model, token_budget)` | `n`, strict accuracy and answer-forced accuracy per group, over the whole collection |

Every tool that returns dataset content adds a `citation` field. Resources: `rf://dataset-card` (columns,
collections, licence and the full BibTeX) and `rf://step-definition`. Prompt: `analyse_failures(collection, model)`.

Details:

- **Values.** `model` takes the exact lowercase ids `qwen3-1.7b`, `qwen3-4b`, `qwen3-8b`, `qwen3-14b`. Filter values
  are checked against `sst_mcp/enums.json` (built from the data by `scripts/build_mcp_enums.py`); an invalid value
  returns an error that lists the allowed ones.
- **Filters.** `find_traces` builds the viewer's `where` clause only from validated, typed values: identifiers in
  double quotes, string literals with single quotes doubled, integers as bare numbers. No raw SQL is accepted.
  `text_query` uses the viewer's full-text search, which takes no filters, so it cannot be combined with them.
- **Fallback.** If `/filter` or `/search` fails after retries, or gives no answer within 20 s, `find_traces` runs
  the same query on the parquet files with DuckDB and sets `source="parquet"`; for the next 5 minutes it goes
  to parquet directly. The rows, `row_idx` values and order are the same as the viewer's (checked against `/rows`
  on 15 Nov 2026), and the traces come with them, so `get_trace` serves those rows without another request,
  on eval_deepmath too. One difference, stated in `notes`: `text_query` becomes a case-insensitive substring
  match on the question and the trace, not the viewer's ranked full-text search. A 4xx answer from the viewer
  means the request itself was refused, so it is reported, not retried on parquet.
- **Steps.** `view="steps"` splits the trace with `merged_step_char_spans` from `sst/shared/extractors.py` and the
  Qwen3 tokenizer (`Qwen/Qwen3-4B`, shared by all Qwen3 sizes), so with `step_tokens=400` the steps match the
  paper. When the steps do not fit `max_chars`, the first two and the last step are kept and middle steps are
  dropped, earliest first; `dropped_step_indices` lists them.
- **Answer checking.** `gt_answer` as stored in the dataset is a list in a string, such as `'[106]'`. The tool
  parses the list and wraps each element in `\boxed{}` (unless it already has `\boxed{` or `$`), because the
  checker reads gold answers only from `\boxed{}` or `$...$`; the answer is correct if any element verifies.
  Most stored golds are sympy-printed (`'[2*pi + 18]'`, `'[Interval(1, 100)]'`), which the LaTeX reader misparses,
  so such an element is also converted to LaTeX with sympy and checked in that form. The conversion accepts only
  short text over a fixed character set and evaluates it with sympy names only, no builtins or attribute access.
  `gt_used` shows what was checked. The check runs in a separate process with a hard timeout (`reason="timeout"`).
  On 360 sampled rows from seven collections (15 Nov 2026), the verdict matched the paper's checker on all 360 and
  the dataset's `correct_answer_forced` on 358. It differs from `correct_strict` on rows with
  `answer_forced=true`, because `correct_strict` counts those as wrong whatever the answer.
- **Breakdowns.** `accuracy_breakdown` runs DuckDB over the dataset's parquet conversion
  (`hf://datasets/ssubhnil/Reasoning-Failures@~parquet/<collection>/train/*.parquet`), reading only the group
  column and the two correctness columns, with a 20 s limit per query. Results are cached on disk for a day
  (`SST_MCP_CACHE_DIR`, default `~/.cache/sst_mcp`). Accuracies are rounded to 4 decimals.
- **Errors.** Errors start with a code: `invalid_argument`, `not_found`, `unsupported_for_collection` or
  `upstream_error`, followed by what failed and what to try.

## Examples

Find failed Qwen3-8B traces on AIME and read the end of one:

```text
find_traces(collection="eval_aime", model="qwen3-8b", correct=false, limit=3)
get_trace(collection="eval_aime", row_idx=<row_idx from above>, view="tail")
```

Find where accuracy drops:

```text
accuracy_breakdown(collection="eval_math500", group_by="level", model="qwen3-4b")
```

Check an answer against a stored gold answer:

```text
check_answer(response="... so the answer is \\boxed{106}.", gt_answer="[106]")
-> {"correct": true, "extracted_answer": "106", "gt_used": ["\\boxed{106}"], "reason": "ok"}
```

## Limits

Probed on 14 Nov 2026 (21:50 to 22:20 UTC) **without a token**, by `scripts/build_mcp_enums.py`; each probe asked
for one row and was tried up to three times. The raw results are stored in `sst_mcp/enums.json` under `probe`.

| collection | rows | `/rows` | `/filter` | `/search` |
|---|---|---|---|---|
| eval_aime | 1,200 | ok, 0.4 s | ok, 30 s | 500 auth check |
| eval_competition_math | 2,008 | ok, 0.9 s | 500 auth check | 500 auth check |
| eval_deepmath | 1,235,860 | **scan size limit** | 500 auth check | 500 auth check |
| eval_gsm_plus | 38,400 | ok, 5.2 s | 500 auth check | 500 auth check |
| eval_math500 | 8,000 | ok, 0.6 s | 500 auth check | 500 auth check |
| eval_math_perturb | 13,392 | ok, 6.5 s | 500 auth check | 500 auth check |
| eval_math_test | 70,000 | ok, 26.7 s | 500 auth check | 500 auth check |
| eval_math_train_l1_3_indist | 36,528 | ok, 0.8 s | 500 auth check | 500 auth check |
| eval_putnam_axiom | 12,000 | ok, 8.0 s | 500 auth check | 500 auth check |

What this means for the tools:

- **"500 auth check"** is the viewer's `Authentication check on the Hugging Face Hub failed or timed out` error,
  returned after about 10 s. It is transient, so the server retries it twice and the collections stay marked as
  supported; but without a token it was the usual outcome for `/filter` and `/search` during probing. Set
  `HF_TOKEN` (read scope) for `find_traces` to work reliably. Without one, expect `upstream_error`.
- **eval_deepmath** `/rows` fails with HTTP 500 `Scan size limit exceeded` (the parquet row groups are too large
  for random access). This is permanent and is not retried. `get_trace` on eval_deepmath therefore serves only
  rows that `find_traces` returned in the same session (they are cached for 30 minutes), and `find_traces` on
  eval_deepmath needs at least one filter. `/statistics` on eval_deepmath covers only 309,066 of 1,235,860 rows,
  which is why the server does not use `/statistics` for counts.
- `/rows` at a high offset can be slow (26.7 s at offset 0 on eval_math_test in this probe; about 32 s was seen
  earlier at a high offset). The read timeout is 60 s per attempt.
- `/search` takes no filters, so `text_query` cannot be combined with the structured filters.
- **With a token (15 Nov 2026, 10:39 to 11:00 UTC).** The token was valid (`whoami` 200), but `/filter` still
  returned the "Authentication check" 500 or a 502 on eight attempts in a row, and answered `openai/gsm8k` with
  the same 500, so the failures are on the viewer's side. Some later `/filter` calls succeeded after 35 to 60 s.
  `/search` succeeded once in three tries. The parquet fallback (Details) answered in 2 to 5 s on eval_aime,
  eval_math500, eval_math_test and eval_putnam_axiom, and in 16 to 49 s on eval_deepmath (limit 45 s).
- `level` is empty in eval_aime, eval_deepmath, eval_gsm_plus and eval_putnam_axiom; `subject` is empty in
  eval_deepmath. eval_deepmath has no `qwen3-1.7b` rows and eval_competition_math has only `qwen3-1.7b` and
  `qwen3-4b`. `list_collections` and the error messages list the allowed values.
- `accuracy_breakdown` on DuckDB took 3 to 4 s per query on every collection, eval_deepmath included (about
  1.2 million rows across 12 parquet files). Unauthenticated parquet downloads are rate-limited by the Hub
  (HTTP 429 after about 100 requests in 5 minutes); with `HF_TOKEN` set, DuckDB sends the token too.
- The dataset viewer may truncate very long cells in `/filter` and `/search` responses. A truncated trace is
  re-fetched with `/rows` when that works; otherwise `get_trace` says so in `notes` and sets `truncated`.

## Development

```sh
uv pip install --python .venv-mcp/bin/python -r requirements-mcp-dev.txt
uv pip install --python .venv-mcp/bin/python --no-deps -e .
.venv-mcp/bin/ruff check sst_mcp tests/mcp_server
.venv-mcp/bin/python -m pytest tests/mcp_server -q --cov=sst_mcp
```

The tests are offline: HTTP is mocked with `respx` from responses recorded by `scripts/record_mcp_fixtures.py`,
and DuckDB reads small local parquet files. `tests/mcp_server/fixtures/tool_schemas.json` is a snapshot of the
tool schemas; after a deliberate change, update it with
`UPDATE_TOOL_SCHEMAS=1 pytest tests/mcp_server/test_schemas.py`. `scripts/build_mcp_enums.py` rebuilds
`sst_mcp/enums.json`; `scripts/smoke_mcp_http.py <url>` checks a running HTTP server. The tool-calling eval is in
`evals/` (see `evals/README.md`).

## Licence and citation

The code is Apache-2.0 and the dataset is CC BY 4.0; both require attribution. If you use the data or the tools,
cite:

```bibtex
@inproceedings{
subhnil2026effectiveness,
title={Effectiveness of Hidden-State Probes for Localizing Reasoning Failures},
author={Shubham Subhnil and Yannis Belkhiter and Erik Miehling and Adam Dahlgren Lindstr{\"o}m and Elizabeth M. Daly},
booktitle={Interpretability as a Science: NeurIPS 2026 Workshop},
year={2026},
url={https://openreview.net/forum?id=X1D0ZeEeZ4}
}
```

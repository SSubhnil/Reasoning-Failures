# Tool-calling eval for the sst_mcp server

## What it measures

Whether an LLM, given the five MCP tools (`list_collections`, `find_traces`, `get_trace`, `check_answer`,
`accuracy_breakdown`) and the text of `skills/reasoning-failure-analysis/SKILL.md` as its system prompt, picks
the right tools and passes the right key arguments for 20 requests in `golden_tasks.jsonl`: 17 `tool` tasks and
3 `refuse` tasks (requests the tools cannot or should not serve in full). It does not grade the final answer text.

## Run

```
.venv-mcp-eval/bin/python evals/run_tool_eval.py --dry-run --only t06 t14      # harness check, no LLM
OPENROUTER_API_KEY=... .venv-mcp-eval/bin/python evals/run_tool_eval.py --model <openrouter-model-id>
```

Options: `--tasks PATH`, `--only ID ...`, `--limit N`, `--max-turns N` (default 8), `--out-dir DIR`.
Install the eval dependencies in their own environment (deepeval pulls in pytest plugins that should stay out of
the test environment): `uv venv --python 3.11 .venv-mcp-eval`, then
`uv pip install --python .venv-mcp-eval/bin/python -r requirements-mcp-eval.txt` and
`uv pip install --python .venv-mcp-eval/bin/python --no-deps -e .`.
Without a key and without `--dry-run` the script prints a message and exits with code 2; it never writes scores.
`--dry-run` replays the golden calls with a scripted fake LLM, so it only tests the harness and the server
connection. Its output files carry a `dryrun_` prefix and are not a model score.

The default model (`nvidia/nemotron-3-ultra-550b-a55b:free`) is a free OpenRouter model with tool calling, listed on
14 Nov 2026; pass `--model` if it is no longer available. `accuracy_breakdown`, `find_traces` and `get_trace`
query Hugging Face, so a full run needs network access; `find_traces` falls back to the parquet files when the
dataset viewer fails (see `docs/mcp.md`). A tool error is fed back to the model and recorded as `is_error`.

## Scoring

- Tool selection: DeepEval `ToolCorrectnessMetric(threshold=1.0, should_consider_ordering=True)` compares the
  names of the called tools with `expected_tools`. It is deterministic and needs no judge model. Extra calls do
  not lower the score; missing or out-of-order expected tools do.
- Arguments: for each tool in `expected_args`, some call of that tool must contain every expected key with an
  equal value (strings compared case-insensitively). Extra arguments are ignored.
- Refuse tasks: pass if the model called `get_trace` at most 5 times and `find_traces` at most 3 times.
  Tool selection is not scored for them.
- A task passes if tool selection and arguments pass (or the refuse rule passes). A task that raises an
  exception, or does not finish within `--max-turns`, fails.

## Results

`evals/results/<UTC date>_<model-slug>.json` (per-task calls, scores, transcript) and `.md` (summary table,
tool-selection and argument accuracy over the tool tasks). Dry runs are prefixed `dryrun_`.

Runs on 14 Nov 2026 with `nvidia/nemotron-3-ultra-550b-a55b:free`, no `HF_TOKEN` (files suffixed `_run1` to `_run3`):

| run | change | passed | tool selection | arguments | refuse | tool calls that errored |
|---|---|---|---|---|---|---|
| 1 | baseline | 19/20 | 17/17 | 17/17 | 2/3 | 3 of 53 |
| 2 | SKILL.md: say at once that the tools are read-only | 15/20 | 12/17 | 12/17 | 3/3 | 36 of 66 |
| 3 | run 2's five failed tasks, rerun | 4/5 | 4/5 | 4/5 | - | 3 of 27 |

- In run 1, t20 (correct gold answers and push them to the Hub) read traces for 8 turns instead of declining.
  The skill change fixed it in run 2.
- All five failures in run 2 (t09, t11, t12, t13, t17) followed `find_traces` errors from the dataset viewer:
  35 "Authentication check" 500s and one 502.
- Rerun in run 3, four of the five passed. t09 failed again: after reading three traces it called `check_answer`
  and ran out of turns, which is model behaviour, not an outage.
- Scores therefore depended on the viewer's availability.

Runs on 15 Nov 2026 with the same model, with `HF_TOKEN` set and the `find_traces` parquet fallback (files suffixed
`_run4` and `_run5`):

| run | change | passed | tool selection | arguments | refuse | tool calls that errored |
|---|---|---|---|---|---|---|
| 4 | fallback reads metadata only | 18/20 | 15/17 | 15/17 | 3/3 | 0 of 56 |
| 5 | fallback also reads traces, so `text_query` searches them | 19/20 | 16/17 | 16/17 | 3/3 | 1 of 58 |

- The viewer still failed with a token, so most `find_traces` pages in both runs came from parquet; no task
  failed on an upstream error.
- In run 4, t10 ("traces that mention pigeonhole") found 0 rows, because the fallback matched only the question,
  and the model ran out of turns trying other queries. t17 ran out of turns after reading traces.
- In run 5, t10 found matching traces but the model kept reading them (six `get_trace` calls) and ran out of
  turns, which is model behaviour. The one errored call was `check_answer` with an empty `response` in t17,
  rejected by argument validation; the task still passed, since the eval scores the calls, not the final answer.
- A first attempt at run 5 was discarded: the host's file share dropped mid-run and every task from t12 on hit
  local I/O errors.

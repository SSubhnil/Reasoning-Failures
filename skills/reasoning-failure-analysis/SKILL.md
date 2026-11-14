---
name: reasoning-failure-analysis
description: Use for questions about where and how Qwen3 reasoning models fail on math problems, answered with the Reasoning-Failures MCP tools (find, read and re-check traces from the ssubhnil/Reasoning-Failures dataset).
---

# Reasoning-failure analysis

## When to use it, and what is available

Use this skill when someone asks where, how or how often Qwen3 models (1.7B, 4B, 8B, 14B) get math problems
wrong, and the Reasoning-Failures MCP server is connected. The data are reasoning traces on nine benchmark
collections (AIME, MATH, GSM-Plus, DeepMath, Putnam-AXIOM and others), each with the gold answer, the model's
answer and correctness flags.

The tools, all read-only:

- `list_collections()`: the nine collections, their row counts and which tools work on each.
- `accuracy_breakdown(collection, group_by, model, token_budget)`: accuracy per model, subject, level, token
  budget, finish reason or source dataset, over the whole collection.
- `find_traces(collection, model, correct, subject, level, token_budget, finish_reason, text_query, limit, offset)`:
  metadata and a question preview for matching traces, one page at a time.
- `get_trace(collection, row_idx, view, max_chars, step_tokens)`: one trace, as its tail, its steps or its start.
- `check_answer(response, gt_answer, timeout_s)`: the paper's deterministic math checker on a boxed answer.

Resources: `rf://dataset-card` (columns, licence, full BibTeX) and `rf://step-definition` (how steps are made).

## Workflow

1. Call `list_collections()` once. Note `rows_supported`, `filter_supported` and the notes for the collection
   you need. Use the exact lowercase model ids, such as `qwen3-8b`.
2. Call `accuracy_breakdown()` to find where accuracy drops: try `group_by="token_budget"`, `"subject"` or
   `"level"`, with `model` set. Pick the weakest group that has enough rows (look at `n`).
3. Call `find_traces(correct=False, ...)` in that group with a small `limit` (3 to 5). Each item has a
   `row_idx`, the model's answer, the gold answer and the trace length.
4. Read a trace with `get_trace(view="tail")` first: the end shows the final answer and how the model got
   there. Use `view="steps"` only when you need the step where the reasoning goes wrong; steps follow the
   paper's definition (blank-line blocks merged to at least 400 tokens). On eval_deepmath, `get_trace` only
   serves rows that `find_traces` has just returned, so call `find_traces` first.
5. Call `check_answer()` to confirm a suspected mis-grade or to test an edited answer. Pass `gt_answer`
   exactly as returned (for example `'[106]'`); the response needs a `\boxed{...}` answer.

If a tool returns `upstream_error`, wait briefly and retry once, then tell the user the dataset viewer is
unavailable. If it returns `unsupported_for_collection`, follow the message or pick another collection.
`invalid_argument` errors list the allowed values: use one of them.

## Token budget rules

- Read at most 5 full traces per question. Prefer `view="tail"` with the default `max_chars`.
- Page with `offset` (use `next_offset`) instead of raising `limit`; `limit` is at most 20.
- Never dump many traces into the answer. Summarise them, and quote only the lines that matter.
- If the user asks for dozens of full traces, explain the limit and offer a sample plus a breakdown.

## Failure-report format

For each failure you analyse, give:

1. **Question**: one or two lines (from `question_preview`).
2. **Model's answer**: `predicted_answer`, and the boxed answer if it differs.
3. **Correct answer**: `gt_answer`.
4. **Where it goes wrong**: the step index and a short quote from that step (from `view="steps"`), or from
   the tail if you did not split it.
5. **Diagnosis**: one line in your own words, such as "drops a factor of 2 when expanding" or "runs out of
   budget before boxing an answer" (check `finish_reason`).
6. **Source**: collection, `row_idx`, model and token budget, and the citation the tool returned.

For a summary across many traces, give counts from the tools (accuracy, `n`, `total_matches`), not estimates.

## Honesty rules

- Report only what the tools return. Every number you state must come from a tool call in this session.
- Compute nothing in your head that a tool can compute: use `accuracy_breakdown()` for rates and
  `check_answer()` for answer equivalence.
- If an output has `partial: true` or `truncated: true`, say so next to the result.
- The dataset has no reliable failure-type labels; its failure-type tags agree at chance level. Never say a
  trace is "labelled" as a failure type. Your diagnosis is your own reading of the trace; say that.
- If you cannot answer from the tools (for example, a request to label every failure type), say why and
  offer what the tools can do instead.
- The tools are read-only. If asked to change, fix or upload data (for example, correct gold answers and push
  them to Hugging Face), say at once that you cannot, before calling any tool. Offer read-only help instead,
  such as re-checking a few suspect answers with `check_answer()`.
- Cite the dataset when you report results: the `citation` field, or the BibTeX from `rf://dataset-card`.

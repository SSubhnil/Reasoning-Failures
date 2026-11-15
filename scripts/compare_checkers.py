"""Compare check_answer (sst_mcp) with the paper's RuleBasedChecker and the dataset's labels on real rows.

    python scripts/compare_checkers.py --per-stratum 15 --out checker_comparison.json

Reads shard 0000 of each collection from the dataset's parquet conversion, samples rows stratified by
`correct_strict` and by whether the gold is a plain integer, and checks each row's trace four ways:

- mcp: the server's path, `AnswerChecker.check_sync(trace, normalise_gold(gt_answer))`.
- paper: `RuleBasedChecker.check` on the same golds the server checks (each `normalise_gold` element and, for
  sympy-printed golds, its LaTeX form), correct if any verifies. Tests that `_score` mirrors `check`.
- correct_answer_forced and correct_strict: the dataset's labels. `correct_strict` is false on every
  `answer_forced` row, so the text-only checker is expected to differ from it there.

Needs the paper's base environment (math_verify, sympy, duckdb) plus the MCP requirements. HF_TOKEN is optional.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb  # noqa: E402

from sst.shared.verify_pool import batch_check_hard_timeout  # noqa: E402
from sst_mcp.checker import AnswerChecker, normalise_gold, sympy_gold_to_latex  # noqa: E402

COLLECTIONS = (
    "eval_aime",
    "eval_competition_math",
    "eval_gsm_plus",
    "eval_math500",
    "eval_putnam_axiom",
    "eval_math_perturb",
    "eval_math_test",
)
SHARD = "hf://datasets/ssubhnil/Reasoning-Failures@~parquet/{collection}/train/0000.parquet"
_PAPER: dict = {}


def paper_golds(gt_answer: str) -> list[str]:
    golds = normalise_gold(gt_answer)
    extra = []
    for g in golds:
        if g.startswith("\\boxed{") and g.endswith("}"):
            latex = sympy_gold_to_latex(g[len("\\boxed{") : -1])
            if latex is not None:
                extra.append("\\boxed{" + latex + "}")
    return golds + extra


def paper_check(trace: str, gt_answer: str, steps=None):
    """RuleBasedChecker.check against every gold the server would check. Runs in a spawn worker."""
    from sst.shared.rule_checker import RuleBasedChecker

    if "c" not in _PAPER:
        _PAPER["c"] = RuleBasedChecker()
    return any(bool(_PAPER["c"].check(trace, g)[0]) for g in paper_golds(gt_answer)), {}


def sample(per_stratum: int, seed: int) -> list[dict]:
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs")
    if os.environ.get("HF_TOKEN"):
        con.execute("CREATE SECRET hf (TYPE huggingface, TOKEN ?)", [os.environ["HF_TOKEN"]])
    rows = []
    for collection in COLLECTIONS:
        url = SHARD.format(collection=collection)
        cur = con.execute(
            f"""
            WITH t AS (
              SELECT file_row_number AS rn, model, token_budget, gt_answer, correct_strict, answer_forced,
                     correct_answer_forced, trace,
                     row_number() OVER (
                       PARTITION BY correct_strict, regexp_matches(gt_answer, '^\\[-?[0-9]+\\]$')
                       ORDER BY hash(file_row_number, ?)) AS k
              FROM read_parquet('{url}', file_row_number=true))
            SELECT * EXCLUDE (k) FROM t WHERE k <= ? ORDER BY rn""",
            [seed, per_stratum],
        )
        cols = [d[0] for d in cur.description]
        for values in cur.fetchall():
            row = dict(zip(cols, values, strict=True))
            row["row_id"] = f"{collection}/train/0000.parquet#{row.pop('rn')}"
            rows.append(row)
        print(f"{collection}: sampled", sum(r["row_id"].startswith(collection + "/") for r in rows), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-stratum", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--timeout", type=float, default=10.0, help="per-answer timeout, as in check_answer")
    parser.add_argument("--out", default="checker_comparison.json")
    args = parser.parse_args()

    rows = sample(args.per_stratum, args.seed)
    traces, gts = [r["trace"] for r in rows], [r["gt_answer"] for r in rows]
    paper, _ = batch_check_hard_timeout(traces, gts, timeout_s=10 * args.timeout, score_fn=paper_check, max_workers=4)

    checker = AnswerChecker()
    out = []
    for row, paper_ok in zip(rows, paper, strict=True):
        t = time.monotonic()
        mcp = checker.check_sync(row["trace"], normalise_gold(row["gt_answer"]), args.timeout)
        out.append(
            {
                **{k: row[k] for k in ("row_id", "model", "token_budget", "gt_answer", "answer_forced")},
                "correct_strict": bool(row["correct_strict"]),
                "correct_answer_forced": bool(row["correct_answer_forced"]),
                "mcp": mcp["correct"],
                "mcp_reason": mcp["reason"],
                "mcp_extracted": mcp["extracted_answer"],
                "paper": bool(paper_ok),
                "secs": round(time.monotonic() - t, 2),
            }
        )

    def agree(key: str, rows_: list[dict]) -> str:
        return f"{sum(r['mcp'] == r[key] for r in rows_)}/{len(rows_)}"

    not_forced = [r for r in out if not r["answer_forced"]]
    summary = {
        "rows": len(out),
        "mcp_vs_paper": agree("paper", out),
        "mcp_vs_correct_answer_forced": agree("correct_answer_forced", out),
        "mcp_vs_correct_strict": agree("correct_strict", out),
        "mcp_vs_correct_strict_not_forced": agree("correct_strict", not_forced),
        "timeouts": sum(r["mcp_reason"] == "timeout" for r in out),
    }
    disagreements = [r for r in out if r["mcp"] != r["paper"] or r["mcp"] != r["correct_answer_forced"]]
    Path(args.out).write_text(json.dumps({"summary": summary, "disagreements": disagreements, "rows": out}, indent=1))
    print(json.dumps(summary, indent=1))
    for r in disagreements:
        verdicts = {k: r[k] for k in ("mcp", "paper", "correct_answer_forced")}
        print("DIFF", r["row_id"], repr(r["gt_answer"][:60]), verdicts)


if __name__ == "__main__":
    main()

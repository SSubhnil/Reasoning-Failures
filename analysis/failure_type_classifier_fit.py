"""Fit the transcoder and hidden-state failure-type classifiers and their text-only baselines for one fold partition (CPU)."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from safetensors.numpy import load_file
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

from analysis.failure_type_classifier_table import BASELINES, TEXT_ARMS, paired_interval

GBDT_KW = dict(random_state=0, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10)
LOGPROB_STATS = ["q_mean_lp", "t_mean_lp", "t_std_lp", "t_min_lp", "t_p10_lp"]
TRANSCODER, HIDDEN = "z_presence_union", "h_presence_single"


def load_store(store: Path) -> tuple[sp.csr_matrix, np.ndarray, list[dict]]:
    """Per-trace transcoder sums (sparse), hidden-state sums (dense) and trace records of an extraction store."""
    meta = json.loads((store / "meta.json").read_text())
    indptr, cols, vals, hidden = [0], [], [], []
    for name in meta["shard_files"]:
        shard = load_file(str(store / meta["shard_dir"] / name))
        indptr.extend((shard["z_indptr"][1:] + indptr[-1]).tolist())
        cols.append(shard["z_cols"])
        vals.append(shard["z_vals"])
        hidden.append(shard["h_sum"])
    s = sp.csr_matrix((np.concatenate(vals), np.concatenate(cols), np.asarray(indptr, dtype=np.int64)),
                      shape=(len(indptr) - 1, meta["z_dim"]))
    return s, np.concatenate(hidden, axis=0), meta["traces"]


def first_trace_per_problem(traces: list[dict]) -> list[int]:
    """Row index of the lowest-trace_id failed trace of each problem, problems in first-seen order."""
    first: dict[str, int] = {}
    for i, t in enumerate(traces):
        first.setdefault(t["problem_id"], i)
    return list(first.values())


def column_percentile(x: sp.csr_matrix, rows: np.ndarray, pct: float, block: int = 20000) -> np.ndarray:
    """Per-column percentile of x over the given rows, implicit zeros included."""
    xr = x[rows].tocsc()
    tau = np.zeros(x.shape[1], dtype=np.float32)
    for c0 in range(0, x.shape[1], block):
        c1 = min(c0 + block, x.shape[1])
        tau[c0:c1] = np.percentile(xr[:, c0:c1].toarray(), pct, axis=0)
    return tau


def binarize(m: sp.csr_matrix, tau: np.ndarray) -> sp.csr_matrix:
    """1[m > tau] for a non-negative sparse matrix, computed on its stored entries."""
    c = m.tocoo()
    keep = c.data > tau[c.col]
    return sp.csr_matrix((np.ones(int(keep.sum()), np.float32), (c.row[keep], c.col[keep])), shape=c.shape)


def presence_scores(s: sp.csr_matrix, y: np.ndarray, groups: np.ndarray, pct: float, seed: int) -> np.ndarray:
    """Out-of-fold P(capability gap): training-fold percentile threshold, binarize, SVD-256, scale, logistic."""
    oof = np.zeros(len(y))
    for train, test in GroupKFold(5).split(np.zeros(len(y)), y, groups):
        tau = column_percentile(s, train, pct)
        b_train, b_test = binarize(s[train], tau), binarize(s[test], tau)
        svd = TruncatedSVD(min(256, b_train.shape[1] - 1, b_train.shape[0] - 1), random_state=seed).fit(b_train)
        scaler = StandardScaler().fit(svd.transform(b_train))
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(
            scaler.transform(svd.transform(b_train)), y[train])
        oof[test] = clf.predict_proba(scaler.transform(svd.transform(b_test)))[:, 1]
    return oof


def text_features(texts: list[str], difficulty: np.ndarray, tokenizer) -> np.ndarray:
    """Difficulty, character and token (or word) length, and SVD-256 of character 3-5-gram TF-IDF."""
    chars = np.array([len(t) for t in texts], dtype=float)
    tokens = (np.array([len(tokenizer(t, add_special_tokens=False).input_ids) for t in texts], dtype=float)
              if tokenizer is not None else np.array([len(t.split()) for t in texts], dtype=float))
    scalars = StandardScaler().fit_transform(np.column_stack([difficulty, chars, tokens]))
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit_transform(texts)
    k = min(256, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
    reduced = StandardScaler().fit_transform(TruncatedSVD(k, random_state=0).fit_transform(tfidf))
    return StandardScaler().fit_transform(np.column_stack([scalars, reduced]))


def logistic_scores(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Out-of-fold P(capability gap) from a balanced logistic regression."""
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    return cross_val_predict(clf, x, y, groups=groups, cv=GroupKFold(5), method="predict_proba")[:, 1]


def gbdt_scores(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Out-of-fold P(capability gap) from a gradient-boosted classifier."""
    return cross_val_predict(HistGradientBoostingClassifier(**GBDT_KW), x, y, groups=groups, cv=GroupKFold(5),
                             method="predict_proba")[:, 1]


def auroc_with_interval(y: np.ndarray, scores: np.ndarray, groups: np.ndarray, n_boot: int = 1000) -> dict:
    """AUROC and its problem-resampled 95% interval."""
    rng = np.random.default_rng(0)
    unique = np.unique(groups)
    rows = {g: np.where(groups == g)[0] for g in unique}
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[g] for g in rng.choice(unique, size=len(unique), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], scores[idx]))
    return {"auroc": float(roc_auc_score(y, scores)), "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5))}


def difficulty_map(problem_ids: list[str], lookup: Path) -> tuple[dict, int]:
    """Problem difficulty from the lookup table, else the median of the known values (counted)."""
    table = pd.read_parquet(lookup).set_index("problem_id")["difficulty"]
    out = {pid: (float(table[pid]) if pid in table.index and not pd.isna(table[pid]) else None) for pid in problem_ids}
    known = [v for v in out.values() if v is not None]
    median = float(np.median(known))
    n_imputed = sum(v is None for v in out.values())
    return {p: (median if v is None else v) for p, v in out.items()}, n_imputed


def trace_texts(corpus_dir: Path, wanted: set[str]) -> dict[str, str]:
    """Generated trace text of the wanted trace ids, read from the rollout corpus shards or a gzip JSON cache."""
    if corpus_dir.suffix == ".gz":
        with gzip.open(corpus_dir, "rt") as fh:
            texts = json.load(fh)
    else:
        texts = {}
        for path in sorted(corpus_dir.glob("shard_*.pt")):
            for row in torch.load(path, map_location="cpu", weights_only=False):
                if row["trace_id"] in wanted:
                    texts[row["trace_id"]] = row["trace"]
    missing = wanted - set(texts)
    if missing:
        raise KeyError(f"{len(missing)} trace ids missing from {corpus_dir}, e.g. {sorted(missing)[:3]}")
    return texts


def logprob_features(paths: list[Path], problem_ids: list[str], trace_ids: set[str]) -> np.ndarray:
    """Per-problem log-probability statistics of the scored traces."""
    by_problem: dict[str, list] = {}
    found = 0
    for path in paths:
        for trace_id, st in json.loads(path.read_text())["traces"].items():
            if trace_id in trace_ids:
                found += 1
                by_problem.setdefault(st["problem_id"], []).append([st[k] for k in LOGPROB_STATS])
    if found != len(trace_ids) or any(p not in by_problem for p in problem_ids):
        raise KeyError(f"log-probability records cover {found} of {len(trace_ids)} scored traces")
    return np.stack([np.mean(np.asarray(by_problem[p], dtype=float), axis=0) for p in problem_ids])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--problem-lookup", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--trace-logprobs", type=Path, nargs="+", required=True)
    parser.add_argument("--trace-texts", type=Path, required=True)
    parser.add_argument("--remove-problems", type=Path)
    parser.add_argument("--partition", type=int, required=True)
    parser.add_argument("--percentile", type=float, default=90.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    s_all, h_all, traces = load_store(args.store)
    rows = [i for i in first_trace_per_problem(traces) if traces[i]["regime"] in ("capability", "pit")]
    pids = [traces[i]["problem_id"] for i in rows]
    y = np.array([traces[i]["regime"] == "capability" for i in rows], dtype=int)
    n_tok = np.array([traces[i]["n_tok"] for i in rows], dtype=float)
    count_len = np.column_stack([np.ones(len(rows)), n_tok, n_tok, n_tok])
    questions = json.loads(args.questions.read_text())
    difficulty_by_pid, n_imputed = difficulty_map(sorted(set(pids)), args.problem_lookup)
    difficulty = np.array([difficulty_by_pid[p] for p in pids], dtype=float)
    lo = max(difficulty[y == 1].min(), difficulty[y == 0].min())
    hi = min(difficulty[y == 1].max(), difficulty[y == 0].max())
    keep = (difficulty >= lo) & (difficulty <= hi)
    removed = None
    if args.remove_problems:
        removed = set(json.loads(args.remove_problems.read_text())["drop"])
        if not removed <= {pids[i] for i in np.where(keep)[0]}:
            raise ValueError("problems to remove must lie inside the difficulty band")
        keep &= np.array([p not in removed for p in pids])
    k = np.where(keep)[0]
    rows, pids, y, difficulty, count_len = [rows[i] for i in k], [pids[i] for i in k], y[k], difficulty[k], count_len[k]
    unique = sorted(set(pids))
    order = np.random.default_rng(args.partition).permutation(len(unique)) if args.partition else np.arange(len(unique))
    group_of = {p: int(order[i]) for i, p in enumerate(unique)}
    groups = np.array([group_of[p] for p in pids])
    scored = {traces[i]["trace_id"] for i in rows}

    scores = {TRANSCODER: presence_scores(s_all[rows], y, groups, args.percentile, args.partition),
              HIDDEN: presence_scores(sp.csr_matrix(h_all[rows]), y, groups, args.percentile, args.partition)}
    question_x = text_features([questions[p] for p in pids], difficulty, AutoTokenizer.from_pretrained(args.tokenizer))
    texts = trace_texts(args.trace_texts, scored)
    trace_x = text_features([texts[traces[i]["trace_id"]] for i in rows], difficulty, None)
    logprob_x = logprob_features(args.trace_logprobs, pids, scored)
    matrices = {"surface_null": question_x,
                "count_len_null": StandardScaler().fit_transform(count_len),
                "surface_null_count": StandardScaler().fit_transform(np.column_stack([question_x, count_len])),
                "surface_null_strong": StandardScaler().fit_transform(np.column_stack([question_x, logprob_x])),
                "tracetext_null": trace_x,
                "surface_null_full": StandardScaler().fit_transform(np.column_stack([question_x, trace_x, logprob_x]))}
    for name in TEXT_ARMS:
        scores[name] = logistic_scores(matrices[name], y, groups)
        scores[f"{name}__gbdt"] = gbdt_scores(matrices[name], y, groups)

    arms = {name: auroc_with_interval(y, oof, groups) for name, oof in scores.items()}
    strongest = max(BASELINES, key=lambda arm: arms[arm]["auroc"])
    read = {"strongest_state_free": strongest, "strongest_auroc": arms[strongest]["auroc"],
            "z_minus_strongest": arms[TRANSCODER]["auroc"] - arms[strongest]["auroc"],
            "z_minus_strongest_ci95": list(paired_interval(y, scores[TRANSCODER], scores[strongest], groups)),
            "z_minus_h_single": arms[TRANSCODER]["auroc"] - arms[HIDDEN]["auroc"],
            "z_minus_h_single_ci95": list(paired_interval(y, scores[TRANSCODER], scores[HIDDEN], groups)),
            "max_logistic_auroc": max(arms[a]["auroc"] for a in TEXT_ARMS),
            "max_gbdt_auroc": max(arms[f"{a}__gbdt"]["auroc"] for a in TEXT_ARMS)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "n_problems": len(pids), "n_cap": int(y.sum()), "n_pit": int((1 - y).sum()),
        "z_dim": int(s_all.shape[1]), "h_dim": int(h_all.shape[1]), "percentile": args.percentile,
        "partition": args.partition, "difficulty_band": [float(lo), float(hi)], "difficulty_imputed": n_imputed,
        "label_set": "defects_removed" if args.remove_problems is not None else "stored", "n_removed": len(removed) if removed else 0,
        "arms": arms, "registered_read": read}, indent=2))
    np.savez_compressed(args.out.with_suffix(".oof.npz"), y=y, groups=groups, pids=np.array(pids),
                        **{f"oof__{name}": oof for name, oof in scores.items()})


if __name__ == "__main__":
    main()

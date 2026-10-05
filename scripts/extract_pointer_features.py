from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from sst.benchmarks.corpus import load_corpus
from sst.probe.bipp_features import BIPP_LAYERS, N_LOGIT_FEATURES, _BLOCK
from sst.shared.extractors import SkyworkExtractor
from sst.student.model import StudentModel

SHARD_KEYS = ("span_feats", "final_feats", "answer_feats", "logit_features")

@dataclass(frozen=True, slots=True)
class Record:
    md5: str
    problem_id: str
    trace_id: str
    prompt: str
    trace: str
    step_spans: list[tuple[int, int]]
    canonical_length: int
    is_correct: bool

def trace_md5(trace: str) -> str:
    return hashlib.md5(trace.encode()).hexdigest()

def records_from_corpus(corpus_dir: Path) -> list[Record]:
    return [
        Record(md5=trace_md5(t.trace), problem_id=str(t.problem_id),
               trace_id=f"{t.problem_id}__{trace_md5(t.trace)[:12]}",
               prompt=t.question, trace=t.trace, step_spans=t.step_spans,
               canonical_length=int(t.canonical_length), is_correct=bool(t.is_correct))
        for t in load_corpus(corpus_dir)
    ]

def records_from_index(index_path: Path, identity_path: Path) -> list[Record]:
    index = json.loads(Path(index_path).read_text())
    identity = torch.load(identity_path, weights_only=False, map_location="cpu")
    order = sorted(index.items(), key=lambda kv: (kv[1][0], kv[1][1]))
    out: list[Record] = []
    cache_path, cache = None, None
    for md5, (path, row, n_steps) in order:
        if md5 not in identity:
            raise ValueError(f"[bipp-feat] md5 {md5} in {index_path} but not {identity_path}")
        if path != cache_path:
            cache = torch.load(path, weights_only=False, map_location="cpu")
            cache_path = path
        trace = cache["traces"][row]
        if trace_md5(trace) != md5:
            raise ValueError(f"[bipp-feat] {path}:{row} hashes to {trace_md5(trace)[:12]}, "
                             f"index says {md5[:12]}")
        L = int(cache["canonical_lengths"][row])
        if L != int(n_steps):
            raise ValueError(f"[bipp-feat] {md5[:12]}: canonical_length {L} != index n_steps "
                             f"{n_steps}; the dense anchors would not align")
        idn = identity[md5]
        out.append(Record(
            md5=md5, problem_id=str(idn["problem_id"]), trace_id=str(idn["trace_id"]),
            prompt=cache["prompts"][row], trace=trace,
            step_spans=cache["step_spans"][row], canonical_length=L,
            is_correct=bool(cache["is_correct"][row]),
        ))
    return out

def shard_path(out_dir: Path, start: int) -> Path:
    return Path(out_dir) / f"shard_{start:08d}.pt"

def shard_is_complete(path: Path, n_expected: int) -> bool:
    if not path.exists():
        return False
    d = torch.load(path, weights_only=False, map_location="cpu")
    return all(k in d for k in SHARD_KEYS) and len(d["span_feats"]) == n_expected

def build_student(student_model: str, target_layers: list[int]) -> StudentModel:
    student = StudentModel(
        student_model, SkyworkExtractor(option="back_line_k"),
        target_layers=list(target_layers), vllm_client=None,
    )
    student.model = student.model.to("cuda").eval()
    return student

def extract_block(
    student: StudentModel,
    block: list[Record],
    *,
    target_layers: list[int],
    top_k: int = 5,
) -> dict:

    prompts = [t.prompt for t in block]
    traces = [t.trace for t in block]
    spans = [t.step_spans for t in block]
    for t in block:
        if int(t.canonical_length) > len(t.step_spans):
            raise ValueError(
                f"[bipp-feat] trace {t.trace_id}: canonical_length="
                f"{t.canonical_length} > len(step_spans)={len(t.step_spans)}"
            )

    pooled = student.extract_span_pooled_features(
        prompts=prompts, traces=traces, step_spans=spans, layers=list(target_layers),
    )
    lf = student.extract_decision_logit_features(
        prompts=prompts, traces=traces, step_spans=spans, top_k=top_k,
    )

    span_feats, final_feats, answer_feats, logit_features = [], [], [], []
    for i, t in enumerate(block):
        L = int(t.canonical_length)
        span_feats.append(pooled["span_mean"][i][:L].contiguous())
        final_feats.append(pooled["step_final"][i][:L].contiguous())
        answer_feats.append(pooled["answer"][i].contiguous())
        logit_features.append(lf[i, :L].float().contiguous())
    return {
        "span_feats": span_feats,
        "final_feats": final_feats,
        "answer_feats": answer_feats,
        "logit_features": logit_features,
        "md5": [t.md5 for t in block],
        "problem_ids": [t.problem_id for t in block],
        "trace_ids": [t.trace_id for t in block],
        "canonical_lengths": [int(t.canonical_length) for t in block],
        "is_correct": [bool(t.is_correct) for t in block],
        "step_spans": [t.step_spans for t in block],
        "layers": list(target_layers),
    }

def extract_records(
    records: list[Record],
    out_dir: Path,
    student_model: str,
    *,
    target_layers: list[int],
    shard_start: int = 0,
    shard_end: int = -1,
    shard_size: int = 200,
    top_k: int = 5,
) -> Path:

    corpus = records
    end = len(corpus) if shard_end < 0 else min(shard_end, len(corpus))
    if not 0 <= shard_start < end:
        raise ValueError(
            f"[bipp-feat] empty slice [{shard_start}, {end}) over {len(corpus)} traces."
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    student = None
    for bs in range(shard_start, end, shard_size):
        be = min(bs + shard_size, end)
        sp = shard_path(out_dir, bs)
        if shard_is_complete(sp, be - bs):
            print(f"[bipp-feat] {sp.name} already complete ({be - bs} traces); skipping", flush=True)
            continue
        if student is None:
            student = build_student(student_model, target_layers)
            width = len(target_layers) * student.hidden_size
            if width != _BLOCK:
                raise ValueError(
                    f"[bipp-feat] block width {len(target_layers)}*{student.hidden_size}"
                    f"={width} != bipp_features._BLOCK={_BLOCK}; the shard would not "
                    f"assemble under the shared contract."
                )
        shard = extract_block(
            student, corpus[bs:be], target_layers=target_layers, top_k=top_k,
        )
        torch.save(shard, sp)
        torch.cuda.empty_cache()
        print(f"[bipp-feat] wrote {sp.name}: {be - bs} traces "
              f"(span={tuple(shard['span_feats'][0].shape)}, "
              f"logit={tuple(shard['logit_features'][0].shape)}, F={N_LOGIT_FEATURES})",
              flush=True)

    print(f"[bipp-feat] done - records[{shard_start}:{end}] -> {out_dir}", flush=True)
    return out_dir

def resolve_records(corpus: Path | None, corpus_index: Path | None,
                    identity_file: Path | None) -> list[Record]:

    if (corpus is None) == (corpus_index is None):
        raise ValueError("[bipp-feat] pass exactly one of --corpus / --corpus-index")
    if corpus is not None:
        return records_from_corpus(corpus)
    if identity_file is None:
        raise ValueError("[bipp-feat] --corpus-index requires --identity-file "
                         "(gold_pits_by_md5.pt: prefixed problem_id + trace_id)")
    return records_from_index(corpus_index, identity_file)

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=None,
                    help="BenchmarkTrace corpus dir (shard_*.pt).")
    ap.add_argument("--corpus-index", type=Path, default=None,
                    help="corpus_index.json {md5: [all_traces_shard, row, n_steps]} - "
                         "extracts exactly the labelled traces.")
    ap.add_argument("--identity-file", type=Path, default=None,
                    help="gold_pits_by_md5.pt; required with --corpus-index.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--student-model", required=True)
    ap.add_argument("--target-layers", type=int, nargs="+", default=list(BIPP_LAYERS))
    ap.add_argument("--shard-start", type=int, default=0,
                    help="First record index of this slice (inclusive).")
    ap.add_argument("--shard-end", type=int, default=-1,
                    help="Last record index of this slice (exclusive); -1 = end.")
    ap.add_argument("--shard-size", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args(argv)

    records = resolve_records(args.corpus, args.corpus_index, args.identity_file)
    print(f"[bipp-feat] {len(records)} source records", flush=True)
    extract_records(
        records, args.out_dir, args.student_model,
        target_layers=args.target_layers, shard_start=args.shard_start,
        shard_end=args.shard_end, shard_size=args.shard_size, top_k=args.top_k,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

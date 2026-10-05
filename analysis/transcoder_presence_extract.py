"""Per-trace transcoder and hidden-state activation sums for one failed trace per problem (GPU)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

from sst.shared.extractors import SkyworkExtractor
from sst.student.model import StudentModel

MAX_TRACE_TOKENS = 16384
CHUNK = 1024


def load_failed_traces(corpus_dir: Path, trace_ids: set[str], regimes: dict[str, str]) -> list[tuple[str, str, str, str]]:
    """(problem_id, trace_id, question, trace) of the requested failed traces, sorted by problem_id then trace_id."""
    shard_paths = sorted(corpus_dir.glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"no shard_*.pt in {corpus_dir}")
    rows = []
    for path in shard_paths:
        for row in torch.load(path, map_location="cpu", weights_only=False):
            if row["trace_id"] in trace_ids and row["problem_id"] in regimes and not row["is_correct"]:
                rows.append((row["problem_id"], row["trace_id"], row["question"], row["trace"]))
    if len(rows) != len(trace_ids):
        raise KeyError(f"found {len(rows)} of {len(trace_ids)} requested failed traces in {corpus_dir}")
    return sorted(rows, key=lambda r: (r[0], r[1]))


def encode(transcoder, x: torch.Tensor) -> torch.Tensor:
    """Transcoder latent activations, the encoder half of the transcoder's forward pass."""
    z = transcoder.enc(x)
    if transcoder._threshold is not None:
        return (z * (z > transcoder._threshold)).to(z.dtype)
    return torch.relu(z)


@torch.no_grad()
def trace_sums(student: StudentModel, prompt: str, trace: str, layers: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]:
    """Sums over every generated token of ReLU(transcoder latents) and ReLU(hidden state) at the given layers."""
    p_ids = student.tokenizer(prompt, add_special_tokens=False).input_ids
    t_ids = student.tokenizer(trace, add_special_tokens=False).input_ids
    truncated = len(t_ids) > MAX_TRACE_TOKENS
    t_ids = t_ids[:MAX_TRACE_TOKENS]
    n_tok = len(t_ids)
    if n_tok == 0:
        raise ValueError("trace has no tokens")
    device = student.model.device
    input_ids = torch.tensor([p_ids + t_ids], dtype=torch.long, device=device)
    positions = torch.arange(len(p_ids), len(p_ids) + n_tok, device=device, dtype=torch.long).unsqueeze(0)
    hidden, transcoder_input = student._collect_selected_hidden_states(
        {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)},
        hidden_state_indices=list(layers), step_end_indices=positions)
    h_sum = torch.cat([torch.relu(hidden[layer].float()).sum(dim=1).squeeze(0).cpu() for layer in layers]).numpy()
    width = student.transcoders[str(layers[0])].d_transcoder
    cols, vals = [], []
    for i, layer in enumerate(layers):
        transcoder = student.transcoders[str(layer)]
        acts = transcoder_input.get(layer, hidden[layer])
        total = torch.zeros(width, dtype=torch.float32, device=device)
        for start in range(0, n_tok, CHUNK):
            total += encode(transcoder, acts[:, start:start + CHUNK, :]).float().sum(dim=1).squeeze(0)
        nonzero = torch.nonzero(total, as_tuple=False).squeeze(-1)
        if nonzero.numel():
            cols.append((nonzero + i * width).to(torch.int32).cpu().numpy())
            vals.append(total[nonzero].cpu().numpy().astype(np.float32))
    torch.cuda.empty_cache()
    return (np.concatenate(cols) if cols else np.empty(0, np.int32),
            np.concatenate(vals) if vals else np.empty(0, np.float32), h_sum.astype(np.float32), n_tok, truncated)


def write_shard(path: Path, indptr: list[int], cols: list[np.ndarray], vals: list[np.ndarray], h: list[np.ndarray]) -> None:
    """One store shard: CSR transcoder sums over its traces and the dense hidden-state sums."""
    save_file({"z_indptr": np.asarray(indptr, dtype=np.int64),
               "z_cols": np.concatenate(cols) if cols else np.empty(0, np.int32),
               "z_vals": np.concatenate(vals) if vals else np.empty(0, np.float32),
               "h_sum": np.stack(h).astype(np.float32)}, str(path))


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--problem-regimes", type=Path, required=True)
    parser.add_argument("--trace-ids", type=Path, required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--transcoder-dir", required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--shard-traces", type=int, default=200)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.out / "meta.json").exists():
        raise FileExistsError(f"{args.out} already holds a store")
    if not torch.cuda.is_available():
        raise RuntimeError("extraction requires a CUDA device")
    shard_dir = args.out / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    meta_in = json.loads(args.problem_regimes.read_text())
    regimes = {p: r for p, r in zip(meta_in["problem_ids"], meta_in["problem_regimes"]) if r in ("capability", "pit")}
    items = load_failed_traces(args.corpus_dir, set(json.loads(args.trace_ids.read_text())), regimes)
    student = StudentModel(args.student_model, SkyworkExtractor(option="original"),
                           pretrained_transcoder_repo=args.transcoder_dir, target_layers=list(args.layers),
                           vllm_client=None)
    student.model = student.model.to("cuda").eval()
    student.transcoders = student.transcoders.to(device="cuda", dtype=torch.bfloat16)
    assert next(student.model.parameters()).device.type == "cuda"
    assert next(student.transcoders.parameters()).device.type == "cuda"
    layers = sorted(args.layers)
    width = student.transcoders[str(layers[0])].d_transcoder
    z_dim, h_dim = len(layers) * width, len(layers) * student.hidden_size

    traces, shard_files = [], []
    indptr, cols, vals, h = [0], [], [], []
    density, n_truncated, n_empty = 0.0, 0, 0
    for problem_id, trace_id, question, trace in items:
        z_cols, z_vals, h_sum, n_tok, truncated = trace_sums(student, question, trace, layers)
        n_empty += n_tok == 0
        n_truncated += truncated
        cols.append(z_cols)
        vals.append(z_vals)
        h.append(h_sum)
        indptr.append(indptr[-1] + len(z_cols))
        traces.append({"trace_id": trace_id, "problem_id": problem_id, "regime": regimes[problem_id],
                       "n_tok": int(n_tok), "truncated": bool(truncated)})
        density += len(z_cols) / z_dim
        if len(indptr) - 1 >= args.shard_traces:
            shard_files.append(f"presence_shard_{len(shard_files):04d}.safetensors")
            write_shard(shard_dir / shard_files[-1], indptr, cols, vals, h)
            indptr, cols, vals, h = [0], [], [], []
    if len(indptr) > 1:
        shard_files.append(f"presence_shard_{len(shard_files):04d}.safetensors")
        write_shard(shard_dir / shard_files[-1], indptr, cols, vals, h)
    (args.out / "meta.json").write_text(json.dumps({
        "z_dim": z_dim, "h_dim": h_dim, "z_per_layer": width, "h_per_layer": student.hidden_size, "layers": layers,
        "maxtok": MAX_TRACE_TOKENS, "chunk": CHUNK, "n_problems": len({t["problem_id"] for t in traces}),
        "n_traces": len(traces), "n_truncated": n_truncated, "n_empty": n_empty,
        "mean_z_density": density / len(traces), "read_site": "post_attention_layernorm output",
        "shard_dir": "shards", "shard_files": shard_files, "traces": traces}, indent=2))


if __name__ == "__main__":
    main()

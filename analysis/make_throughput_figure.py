from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .fig_style import apply, mark_kw, NEUTRAL, FIG_W

apply()

SEQ_COLOR = {64: "#7A5195", 128: "#B8860B", 256: "#005AAC", 512: "#C34E02", 1024: "#008A59"}
MODEL_LABEL = {"4b": "4B", "8b": "8B"}

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("SST_FIG_THROUGHPUT", ROOT / "docs/throughput_l40s.json"))
OUT_DIR = Path(os.environ.get("SST_FIG_OUT_DIR", ROOT / "figures"))


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text())["cells"]


def ok(cells: list[dict]) -> list[dict]:
    return [c for c in cells if c["status"] == "ok"]


def rollout_figure(cells: list[dict], size: str, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, 3.6))
    axes = list(axes.flat)
    panels = [("gen_tokens_per_s", "generated tokens/s\n(end to end)"), ("seqs_per_s", "completed sequences/s\n(end to end)"),
              ("util_span", "utilization >= 90 %\n(fraction of span)"), ("kv_pct_mean", "mean KV-cache usage (%)")]
    for mns in sorted({c["max_num_seqs"] for c in cells}):
        row = sorted((c for c in ok(cells) if c["max_num_seqs"] == mns and not c["replicate"]),
                     key=lambda c: c["prefix_batch"])
        x = [c["prefix_batch"] for c in row]
        for ax, (key, _) in zip(axes, panels):
            y = [c[key]["frac_ge90"] if key == "util_span" else c[key] for c in row]
            post = all(c["post_hoc"] for c in row)
            ax.plot(x, y, "--o" if post else "-o", ms=3.4, lw=1.4, label=f"max-num-seqs {mns}" + (" (post hoc)" if post else ""),
                    **mark_kw(SEQ_COLOR[mns], size))
        axes[0].plot(x, [c["incall_gen_tokens_per_s"] for c in row], ":", lw=1.0, color=SEQ_COLOR[mns],
                     label=f"in-call, {mns}")
        for c in cells:
            if c["max_num_seqs"] == mns and c["status"] != "ok":
                axes[0].annotate(c["status"], (c["prefix_batch"], 0), fontsize=6.5, color=NEUTRAL,
                                 ha="center", va="bottom")
    for c in ok(cells):
        if c["replicate"]:
            for ax, (key, _) in zip(axes, panels):
                y = c[key]["frac_ge90"] if key == "util_span" else c[key]
                ax.plot([c["prefix_batch"]], [y], "x", ms=5, color=NEUTRAL,
                        label="replicate" if ax is axes[0] else None)
    for ax, (_, label) in zip(axes, panels):
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 16, 64])
        ax.set_xticklabels(["8", "16", "64"])
        ax.set_xlabel("--prefix-batch")
        ax.set_ylabel(label)
    axes[2].set_ylim(0, 1.05)
    axes[3].set_ylim(0, 105)
    for ax in axes[:2]:
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    axes[0].legend(loc="lower right", fontsize=6.0)
    fig.suptitle(f"Rollouts, Qwen3-{MODEL_LABEL[size]}, one L40S", fontsize=8.5)
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.95))
    save(fig, out)
    plt.close(fig)


def save(fig, out: Path) -> None:
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = load(DATA)
    for size in ("4b", "8b"):
        sel = [c for c in cells if c["size"] == size and c["kind"] == "roll"]
        rollout_figure(sel, size, OUT_DIR / f"fig_throughput_roll_{size}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

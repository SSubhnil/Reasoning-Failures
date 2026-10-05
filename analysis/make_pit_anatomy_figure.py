from __future__ import annotations

import csv as _csv
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .fig_style import (apply, type_color, NEUTRAL, REF, FIG_W, MIN_FONT)

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/paper_tables"
OUT = ROOT / "figures/fig_pit_anatomy.pdf"

apply()

ORDER = ["capability", "pit", "reaches"]
DISPLAY = ["capability\ngap", "pit", "near miss"]

def load_pits(m: str) -> tuple[np.ndarray, np.ndarray]:
    rows = list(_csv.DictReader(open(D / f"mc_pit_full_{m}.csv")))
    return (np.array([float(r["rel_depth"]) for r in rows]),
            np.array([float(r["pre_pit_value"]) for r in rows]))

def main() -> None:
    regf = pd.read_csv(D / "mc_regimes_full.csv")
    rd4, pv4 = load_pits("4b")
    rd8, pv8 = load_pits("8b")

    fig, axs = plt.subplots(1, 3, figsize=(FIG_W - 0.15, 2.00),
                            gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})

    ax = axs[0]
    x = np.arange(3)
    w = 0.36

    for scale, model, off, lab_off in (("4b", "qwen3-4b", -0.20, -0.09),
                                       ("8b", "qwen3-8b", 0.20, 0.09)):
        sub = regf[regf.model == model].set_index("regime")
        vals = [float(sub.loc[r, "fraction"]) for r in ORDER]
        colors = [type_color(r, scale) for r in ORDER]
        bars = ax.bar(x + off, vals, width=w, color=colors, linewidth=0)

        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2 + lab_off, v + 0.012, f"{v:.2f}",
                    ha="center", fontsize=MIN_FONT)
    ax.set_xticks(x)
    ax.set_xticklabels(DISPLAY)

    ax.set_xlim(-0.58, 2.58)

    ax.set_ylim(0, 0.78)
    ax.set_yticks(np.arange(0, 0.71, 0.2))
    ax.set_ylabel("fraction of incorrect traces")

    scale_handles = [tuple(Patch(facecolor=type_color(r, s), edgecolor="none") for r in ORDER)
                     for s in ("4b", "8b")]
    ax.legend(scale_handles, ["4B", "8B"],
              handler_map={tuple: HandlerTuple(ndivide=len(ORDER), pad=0.0)},
              frameon=False, loc="upper center",
              ncol=2, bbox_to_anchor=(0.5, 1.06), columnspacing=1.2, handlelength=2.4,
              handletextpad=0.4)

    ax.set_title("Failure types", linespacing=1.15)

    ax = axs[1]
    c4, c8 = type_color("pit", "4b"), type_color("pit", "8b")
    b2 = np.linspace(0, 1, 21)
    ax.hist(rd4, bins=b2, density=True, histtype="step", lw=1.6, color=c4,
            label=f"4B, n={len(rd4):,}")
    ax.hist(rd8, bins=b2, density=True, histtype="step", lw=1.6, color=c8,
            label=f"8B, n={len(rd8):,}")
    m4, m8 = float(np.median(rd4)), float(np.median(rd8))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3.40)
    ax.set_yticks([0, 1, 2, 3])

    ax.axvline(m4, ymax=0.70, **{**REF["operating"], "color": c4})
    ax.axvline(m8, ymax=0.70, **{**REF["operating"], "color": c8})
    ax.annotate("medians", xy=(m4, 2.10), xytext=(0.46, 2.10), fontsize=MIN_FONT,
                color=NEUTRAL, ha="right", va="center",
                arrowprops=dict(arrowstyle="->", color=NEUTRAL, lw=0.8))
    ax.set_xlabel("relative depth of the pit step")
    ax.set_ylabel("density")
    ax.legend(frameon=False, loc="upper left", handlelength=1.2,
              handletextpad=0.5, borderpad=0.1, labelspacing=0.3)
    ax.set_title("Depth of the\npit step", linespacing=1.15)

    ax = axs[2]
    bins = np.arange(-0.5, 17) / 16
    ax.hist(pv4, bins=bins, density=True, histtype="step", lw=1.6, color=c4, label="4B")
    ax.hist(pv8, bins=bins, density=True, histtype="step", lw=1.6, color=c8, label="8B")
    thr = 0.25
    f4 = float((pv4 > thr).mean())
    f8 = float((pv8 > thr).mean())
    ax.axvline(thr, **REF["operating"])

    ax.text(thr + 0.02, ax.get_ylim()[1] * 0.90, f"{thr}", fontsize=MIN_FONT,
            color=NEUTRAL, ha="left", va="top")

    ax.set_xlabel("best reachable value\n" r"$V^*_{\mathrm{pit}-1}$", linespacing=1.15)
    ax.set_ylabel("density")
    ax.set_xlim(0, 1)

    ax.text(0.97, ax.get_ylim()[1] * 0.58,
            f"above {thr}:\n4B {f4 * 100:.0f}%\n8B {f8 * 100:.0f}%",
            fontsize=MIN_FONT, color=NEUTRAL, ha="right", va="top")
    ax.legend(frameon=False, loc="upper right", handlelength=1.2,
              handletextpad=0.5, borderpad=0.1, labelspacing=0.3)
    ax.set_title("Best reachable value\nbefore the pit step", linespacing=1.15)

    fig.tight_layout(pad=0.4, w_pad=1.2)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

from .fig_style import (BAND, FIG_W, MIN_FONT, NEUTRAL, REF, TICK_FONT,
                       TITLE_FONT, apply, mark_kw, readout_color)

ROOT = Path(__file__).resolve().parents[1]
LAYER_CSV = ROOT / "data/paper_tables/probe_layer_localization.csv"
POINTER_JSON = ROOT / "runs/resp/bipp_locus_4b/score_excl818.json"
DETECTION_JSON = ROOT / "outputs/detection_controls.json"
OUT = ROOT / "figures/fig_wall.pdf"

apply()
H = readout_color("h", "4b")

UNIFORM_DENSE = (0.2826, 0.2729, 0.2933)

LENGTH_DENSE = (0.4413, 0.4076, 0.4745)

LENGTH_BROAD = (0.4462, 0.4122, 0.4796)

AXES_POSITIONS = ([0.09861111111111111, 0.15534449891067537, 0.4502789483749283, 0.7603894335511981],
                  [0.6570047714201956, 0.15534449891067537, 0.29050254733866343, 0.7603894335511981])

def _xerr(value: float, lo: float, hi: float) -> list[list[float]]:
    return [[value - lo], [hi - value]]

def main() -> None:
    layers = pd.read_csv(LAYER_CSV).sort_values("layer")
    pointer = json.loads(POINTER_JSON.read_text())
    pointer_rows = [row for row in pointer["rows"] if row["seed"] != "pooled"]
    detector_auroc = json.loads(DETECTION_JSON.read_text())["per_cell"]["4b_bce"]["raw_auroc"]

    fig, axs = plt.subplots(1, 2, figsize=(FIG_W - 0.10, 2.55),
                            gridspec_kw={"width_ratios": [1.55, 1.0]})

    ax = axs[0]
    retrain_y = list(range(len(pointer_rows)))

    ref_y_lo, ref_y_hi = retrain_y[0] - 0.65, retrain_y[-1] + 0.65

    ax.fill_betweenx([ref_y_lo, ref_y_hi], LENGTH_BROAD[1], LENGTH_BROAD[2], color=BAND, zorder=0)
    ax.plot([UNIFORM_DENSE[0], UNIFORM_DENSE[0]], [ref_y_lo, ref_y_hi], zorder=1, **REF["floor"])
    ax.plot([LENGTH_BROAD[0], LENGTH_BROAD[0]], [ref_y_lo, ref_y_hi], zorder=1, **{**REF["operating"], "ls": "-"})
    ax.plot([LENGTH_DENSE[0], LENGTH_DENSE[0]], [ref_y_lo, ref_y_hi], zorder=1, **REF["operating"])

    for y, row in zip(retrain_y, pointer_rows):
        value = row["pit_hit_pm2"]
        ax.plot([value, LENGTH_BROAD[0]], [y, y], color=NEUTRAL, lw=0.7, zorder=1)
        ax.plot([LENGTH_BROAD[0]], [y], marker="s", ls="none", ms=3.5,
                markerfacecolor="white", markeredgecolor=NEUTRAL, markeredgewidth=0.8)
        ax.errorbar([value], [y],
                    xerr=_xerr(value, row["pit_hit_pm2_lo"], row["pit_hit_pm2_hi"]),
                    fmt="o", ms=4.8, capsize=2, elinewidth=0.9, ecolor=H,
                    **mark_kw(H, "4b"))

    ax.set_yticks(retrain_y)
    ax.set_yticklabels([f"seed {row['seed']}" for row in pointer_rows], fontsize=TICK_FONT)
    ax.set_ylim(retrain_y[-1] + 1.50, -1.35)
    ax.set_xlim(0.19, 0.58)
    ax.text(0.195, -1.00, "Qwen3-4B, pointer probe,\n818 pits of the gold collection", fontsize=MIN_FONT,
            color=NEUTRAL, style="italic", va="center")
    halo = dict(facecolor="white", edgecolor="none", pad=0.5)
    ax.text(UNIFORM_DENSE[0], ref_y_hi + 0.08, "uniform\n0.283", fontsize=MIN_FONT,
            color=NEUTRAL, ha="center", va="top", bbox=halo)
    ax.text(LENGTH_DENSE[0], ref_y_hi + 0.08, "length rule\n0.441 / 0.446", fontsize=MIN_FONT,
            color=NEUTRAL, ha="center", va="top", bbox=halo)
    ax.set_xlabel(r"localization Hit@$\pm2$")
    ax.set_title("(A) The pointer probe against the length rule", fontsize=TITLE_FONT)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax = axs[1]

    yerr = [layers.auroc - layers.auroc_lo, layers.auroc_hi - layers.auroc]
    ax.errorbar(layers.layer, layers.auroc, yerr=yerr, fmt="o-", color=H, lw=1.4, ms=3.5,
                capsize=2, elinewidth=0.9)

    ax.axhline(0.5, zorder=1, **REF["chance"])
    x_lo, x_hi = 7.0, 37.5
    label_x = x_hi - 0.5
    halo = dict(facecolor="white", edgecolor="none", pad=0.8)
    ax.text(label_x, 0.5 - 0.021, "chance", fontsize=MIN_FONT, color=NEUTRAL,
            ha="right", va="top", zorder=5, bbox=halo)
    ax.axhline(detector_auroc, zorder=1, **REF["ceiling"])
    ax.text(x_lo + 0.3, detector_auroc - 0.020,
            f"confidence probe {detector_auroc:.3f}", fontsize=MIN_FONT,
            color=NEUTRAL, va="top", zorder=5, bbox=halo)
    ax.text(label_x, 0.545, "distance score", fontsize=MIN_FONT, color=H,
            ha="right", va="bottom")
    ax.set_xlabel("layer")
    ax.set_ylabel("pit-step AUROC")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0.45, 0.86)

    ax.text(0.02, 0.03, "Qwen3-4B", transform=ax.transAxes, fontsize=MIN_FONT,
            color=NEUTRAL, style="italic", ha="left", va="bottom")
    ax.set_title("(B) Distance score by layer", fontsize=TITLE_FONT)

    for axis, position in zip(axs, AXES_POSITIONS):
        axis.set_position(position)
    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    dx = (381.21 / 72 - tight.width) / 2
    dy = (181.6 / 72 - tight.height) / 2
    fig.savefig(OUT, bbox_inches=Bbox.from_extents(tight.x0 - dx, tight.y0 - dy, tight.x1 + dx, tight.y1 + dy))
    plt.close(fig)

if __name__ == "__main__":
    main()

"""Read-site selection figure: correctness AUROC by layer and by number of layers, Qwen3-4B and Qwen3-8B."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

from .fig_style import apply, readout_color, mark_kw, NEUTRAL, REF, FIG_W, MIN_FONT

apply()

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs/layer_selection"
OUT = ROOT / "figures/fig_layers.pdf"

SCALES = ["4b", "8b"]
SCALE_LABEL = {"4b": "4B", "8b": "8B"}
COUNT_KS = [2, 3, 5, 8]


def _load_depth(scale: str) -> dict[str, list[dict]]:
    """Rows of depth_curve.csv for one scale, split by representation and sorted by layer."""
    path = RUN_DIR / f"qwen3-{scale}" / "depth_curve.csv"
    with path.open() as f:
        rows = list(csv.DictReader(f))
    out: dict[str, list[dict]] = {"hidden": [], "transcoder": []}
    for r in rows:
        r["layer"] = int(r["layer"])
        r["auroc_sel"] = float(r["auroc_sel"])
        r["sel_ci_lo"] = float(r["sel_ci_lo"])
        r["sel_ci_hi"] = float(r["sel_ci_hi"])
        out[r["representation"]].append(r)
    for rep in out:
        out[rep].sort(key=lambda r: r["layer"])
    return out


def _load_count(scale: str) -> dict[str, list[dict]]:
    """Rows of count_curve.csv for one scale, split by representation and sorted by k."""
    path = RUN_DIR / f"qwen3-{scale}" / "count_curve.csv"
    with path.open() as f:
        rows = list(csv.DictReader(f))
    out: dict[str, list[dict]] = {"hidden": [], "transcoder": []}
    for r in rows:
        r["k"] = int(r["k"])
        r["auroc_sel"] = float(r["auroc_sel"])
        r["sel_ci_lo"] = float(r["sel_ci_lo"])
        r["sel_ci_hi"] = float(r["sel_ci_hi"])
        out[r["representation"]].append(r)
    for rep in out:
        out[rep].sort(key=lambda r: r["k"])
    return out


def _decision(scale: str) -> dict:
    return json.loads((RUN_DIR / f"qwen3-{scale}" / "decision.json").read_text())


def main() -> int:
    depth = {s: _load_depth(s) for s in SCALES}
    count = {s: _load_count(s) for s in SCALES}
    decision = {s: _decision(s) for s in SCALES}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(FIG_W - 0.26, 2.5),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.17, wspace=0.92)

    ends: list[tuple[float, str, str, str]] = []
    for scale in SCALES:
        ls = "-" if scale == "8b" else (0, (5, 2))
        c_h = readout_color("h", scale)
        hid = depth[scale]["hidden"]
        tc = depth[scale]["transcoder"]
        xs_h = [r["layer"] for r in hid]
        ys_h = [r["auroc_sel"] for r in hid]
        lo_h = [r["sel_ci_lo"] for r in hid]
        hi_h = [r["sel_ci_hi"] for r in hid]
        axA.plot(xs_h, ys_h, ls=ls, color=c_h, lw=1.6, zorder=4)
        axA.fill_between(xs_h, lo_h, hi_h, color=c_h, alpha=0.13, zorder=2)
        ends.append((ys_h[-1], f"hidden {SCALE_LABEL[scale]}", c_h, ls))

        c_z = readout_color("z", scale)
        xs_t = [r["layer"] for r in tc]
        ys_t = [r["auroc_sel"] for r in tc]
        axA.plot(xs_t, ys_t, ls=ls, color=c_z, lw=1.3, zorder=3)
        ends.append((ys_t[-1], f"transcoder {SCALE_LABEL[scale]}", c_z, ls))

    x_end = max(r["layer"] for r in depth["8b"]["hidden"])
    ends.sort(key=lambda e: e[0])
    fan_center = sum(e[0] for e in ends) / len(ends)
    fan_pitch = 0.042
    for i, (y_end, text, color, ls) in enumerate(ends):
        y_lab = fan_center + fan_pitch * (i - (len(ends) - 1) / 2)
        axA.plot([x_end, x_end + 2.6], [y_end, y_lab], ls="-", lw=0.6, color=color,
                 zorder=3, clip_on=False)
        axA.text(x_end + 3.2, y_lab, text, fontsize=MIN_FONT, color=color,
                 ha="left", va="center", clip_on=False)

    peak_layer = decision["4b"]["per_rep_peaks"]["hidden"]["peak_layer"]
    peak_auroc = decision["4b"]["per_rep_peaks"]["hidden"]["peak_auroc"]
    axA.axvline(peak_layer, **REF["operating"], zorder=1)
    axA.plot(peak_layer, peak_auroc, marker="*", ms=11, color=NEUTRAL,
             markeredgecolor="white", markeredgewidth=0.6, zorder=6)
    axA.annotate("peak, layer 23", xy=(peak_layer, peak_auroc + 0.012), xytext=(29.0, 0.828),
                 fontsize=MIN_FONT, color=NEUTRAL, ha="center", va="bottom",
                 arrowprops={"arrowstyle": "-", "lw": 0.6, "color": NEUTRAL,
                             "shrinkA": 2, "shrinkB": 2})
    axA.axhline(0.5, **REF["chance"], zorder=1)
    axA.text(29.0, 0.512, "chance", fontsize=MIN_FONT, color=NEUTRAL, va="bottom",
             bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8}, zorder=2)
    axA.set_xlabel("probe layer")
    axA.set_ylabel("detection AUROC")
    axA.set_title("(A) AUROC by layer")
    axA.set_xlim(0, 37)
    axA.set_ylim(0.44, 0.90)

    for scale in SCALES:
        color = readout_color("h", scale)
        ls = "-" if scale == "8b" else (0, (5, 2))
        hid = count[scale]["hidden"]
        xs = [r["k"] for r in hid]
        ys = [r["auroc_sel"] for r in hid]
        lo = [r["auroc_sel"] - r["sel_ci_lo"] for r in hid]
        hi = [r["sel_ci_hi"] - r["auroc_sel"] for r in hid]
        axB.errorbar(xs, ys, yerr=[lo, hi], marker="s", ms=4.5, lw=1.5, ls=ls,
                     capsize=3, zorder=4, ecolor=color, **mark_kw(color, scale))
        axB.text(xs[-1] + 0.55, ys[-1], SCALE_LABEL[scale], fontsize=MIN_FONT,
                 color=color, ha="left", va="center")

    axB.axvline(2, **REF["operating"], zorder=1)
    axB.text(2.3, 0.754, r"selected $\mathcal{S}^\star$={23,24}", fontsize=MIN_FONT,
              color=NEUTRAL, va="bottom",
              bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8}, zorder=5)
    axB.set_xlabel("number of adjacent layers")
    axB.set_ylabel("detection AUROC")
    axB.set_title("(B) AUROC by number of layers")
    axB.set_xticks(COUNT_KS)
    axB.set_xlim(1.4, 9.4)
    axB.set_ylim(0.668, 0.765)

    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    dx = (394.89 / 72 - tight.width) / 2
    dy = (181.31 / 72 - tight.height) / 2
    fig.savefig(OUT, bbox_inches=Bbox.from_extents(tight.x0 - dx, tight.y0 - dy, tight.x1 + dx, tight.y1 + dy))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

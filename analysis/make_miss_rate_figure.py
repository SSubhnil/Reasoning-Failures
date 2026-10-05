from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch

from .fig_style import (apply, type_color, readout_color, NEUTRAL, REF,
                       FIG_W, MIN_FONT)

apply()

ROOT = Path(__file__).resolve().parents[1]

T3_JSON = Path(os.environ.get("SST_FIG_T3", ROOT / "outputs/conceal/t3_per_class_cer.json"))
T1_JSON = Path(os.environ.get("SST_FIG_T1", ROOT / "outputs/conceal/t1_value_control.json"))
CALIB_JSON = Path(os.environ.get("SST_FIG_CALIB",
                                 ROOT / "outputs/calibration_matched.json"))
OUT = Path(os.environ.get("SST_FIG_OUT", ROOT / "figures/fig_concealment.pdf"))

AXES_POSITIONS = ([0.08277777777777778, 0.18028007346189168, 0.2189446066166249, 0.673672807621671],
                 [0.4435150820127884, 0.18028007346189168, 0.1980927393198036, 0.673672807621671],
                 [0.7834005189509776, 0.18028007346189168, 0.20851867296821425, 0.673672807621671])

TYPES_A = ["capability", "pit", "reaches"]

DISPLAY = {"capability": "capability\ngap", "pit": "pit", "reaches": "near miss"}
MODEL_LABEL = {"4b": "4B", "8b": "8B"}

def asymmetric_yerr(vals: list[float], los: list[float], his: list[float]) -> np.ndarray:
    return np.array([[v - l for v, l in zip(vals, los)], [h - v for v, h in zip(vals, his)]])

def _neutralize_caps(container) -> None:
    if container.errorbar is None:
        return
    for cap in container.errorbar[1]:
        cap.set_color(NEUTRAL)
        cap.set_markeredgecolor(NEUTRAL)

LABEL_PAD_PT = 2.0
LABEL_STEP_PT = 9.0

def stacked_value_labels(ax, x_centers, extreme, values, above: bool) -> None:
    sign = 1.0 if above else -1.0
    for row, (xc, v) in enumerate(zip(x_centers, values)):
        ax.annotate(f"{v:.2f}", xy=(xc, extreme),
                    xytext=(0, sign * (LABEL_PAD_PT + LABEL_STEP_PT * row)),
                    textcoords="offset points", ha="center",
                    va="bottom" if above else "top", fontsize=MIN_FONT)

def main() -> int:
    t3 = json.loads(T3_JSON.read_text())["per_model"]
    t1all = json.loads(T1_JSON.read_text())["figure_inputs"]
    t1 = {sc: t1all[sc] for sc in ("4b", "8b")}
    calib = json.loads(CALIB_JSON.read_text())

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(FIG_W, 2.42),
                                        gridspec_kw={"width_ratios": [1.05, 0.95, 1.0]})

    x = np.arange(len(TYPES_A))
    width = 0.35
    a_vals: dict[str, list[float]] = {}
    a_his: dict[str, list[float]] = {}
    for model, off in (("4b", -width / 2), ("8b", width / 2)):
        cer = t3[model]["cer"]
        vals = [cer[t]["point"] for t in TYPES_A]
        los = [cer[t]["ci_lo"] for t in TYPES_A]
        his = [cer[t]["ci_hi"] for t in TYPES_A]
        yerr = asymmetric_yerr(vals, los, his)
        colors = [type_color(t, model) for t in TYPES_A]
        bars = axA.bar(x + off, vals, width, yerr=yerr, capsize=2.5, color=colors,
                        linewidth=0, zorder=3, error_kw={"lw": 0.8, "ecolor": NEUTRAL})
        _neutralize_caps(bars)
        a_vals[model], a_his[model] = vals, his

    for i in range(len(TYPES_A)):
        stacked_value_labels(axA, (x[i] - width / 2, x[i] + width / 2),
                             max(a_his["4b"][i], a_his["8b"][i]),
                             (a_vals["4b"][i], a_vals["8b"][i]), above=True)

    axA.set_xticks(x)
    axA.set_xticklabels([DISPLAY[t] for t in TYPES_A])

    axA.set_ylim(0, 1.25)
    axA.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axA.set_ylabel("miss rate")

    axA.set_title("(A) miss rate by\nfailure type")

    scale_handles = [
        tuple(Patch(facecolor=type_color(t, model), edgecolor="none") for t in TYPES_A)
        for model in ("4b", "8b")
    ]

    axA.legend(scale_handles, ["4B", "8B"],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3,
               frameon=False, columnspacing=0.4, handletextpad=0.25,
               handlelength=1.2, borderpad=0.0)

    xs = t1["4b"]["vstar_bin_centers"]
    for sc, dash in (("4b", "-"), ("8b", "--")):
        d = t1[sc]
        pit, reaches, cap = d["pit"], d["reaches"], d["capability"]
        c_pit, c_reach = type_color("pit", sc), type_color("reaches", sc)

        axB.fill_between(xs, pit["ci_lo"], pit["ci_hi"], color=c_pit, alpha=0.13, zorder=2)
        axB.fill_between(xs, reaches["ci_lo"], reaches["ci_hi"], color=c_reach,
                         alpha=0.13, zorder=2)
        axB.plot(xs, pit["cer"], dash, marker="o", color=c_pit, lw=1.6, ms=3.4, zorder=4,
                 label="pit" if sc == "4b" else None)
        axB.plot(xs, reaches["cer"], dash, marker="o", color=c_reach, lw=1.6, ms=3.4, zorder=4,
                 label=DISPLAY["reaches"] if sc == "4b" else None)
        axB.errorbar([cap["vstar"]], [cap["cer"]],
                     yerr=[[cap["cer"] - cap["ci"][0]], [cap["ci"][1] - cap["cer"]]],
                     fmt="*", color=type_color("capability", sc), ms=9, mec=NEUTRAL, mew=0.6,
                     capsize=2.5, zorder=5,
                     label="capability gap" if sc == "4b" else None)

    axB.set_xlabel(r"best reachable value at step 0, $V^*_0$ (binned)")
    axB.set_ylabel("miss rate")
    axB.set_ylim(0, 0.9)

    axB.set_xlim(-0.11, 0.95)
    axB.set_title("(B) pit and near miss,\nbinned by $V^*_0$")

    axB.legend(loc="lower right", frameon=False, handlelength=1.1,
               handletextpad=0.4, labelspacing=0.3, borderpad=0.1)

    scales = ["4b", "8b"]
    xc = np.arange(2)
    bce = [calib[s]["bce_held"]["pit_vs_reaches"] for s in scales]
    mc = [calib[s]["mc_held"]["pit_vs_reaches"] for s in scales]
    bce_vals, bce_lo, bce_hi = ([d["auroc"] for d in bce], [d["ci_lo"] for d in bce],
                                 [d["ci_hi"] for d in bce])
    mc_vals, mc_lo, mc_hi = ([d["auroc"] for d in mc], [d["ci_lo"] for d in mc],
                              [d["ci_hi"] for d in mc])

    c_h = readout_color("h")
    for j, s in enumerate(scales):
        vals = [bce_vals[j], mc_vals[j]]
        los, his = [bce_lo[j], mc_lo[j]], [bce_hi[j], mc_hi[j]]

        face = "white" if s == "4b" else c_h
        bars = axC.bar(xc + (j - 0.5) * width, [v - 0.5 for v in vals], width, bottom=0.5,
                        yerr=asymmetric_yerr(vals, los, his), capsize=2.5, color=face,
                        edgecolor=c_h, linewidth=1.1, zorder=3, error_kw={"lw": 0.8, "ecolor": NEUTRAL},
                        label=MODEL_LABEL[s])
        _neutralize_caps(bars)

    stacked_value_labels(axC, (xc[0] - width / 2, xc[0] + width / 2), max(bce_hi),
                         (bce_vals[0], bce_vals[1]), above=True)
    stacked_value_labels(axC, (xc[1] + width / 2, xc[1] - width / 2), min(mc_lo),
                         (mc_vals[1], mc_vals[0]), above=False)

    axC.axhline(0.5, zorder=1, **REF["chance"])

    axC.text(1.52, 0.512, "chance", fontsize=MIN_FONT, color=NEUTRAL, ha="right", va="bottom",
             bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8}, zorder=4)
    axC.set_xticks(xc)
    axC.set_xticklabels(["confidence\nprobe", "combined\nprobe"])
    axC.set_xlim(-0.55, 1.55)

    axC.set_ylim(0.25, 0.87)
    axC.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    axC.set_ylabel(f"pit vs {DISPLAY['reaches']} AUROC")
    axC.set_title("(C) pit vs near miss\nAUROC, by probe")

    axC.legend(loc="lower left", ncol=1, frameon=False,
               handletextpad=0.4, handlelength=1.1, labelspacing=0.3, borderpad=0.1)

    for axis, position in zip((axA, axB, axC), AXES_POSITIONS):
        axis.set_position(position)

    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

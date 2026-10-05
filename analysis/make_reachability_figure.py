from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import torch

from .fig_style import apply, type_color, NEUTRAL, REF, FIG_W, MIN_FONT

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/fig_reachability.pdf"

SAVE_BBOX = (-0.041597222222222195, -0.04605034722222205, 5.404444444444444, 2.408336805555556)
AXES_LAYOUT = ([[0.08701285240464346, 0.17080378250591022, 0.24810722698642973, 0.6752009456264774], [-1.9500000000000002, 40.95], [-0.11, 1.38], [-20.0, 0.0, 20.0, 40.0, 60.0], [0.0, 0.5, 1.0]],
              [[0.41427039148190953, 0.17080378250591022, 0.24810722698642967, 0.6752009456264774], [-1.2, 24.5], [-0.11, 1.38], [-10.0, 0.0, 10.0, 20.0, 30.0], [0.0, 0.5, 1.0]],
              [[0.7415279305591755, 0.17080378250591022, 0.24810722698642973, 0.6752009456264774], [-1.1, 23.1], [-0.11, 1.38], [-10.0, 0.0, 10.0, 20.0, 30.0], [0.0, 0.5, 1.0]])

THETA = 0.1

apply()

EX = {
    "pit": "deepmath_shard4_ge7/12879__2a9eec603a4a",
    "near miss": "deepmath_shard3_ge7/4426__5e074a3ca3f6",
    "capability": "deepmath_shard0_ge7/7476__bc88060d454a",
}

PANELS = [
    ("capability", "Capability gap"),
    ("pit", "Pit"),
    ("near miss", "Near miss"),
]

def _fetch(name: str) -> str:
    return str(ROOT / "runs/mc_gold/campaign_final" / name)

def series(mc: dict, tid: str) -> tuple[list[int], list[float]]:
    d = mc[tid]
    idx = sorted(d.keys())
    return idx, [d[i]["p_correct_mc"] for i in idx]

def envelope(p: list[float]) -> list[float]:
    e = [0.0] * len(p)
    m = 0.0
    for i in range(len(p) - 1, -1, -1):
        m = max(m, p[i])
        e[i] = m
    return e

def main() -> None:
    mc = torch.load(_fetch("mc_ground_truth.pt"), map_location="cpu", weights_only=False)

    fig, axs = plt.subplots(1, 3, figsize=(FIG_W - 0.14, 2.35), sharey=True)
    for ax, (name, title) in zip(axs, PANELS):
        tid = EX[name]
        idx, p = series(mc, tid)
        e = envelope(p)
        col = type_color(name, "8b")
        ax.step(idx, e, where="post", color=col, lw=1.8, zorder=3)

        ax.plot(idx, p, "o-", color=NEUTRAL, lw=0.8, ms=3.2, alpha=0.55, zorder=2)
        ax.axhline(THETA, **REF["chance"])
        ax.set_title(title)
        ax.set_xlabel("reasoning step")

        ax.set_ylim(-0.11, 1.38)
        ax.set_yticks([0, 0.5, 1.0])
        if name == "pit":

            ax.set_xlim(-1.2, 24.5)
            fp = next(i for i, v in enumerate(e) if v <= THETA)
            xp = idx[fp]

            ax.axvline(xp, ymax=0.72, color=NEUTRAL, lw=1.4)

            ax.text(xp + 0.6, 0.72, "pit step", ha="left", va="center", color=NEUTRAL,
                    fontsize=MIN_FONT)

    axs[0].set_ylabel("value")

    axs[0].text(0.5, 0.55, "Qwen3-4B, gold collection", transform=axs[0].transAxes,
                fontsize=MIN_FONT, color=NEUTRAL, style="italic", ha="center", va="center")

    envelope_swatch = tuple(
        Line2D([0], [0], color=type_color(n, "8b"), lw=1.8) for n, _ in PANELS
    )
    legend_handles = [
        envelope_swatch,
        Line2D([0], [0], color=NEUTRAL, marker="o", lw=0.8, ms=3.2, alpha=0.55),
        Line2D([0], [0], **REF["chance"]),
    ]
    legend_labels = [
        r"envelope",
        r"rollout rate $\hat p_t$",
        r"threshold $\theta{=}0.1$",
    ]
    axs[0].legend(legend_handles, legend_labels,
                  handler_map={tuple: HandlerTuple(ndivide=len(PANELS), pad=0.0)},
                  frameon=False, loc="upper left", handlelength=2.1,
                  handletextpad=0.5, borderaxespad=0.15, labelspacing=0.3)
    for axis, (position, xlim, ylim, xticks, yticks) in zip(axs, AXES_LAYOUT):
        axis.set_position(position)
        axis.set_xticks(xticks)
        axis.set_yticks(yticks)
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
    fig.savefig(OUT, bbox_inches=Bbox.from_extents(*SAVE_BBOX))
    plt.close(fig)

if __name__ == "__main__":
    main()

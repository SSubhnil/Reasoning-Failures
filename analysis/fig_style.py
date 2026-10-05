from __future__ import annotations

import matplotlib as mpl

TYPE = {
    "capability": {"8b": "#C34E02", "4b": "#E68E68"},
    "pit":        {"8b": "#005AAC", "4b": "#598CC9"},
    "near miss":  {"8b": "#008A59", "4b": "#5FB68F"},
}
TYPE["reaches"] = TYPE["near miss"]

READOUT = {
    "z": {"8b": "#942A94", "4b": "#BE71BC"},
    "h": {"8b": "#7A6E14", "4b": "#7A6E14"},
}

NEUTRAL = "#3F3F3F"
BAND = "#E4E4E4"

REF = {
    "chance":    {"ls": (0, (4, 3)), "lw": 0.9, "color": NEUTRAL},
    "floor":     {"ls": (0, (1, 2)), "lw": 0.9, "color": NEUTRAL},
    "required":  {"ls": (0, (6, 2)), "lw": 1.4, "color": NEUTRAL},
    "ceiling":   {"ls": (0, (5, 2, 1, 2)), "lw": 0.9, "color": NEUTRAL},
    "operating": {"ls": (0, (2, 2)), "lw": 1.1, "color": NEUTRAL},
}

def type_color(kind: str, scale: str = "8b") -> str:
    return TYPE[kind][scale]

def readout_color(rep: str, scale: str = "8b") -> str:
    return READOUT[rep][scale]

def mark_kw(color: str, scale: str) -> dict:
    if scale == "4b":
        return {"markerfacecolor": "white", "markeredgecolor": color,
                "markeredgewidth": 1.2, "color": color}
    return {"markerfacecolor": color, "markeredgecolor": color,
            "markeredgewidth": 0.0, "color": color}

TITLE_FONT = 8.5
LABEL_FONT = 8.0
TICK_FONT = 7.0
LEGEND_FONT = 7.0
MIN_FONT = 7.0

def apply() -> None:
    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.size": LABEL_FONT,
        "axes.titlesize": TITLE_FONT,
        "axes.labelsize": LABEL_FONT,
        "xtick.labelsize": TICK_FONT,
        "ytick.labelsize": TICK_FONT,
        "legend.fontsize": LEGEND_FONT,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "lines.linewidth": 1.8,
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

FIG_W = 5.5

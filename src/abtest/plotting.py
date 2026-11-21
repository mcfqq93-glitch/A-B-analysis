"""Consistent, minimal matplotlib styling for all project figures."""
import matplotlib.pyplot as plt

CONTROL = "#2a78d6"     # blue
TREATMENT = "#eb6834"   # orange
ACCENT = "#1baf7a"      # aqua (3rd series)
MUTED = "#52514e"
GRID = "#e6e5e1"
GROUP_COLORS = {"control": CONTROL, "treatment": TREATMENT}


def set_style():
    plt.rcParams.update({
        "figure.figsize": (8, 4.2), "figure.dpi": 110, "savefig.dpi": 150,
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb", "savefig.bbox": "tight",
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "lines.linewidth": 2,
        "legend.frameon": False, "font.size": 10,
    })

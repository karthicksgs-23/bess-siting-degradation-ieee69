"""Figures that do not depend on results: feeder single-line diagram and method flowchart."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from net69 import load_case

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10})


def layout():
    """Hand layout: main feeder 1-27 on y=0, laterals as branches."""
    pos = {}
    for i, b in enumerate(range(1, 28)):
        pos[b] = (i, 0)
    for i, b in enumerate(range(28, 36)):
        pos[b] = (2 + 0.0 + i, 1.2)          # from bus 3
    for i, b in enumerate(range(36, 47)):
        pos[b] = (2 + i, 2.4)                # from bus 3
    for i, b in enumerate(range(47, 51)):
        pos[b] = (3 + i, -1.2)               # from bus 4
    pos[51], pos[52] = (6.2, -2.4), (5.2, -2.4)  # from bus 8 (drawn to the left)
    for i, b in enumerate(range(53, 66)):
        pos[b] = (8 + i, -3.6)               # from bus 9
    pos[66], pos[67] = (10, 1.2), (11, 1.2)  # from bus 11
    pos[68], pos[69] = (11.2, -1.2), (12.2, -1.2)  # from bus 12
    return pos


# BESS sites chosen by the planning runs (results/analysis.json): bus -> (label, lower-cost case only)
SITES = {64: ("Case A: M0\nCase B: all", False), 65: ("Case A:\nM1, M2", False),
         49: ("", True), 50: ("", True), 57: ("M0", True), 61: ("M1, M2", True)}


def feeder(chosen=None, path=FIG / "fig1_feeder.png"):
    net = load_case()
    pos = layout()
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for f, t in zip(net["f"] + 1, net["t"] + 1):
        (x0, y0), (x1, y1) = pos[f], pos[t]
        if y0 != y1:
            ax.plot([x0, x0, x1], [y0, y1, y1], color="0.35", lw=1.1, zorder=1)
        else:
            ax.plot([x0, x1], [y0, y1], color="0.35", lw=1.1, zorder=1)
    for b, (x, y) in pos.items():
        ax.scatter(x, y, s=14, color="black", zorder=3)
        if b in (3, 4, 8, 9, 11, 12, 18, 27, 35, 46, 49, 50, 52, 57, 61, 64, 65, 67, 69):
            ax.annotate(str(b), (x, y), textcoords="offset points", xytext=(-11, 8) if b in SITES else (-7, 5),
                        ha="center", fontsize=8)
    ax.annotate("1", pos[1], textcoords="offset points", xytext=(-12, -3), ha="center", fontsize=8)
    # substation
    ax.scatter(*pos[1], marker="s", s=120, color="#444444", zorder=4)
    ax.annotate("Substation\n12.66 kV, 1.02 pu", pos[1], textcoords="offset points", xytext=(-4, -24), fontsize=8, ha="left")
    for b in (18, 49, 64):
        ax.scatter(*pos[b], marker="*", s=200, color="#E69F00", edgecolor="black", lw=0.5, zorder=5)
    chosen = chosen if chosen is not None else SITES
    for b, (lab, low) in chosen.items():
        x, y = pos[b]
        ax.scatter(x, y, marker="s" if low else "o", s=190, facecolor="none",
                   edgecolor="#CC79A7" if low else "#0072B2", lw=1.8, ls="--" if low else "-", zorder=6)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext={64: (-24, -26), 65: (20, -26)}.get(b, (0, -17)),
                    ha="center", va="center" if b in (64, 65) else "baseline",
                    color="#CC79A7" if low else "#0072B2", fontsize=7.5)
    ax.scatter([], [], marker="*", s=120, color="#E69F00", edgecolor="black", lw=0.5, label="PV plant (18: 0.5, 49: 0.6, 64: 0.5 MWp)")
    ax.scatter([], [], marker="o", s=90, facecolor="none", edgecolor="#0072B2", lw=1.8,
               label="BESS site at ₹14,000/kWh (0.6 MW, 2.25 MWh)")
    ax.scatter([], [], marker="s", s=90, facecolor="none", edgecolor="#CC79A7", lw=1.8, ls="--",
               label="Additional BESS sites at ₹9,000–7,000/kWh")
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    ax.set_axis_off()
    ax.set_xlim(-1, 27.5); ax.set_ylim(-4.4, 3.2)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def flowchart(path=FIG / "fig3_method.png"):
    fig, ax = plt.subplots(figsize=(10, 2.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.0); ax.set_axis_off()
    boxes = [
        (0.1, 1.75, "Inputs\nIEEE 69-bus, PVGIS PV,\nload shapes, prices"),
        (2.6, 1.75, "Planning MILP\n(LinDistFlow + losses)\nM0 / M1 / M2"),
        (5.1, 1.75, "Plan + dispatch\nsites, MW, MWh,\nhourly SOC"),
        (7.6, 1.75, "Ex-post audit\nrainflow ageing +\nexact AC power flow"),
        (5.1, 0.15, "Re-dispatch\nfixed plan,\nM2 operation"),
        (7.6, 0.15, "Regret and\nproxy error"),
    ]
    for x, y, t in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 2.2, 1.1, boxstyle="round,pad=0.04", fc="#EAF1FB", ec="#2F5597", lw=1.2))
        ax.text(x + 1.1, y + 0.55, t, ha="center", va="center", fontsize=11)
    arr = dict(arrowstyle="-|>", mutation_scale=14, color="#2F5597", lw=1.3)
    for (x0, y0), (x1, y1) in [((2.35, 2.3), (2.6, 2.3)), ((4.85, 2.3), (5.1, 2.3)), ((7.35, 2.3), (7.6, 2.3)),
                               ((6.2, 1.7), (6.2, 1.3)), ((7.35, 0.7), (7.6, 0.7)), ((8.7, 1.7), (8.7, 1.3))]:
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), **arr))
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    feeder()
    flowchart()
    print("ok")

"""Result-dependent figures (and the input figure)."""
import json
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from net69 import load_case, bfs
from inputs import build, SEASONS

HERE = Path(__file__).parent
FIG = HERE / "figures"
RES = HERE / "results"
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10})
COL = {"M0": "#D55E00", "M1": "#0072B2", "M2": "#009E73", "none": "#555555"}
SCOL = {"summer": "#D55E00", "monsoon": "#0072B2", "winter": "#009E73"}


def fig_inputs():
    net = load_case()
    a = build(net, price_case="tod")
    b = build(net, price_case="market")
    h = np.arange(24)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.1))
    for k, s in enumerate(SEASONS):
        sl = slice(24 * k, 24 * k + 24)
        ax[0].plot(h, a["P"][sl].sum(axis=1), color=SCOL[s], label=s.capitalize())
        ax[1].plot(h, a["PV"][sl].sum(axis=1), color=SCOL[s], label=s.capitalize())
        ax[2].step(h, b["price"][sl] / 1000, where="post", color=SCOL[s], label=f"Case B, {s}")
    ax[2].step(h, a["price"][:24] / 1000, where="post", color="black", ls="--", label="Case A (all seasons)")
    ax[0].set_ylabel("Feeder load (MW)"); ax[1].set_ylabel("PV output, 1.6 MWp (MW)")
    ax[2].set_ylabel("Purchase price (₹/kWh)")
    for x, t in zip(ax, ["(a)", "(b)", "(c)"]):
        x.set_xlabel("Hour (IST)"); x.set_xticks(range(0, 25, 6)); x.grid(alpha=0.3)
        x.set_title(t, loc="left", fontsize=10)
    ax[0].legend(frameon=False, fontsize=8); ax[2].legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig(FIG / "fig2_inputs.png", dpi=300); plt.close(fig)


def load_pkl(name):
    return pickle.load(open(RES / name, "rb"))


def fig_voltage(pkl="dispatch_market.pkl", out="fig4_voltage.png"):
    d = load_pkl(pkl)
    net = load_case(); data = build(net, price_case="market")
    V0, _, _ = bfs(net, data["P"] - data["PV"], data["Q"], v0=1.02)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    x = np.arange(72)
    ax.plot(x, V0[:, 64], color=COL["none"], lw=1.4, label="No BESS")
    for m in ["M0", "M2"]:
        V = d["keep"][f"{m}-own"]["V"]
        ax.plot(x, V[:, 64], color=COL[m], lw=1.4, ls="-" if m == "M2" else "--", label=f"{m} plan")
    ax.axhline(0.95, color="black", lw=0.8, ls=":")
    for k in (24, 48):
        ax.axvline(k, color="0.6", lw=0.6)
    for k, s in enumerate(SEASONS):
        ax.text(24 * k + 12, 0.97, s.capitalize(), ha="center", va="top", fontsize=9.5, transform=ax.get_xaxis_transform())
    ax.set_xlim(0, 72); ax.set_xticks(range(0, 73, 6)); ax.set_xticklabels([str(v % 24) for v in range(0, 73, 6)])
    ax.set_xlabel("Hour of representative day"); ax.set_ylabel("Voltage at bus 65 (pu)")
    ax.legend(frameon=False, ncol=3, loc="lower right", fontsize=8.5); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / out, dpi=300); plt.close(fig)


def fig_soc(pkl="dispatch_market.pkl", out="fig5_soc.png", title_suffix=""):
    d = load_pkl(pkl)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    x = np.arange(72)
    for m in ["M0", "M1", "M2"]:
        k = d["keep"][f"{m}-own"]
        tot_soc = k["soc"].sum(axis=1)
        E = sum(v[1] * 0.25 for v in k["plan"].values())
        ax.plot(x, 100 * tot_soc / E, color=COL[m], lw=1.4, label=f"{m} ({E:.2f} MWh)")
    for k in (24, 48):
        ax.axvline(k, color="0.6", lw=0.6)
    ax.set_xlim(0, 72); ax.set_xticks(range(0, 73, 6)); ax.set_xticklabels([str(v % 24) for v in range(0, 73, 6)])
    ax.set_xlabel("Hour of representative day (summer | monsoon | winter)"); ax.set_ylabel("State of charge (%)")
    ax.legend(frameon=False, ncol=3, fontsize=8.5, loc="upper left"); ax.grid(alpha=0.3); ax.set_ylim(-3, 108)
    fig.tight_layout(); fig.savefig(FIG / out, dpi=300); plt.close(fig)


def fig_depth(pkls=("dispatch_market.pkl",), out="fig6_depth.png"):
    bins = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    width = 0.028
    for j, m in enumerate(["M0", "M1", "M2"]):
        d = load_pkl(pkls[0])
        dl = []
        for b, lst in d["keep"][f"{m}-own"]["depth_log"].items():
            dl += lst
        if not dl:
            continue
        dd = np.array([x for x, _ in dl]); ww = np.array([c for _, c in dl])
        hist, _ = np.histogram(dd, bins=bins, weights=ww)
        ax.bar(bins[:-1] + 0.05 + (j - 1) * width, hist, width=width, color=COL[m], label=m)
    ax.set_xlabel("Cycle depth (fraction of capacity)"); ax.set_ylabel("Cycles per year")
    ax.set_xticks(np.linspace(0, 1, 6)); ax.legend(frameon=False); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(FIG / out, dpi=300); plt.close(fig)


def fig_capex(out="fig7_capex.png"):
    A = json.loads((RES / "analysis.json").read_text())
    base = [c for c in A if c["group"] == "base" and c["price_case"] == "market"][0]
    cases = [(14.0, base)] + [(c["c_en"] / 1e6, c) for c in A if c["group"] == "capex"]
    cases.sort(key=lambda t: -t[0])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
    x = np.arange(len(cases)); w = 0.26
    for j, m in enumerate(["M0", "M1", "M2"]):
        E = [c["plans"][m]["E_tot"] for _, c in cases]
        R = [c["plans"][m]["regret_total"] / 1e6 for _, c in cases]
        ax[0].bar(x + (j - 1) * w, E, w, color=COL[m], label=m)
        ax[1].bar(x + (j - 1) * w, R, w, color=COL[m], label=m)
    for a in ax:
        a.set_xticks(x); a.set_xticklabels([f"₹{v:,.0f}k" for v, _ in cases])
        a.set_xlabel("Battery energy cost (per kWh)"); a.grid(alpha=0.3, axis="y")
    ax[0].set_ylabel("Installed energy (MWh)"); ax[1].set_ylabel("Audited regret (₹ million/yr)")
    ax[0].set_title("(a)", loc="left", fontsize=10); ax[1].set_title("(b)", loc="left", fontsize=10)
    ax[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / out, dpi=300); plt.close(fig)


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["inputs"]
    if "inputs" in what:
        fig_inputs()
    if "base" in what:
        fig_voltage(); fig_soc(); fig_depth()
    if "capex" in what:
        fig_capex()
    print("figures ok")

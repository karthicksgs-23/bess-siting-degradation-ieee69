"""Time-series inputs: three representative days (summer, monsoon, winter).
PV  : PVGIS-ERA5 hourly output for Hyderabad (17.385 N, 78.487 E), 2020, 1 kWp,
      17 deg tilt, south, 14 % system loss; averaged per season hour (IST).
Load: seasonal normalised shapes for an urban mixed (residential + commercial) feeder
      (assumed shapes, defined below), scaled to the Baran-Wu nominal loads.
Price: ToD purchase price - base x {0.8 solar hours 09-17, 1.2 peak 18-22, 1.0 otherwise},
      following the ToD structure of the Electricity (Rights of Consumers) Amendment Rules 2023."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
SEASONS = ["summer", "monsoon", "winter"]
SEASON_MONTHS = {"summer": [3, 4, 5], "monsoon": [6, 7, 8, 9], "winter": [10, 11, 12, 1, 2]}
DAY_WEIGHT = {"summer": 92, "monsoon": 122, "winter": 151}  # days per year (sum 365)
PEAK_FACTOR = {"summer": 1.00, "monsoon": 0.90, "winter": 0.82}

LOAD_SHAPE = {  # hour 0..23 IST, normalised to seasonal peak = 1
    "summer":  [0.72, 0.68, 0.65, 0.63, 0.62, 0.63, 0.67, 0.72, 0.78, 0.83, 0.87, 0.90,
                0.92, 0.93, 0.93, 0.92, 0.90, 0.90, 0.94, 0.98, 1.00, 0.98, 0.90, 0.80],
    "monsoon": [0.66, 0.62, 0.60, 0.58, 0.58, 0.61, 0.67, 0.74, 0.79, 0.82, 0.84, 0.85,
                0.85, 0.85, 0.84, 0.83, 0.84, 0.88, 0.95, 1.00, 1.00, 0.96, 0.86, 0.74],
    "winter":  [0.55, 0.52, 0.50, 0.50, 0.52, 0.60, 0.72, 0.85, 0.90, 0.86, 0.80, 0.76,
                0.73, 0.71, 0.70, 0.71, 0.75, 0.85, 0.97, 1.00, 0.96, 0.88, 0.75, 0.62],
}
BASE_PRICE = 5500.0  # INR/MWh (average DISCOM power-purchase cost, assumed)


def tod_multiplier(h):
    if 9 <= h < 17:
        return 0.8
    if 18 <= h < 22:
        return 1.2
    return 1.0


def pv_profiles():
    js = json.loads((HERE / "data" / "pvgis_hyd_2020.json").read_text())
    rows = js["outputs"]["hourly"]
    acc = {s: np.zeros(24) for s in SEASONS}
    cnt = {s: np.zeros(24) for s in SEASONS}
    for r in rows:
        tstr = r["time"]  # 'YYYYMMDD:HHMM' UTC
        month = int(tstr[4:6])
        hh, mm = int(tstr[9:11]), int(tstr[11:13])
        ist = (hh * 60 + mm + 330) / 60.0
        day_shift = ist >= 24
        h = int(ist % 24)
        if day_shift:  # crossing midnight only matters for month at year end; ignore
            pass
        s = next(k for k, v in SEASON_MONTHS.items() if month in v)
        acc[s][h] += r["P"] / 1000.0  # kW per kWp -> p.u. of kWp
        cnt[s][h] += 1
    return {s: acc[s] / np.maximum(cnt[s], 1) for s in SEASONS}


# Scenario B: market-exposed purchase price (INR/MWh), stylised on the "duck-curve" shape of the
# IEX day-ahead market (cheap solar hours, expensive evening ramp).  A SCENARIO, not measured data.
MARKET = {
    "summer":  [4500]*6 + [5000]*3 + [3000]*8 + [6000] + [9500]*5 + [6500],
    "monsoon": [4000]*6 + [4500]*3 + [3800]*8 + [5500] + [8000]*5 + [5500],
    "winter":  [4000]*6 + [5500]*3 + [2800]*8 + [6000] + [8500]*5 + [5000],
}


def build(net, pv_sites=None, price_case="tod"):
    """Returns dict with T x n arrays (MW, MVAr), price (INR/MWh), weights (days)."""
    if pv_sites is None:
        pv_sites = {18: 0.5, 49: 0.6, 64: 0.5}  # bus (1-indexed): MWp
    pv = pv_profiles()
    P, Q, PV, price, w, lbl = [], [], [], [], [], []
    for s in SEASONS:
        for h in range(24):
            f = PEAK_FACTOR[s] * LOAD_SHAPE[s][h]
            P.append(net["Pd"] * f)
            Q.append(net["Qd"] * f)
            pvrow = np.zeros(net["n"])
            for b, mw in pv_sites.items():
                pvrow[b - 1] = mw * pv[s][h]
            PV.append(pvrow)
            price.append(BASE_PRICE * tod_multiplier(h) if price_case == "tod" else MARKET[s][h])
            w.append(DAY_WEIGHT[s])
            lbl.append((s, h))
    return {"P": np.array(P), "Q": np.array(Q), "PV": np.array(PV),
            "price": np.array(price), "w": np.array(w, float), "lbl": lbl,
            "pv_cf": pv, "pv_sites": pv_sites}


if __name__ == "__main__":
    from net69 import load_case, bfs
    net = load_case()
    d = build(net)
    for s in SEASONS:
        print(s, "PV daily kWh/kWp = %.2f" % d["pv_cf"][s].sum(), np.round(d["pv_cf"][s], 2))
    V, _, lt = bfs(net, d["P"] - d["PV"], d["Q"])
    for i, (s, h) in enumerate(d["lbl"]):
        if h in (3, 12, 20):
            print(s, h, "Vmin %.4f @%d  Vmax %.4f  loss %.1f kW  net head %.2f MW" % (
                V[i, 1:].min(), V[i].argmin() + 1, V[i].max(), lt[i] * 1e3,
                (d["P"][i] - d["PV"][i]).sum() + lt[i]))
    print("hours with Vmin<0.95:", int((V.min(axis=1) < 0.95).sum()), "of", len(V))
    ann_loss = (lt * d["w"]).sum()
    print("annual loss %.1f MWh" % ann_loss)

"""Re-solve M2 for the low-capex cases where the 20-candidate MILP found no incumbent.
Candidate set = buses used by the M0 and M1 plans + buses with energy in the M2 LP relaxation (> 0.05 MWh).
Then audit, re-dispatch all plans with M2 and rewrite the capex case in capex_results.json."""
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np

from net69 import load_case
from inputs import build
from model import solve, DEFAULT
from audit import audit
from run_study import candidates, summarise, OUT


def main(ce):
    p = dict(DEFAULT)
    q = dict(p, c_en=ce, c_cell=ce * 9.0 / 14.0, c_pow=p["c_pow"] * ce / 14.0e6)
    label = f"capex{ce/1e6:.0f}"
    net = load_case(); data = build(net, price_case="market")
    cand20 = candidates(net, data, q)
    plans = {}
    for m in ["M0", "M1"]:
        r = pickle.load(open(OUT / "cache" / f"{label}_market_{m}.pkl", "rb"))
        plans[m] = r
    rel = solve("M2", net, data, q, cand=cand20, relax=True, time_limit=900)
    relsites = [b for b, e in rel["relaxed_E"].items() if e > 0.05]
    cset = sorted(set(relsites) | {b for m in plans for b in plans[m]["plan"]})
    t0 = time.time()
    print(label, "relaxed E (1-idx):", {b + 1: round(e, 2) for b, e in rel["relaxed_E"].items() if e > 0.01},
          "-> M2 candidates", [b + 1 for b in cset], flush=True)
    r2 = solve("M2", net, data, q, cand=np.array(cset), time_limit=1800)
    print(label, "M2 status", r2["status"], r2["message"], "gap", r2["mip_gap"], "t", round(time.time() - t0), flush=True)
    if "plan" not in r2:
        return
    pickle.dump(r2, open(OUT / "cache" / f"{label}_market_M2_repair.pkl", "wb"))
    rows = []
    for m, r in list(plans.items()) + [("M2", r2)]:
        a = audit(r, net, data, q)
        s = summarise(f"{m}-own", r, a, q); rows.append(s)
        print(f"  {m}-own {s['plan']} proxy {s['proxy_cyc']/1e6:.3f} rf {s['cyc_ref']/1e6:.3f} total {s['total_ref']/1e6:.3f}", flush=True)
    for m, r in list(plans.items()) + [("M2", r2)]:
        rr = solve("M2", net, data, q, cand=cand20, fixed=r["plan"], time_limit=900)
        rr["model"] = m
        a = audit(rr, net, data, q)
        s = summarise(f"{m}-plan/M2-op", rr, a, q); rows.append(s)
        print(f"  {m}-redisp total {s['total_ref']/1e6:.3f} cyc {s['cyc_ref']/1e6:.3f}", flush=True)
    res = dict(price_case="market", label=label, rows=rows, cand=[int(b) + 1 for b in cand20],
               m2_cand=[int(b) + 1 for b in cset], c_en=ce, c_cell=q["c_cell"], c_pow=q["c_pow"],
               nobat=None)
    f = OUT / f"capex_repair_{label}.json"
    f.write_text(json.dumps(res, indent=1, default=float))
    print("saved", f)


if __name__ == "__main__":
    main(float(sys.argv[1]) * 1e6)

"""Post-processing: polish time-limited M2 plans, compute proxy error and regret, write tables."""
import json
from pathlib import Path

RES = Path(__file__).parent / "results"


def m2_objective(row_redisp):
    """M2 planning objective of a fixed plan = operating objective (energy, loss, M2 wear) + capex + calendar."""
    return row_redisp["planner_obj"] + row_redisp["ann_capex"] + row_redisp["cal"]


def process_case(case):
    rows = {r["tag"]: r for r in case["rows"]}
    own = {m: rows.get(f"{m}-own") for m in ["M0", "M1", "M2"]}
    red = {m: rows.get(f"{m}-plan/M2-op") for m in ["M0", "M1", "M2"]}
    out = dict(price_case=case["price_case"], label=case["label"], nobat=case["nobat"])
    # polish M2: pick the plan with the lowest M2 objective among M2 incumbent and the other plans
    m2_obj = {m: m2_objective(red[m]) for m in red if red[m]}
    best = min(m2_obj, key=m2_obj.get)
    polished = own["M2"]["status"] != 0 and best != "M2" and m2_obj[best] < m2_obj["M2"] - 1e-6
    out["m2_status"] = own["M2"]["status"]; out["m2_gap"] = own["M2"]["mip_gap"]
    out["m2_polished_from"] = best if polished else None
    plans = {}
    for m in ["M0", "M1", "M2"]:
        src = best if (m == "M2" and polished) else m
        o = own[src] if not (m == "M2" and polished) else red[src]  # polished M2 operates with M2 cost
        r = red[src]
        plans[m] = dict(plan=r["plan"], P_tot=r["P_tot"], E_tot=r["E_tot"],
                        proxy_cyc=(o["proxy_cyc"] if m != "M2" or not polished else r["proxy_cyc"]),
                        cyc_ref_own=o["cyc_ref"], total_own=o["total_ref"],
                        total_redisp=r["total_ref"], cyc_ref_redisp=r["cyc_ref"],
                        vmin_own=o["vmin"], loss_own=o["loss_mwh"], fec=o["fec"], life=o["life"],
                        energy_own=o["energy"], capex=o["ann_capex"], cal=o["cal"])
    best_red = min(p["total_redisp"] for p in plans.values())
    best_own = min(p["total_own"] for p in plans.values())
    for m, p in plans.items():
        p["regret_plan"] = p["total_redisp"] - best_red
        p["regret_total"] = p["total_own"] - best_own
        p["proxy_err"] = p["proxy_cyc"] - p["cyc_ref_own"]
        p["proxy_err_pct"] = 100 * p["proxy_err"] / p["cyc_ref_own"] if p["cyc_ref_own"] else float("nan")
    out["plans"] = plans
    return out


def load_all():
    allc = []
    f = RES / "base_results.json"
    if f.exists():
        allc += [dict(process_case(c), group="base") for c in json.loads(f.read_text())]
    for f in sorted(RES.glob("sens_results_kp*.json")):
        for c in json.loads(f.read_text()):
            d = process_case(c); d.update(group="sens", kp=c["kp"], c_cell=c["c_cell"]); allc.append(d)
    f = RES / "capex_results.json"
    caps = {}
    if f.exists():
        for c in json.loads(f.read_text()):
            caps[c["label"]] = c
    for f in RES.glob("capex_repair_*.json"):
        c = json.loads(f.read_text()); caps[c["label"]] = c
    for lab, c in sorted(caps.items(), key=lambda kv: -kv[1]["c_en"]):
        tags = {r["tag"] for r in c["rows"]}
        if "M2-own" not in tags:
            print("skip incomplete", lab); continue
        d = process_case(c); d.update(group="capex", c_en=c["c_en"], c_cell=c["c_cell"], m2_cand=c.get("m2_cand"))
        allc.append(d)
    return allc


if __name__ == "__main__":
    allc = load_all()
    (RES / "analysis.json").write_text(json.dumps(allc, indent=1, default=float))
    for c in allc:
        print(f"== {c['group']} {c['label']} {c['price_case']}  M2 status {c['m2_status']} gap {c['m2_gap']:.1e} polished<-{c['m2_polished_from']}")
        for m, p in c["plans"].items():
            print(f"  {m}: {p['plan']}  proxy {p['proxy_cyc']/1e6:.3f} rf {p['cyc_ref_own']/1e6:.3f} ({p['proxy_err_pct']:+.1f}%)"
                  f" own {p['total_own']/1e6:.3f} redisp {p['total_redisp']/1e6:.3f} regret_plan {p['regret_plan']/1e3:.1f}k"
                  f" regret_tot {p['regret_total']/1e3:.1f}k fec {p['fec']} life {p['life']} vmin {p['vmin_own']:.4f}")

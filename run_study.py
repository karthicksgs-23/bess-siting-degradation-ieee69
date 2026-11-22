"""Main study: plan with M0/M1/M2 under two price cases, audit each plan with rainflow + AC
power flow, then re-dispatch every plan with the cycle-depth proxy (M2) to separate the
siting/sizing error from the dispatch error."""
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np

from net69 import load_case, bfs
from inputs import build
from model import solve, DEFAULT
from audit import audit

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)


def candidates(net, data, p):
    V, _, _ = bfs(net, data["P"] - data["PV"], data["Q"], v0=p["v0"])
    weak = list(np.argsort(V.min(axis=0))[:15])
    big = list(np.argsort(-net["Pd"])[:6])
    pv = [b - 1 for b in data["pv_sites"]]
    return np.array(sorted(set(int(b) for b in weak + big + pv) - {0}))


def summarise(tag, r, a, p):
    plan1 = {b + 1: dict(P_MW=round(v[0] * p["p_mod"], 2), E_MWh=round(v[1] * p["e_mod"], 2))
             for b, v in r["plan"].items()}
    return dict(tag=tag, model=r["model"], status=int(r["status"]), mip_gap=float(r["mip_gap"] or 0.0),
                t_solve=round(r["t_solve"], 1), plan=plan1,
                P_tot=a["P_tot"], E_tot=a["E_tot"],
                planner_obj=r["obj"], proxy_cyc=r.get("proxy_cyc", 0.0), cyc_ref=float(a["cyc_ref"]),
                ann_capex=a["ann_capex"], cal=a["cal"], energy=a["energy"], total_ref=float(a["total_ref"]),
                loss_mwh=a["loss_mwh"], vmin=float(a["vmin"]), n_viol=a["n_viol"],
                life={b + 1: round(float(v), 2) for b, v in a["life"].items()},
                fec={b + 1: round(float(v), 1) for b, v in a["fec"].items()},
                simul=a["simul"], slack=float(r["slack"]))


def run_case(price_case, p, tl=900, label="base", log=print):
    net = load_case()
    data = build(net, price_case=price_case)
    cand = candidates(net, data, p)
    log(f"[{label}/{price_case}] candidates (1-idx): {[int(b) + 1 for b in cand]}")
    rows, keep = [], {}
    # no-battery reference
    V0, _, lt0 = bfs(net, data["P"] - data["PV"], data["Q"], v0=p["v0"])
    head0 = (data["P"] - data["PV"]).sum(axis=1) + lt0
    nobat = dict(energy=float(np.sum(data["w"] * data["price"] * head0)),
                 loss_mwh=float(np.sum(data["w"] * lt0)), vmin=float(V0[:, 1:].min()),
                 n_viol=int((V0[:, 1:].min(axis=1) < p["vmin"]).sum()))
    plans = {}
    for m in ["M0", "M1", "M2"]:
        ck = OUT / "cache" / f"{label}_{price_case}_{m}.pkl"
        ck.parent.mkdir(exist_ok=True)
        if ck.exists():
            r = pickle.load(open(ck, "rb"))
        else:
            r = solve(m, net, data, p, cand=cand, time_limit=tl)
            if "plan" in r:
                pickle.dump(r, open(ck, "wb"))
        if "plan" not in r:
            log(f"  {m} FAILED {r['message']}")
            continue
        a = audit(r, net, data, p)
        s = summarise(f"{m}-own", r, a, p)
        rows.append(s); plans[m] = r["plan"]
        keep[f"{m}-own"] = dict(r=r, a=a)
        log(f"  {m}: plan {s['plan']} gap {s['mip_gap']:.2e} t {s['t_solve']}s | proxy cyc {s['proxy_cyc']/1e6:.3f} "
            f"vs rainflow {s['cyc_ref']/1e6:.3f} MINR | total_ref {s['total_ref']/1e6:.3f} | vmin {s['vmin']:.4f}")
    # re-dispatch each plan with the cycle-depth proxy (operation only)
    for m, plan in plans.items():
        r = solve("M2", net, data, p, cand=cand, fixed=plan, time_limit=tl)
        r["model"] = f"{m}"
        a = audit(r, net, data, p)
        s = summarise(f"{m}-plan/M2-op", r, a, p)
        rows.append(s); keep[f"{m}-redisp"] = dict(r=r, a=a)
        log(f"  {m} plan re-dispatched with M2: total_ref {s['total_ref']/1e6:.3f} cyc_ref {s['cyc_ref']/1e6:.3f} vmin {s['vmin']:.4f}")
    return dict(price_case=price_case, label=label, nobat=nobat, rows=rows,
                cand=[int(b) + 1 for b in cand]), keep, data


if __name__ == "__main__":
    logf = open(OUT / "run_log.txt", "a", encoding="utf8")

    def log(msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    mode = sys.argv[1] if len(sys.argv) > 1 else "base"
    p = dict(DEFAULT)
    if mode == "base":
        allres = []
        for pc in ["tod", "market"]:
            res, keep, data = run_case(pc, p, log=log)
            allres.append(res)
            with open(OUT / f"dispatch_{pc}.pkl", "wb") as fh:
                pickle.dump(dict(keep={k: dict(plan=v["r"]["plan"], ch=v["r"]["ch"], dis=v["r"]["dis"],
                                               soc=v["r"]["soc"], cand=v["r"]["cand"], V=v["a"]["V"],
                                               depth_log=v["a"]["depth_log"], loss_t=v["a"]["loss_t"])
                                      for k, v in keep.items()},
                                 price=data["price"], lbl=data["lbl"], w=data["w"],
                                 pv=data["PV"], P=data["P"]), fh)
        (OUT / "base_results.json").write_text(json.dumps(allres, indent=1, default=float))
        log("BASE DONE")
    elif mode == "sens":
        allres = []
        kps = [float(sys.argv[2])] if len(sys.argv) > 2 else [1.2, 1.8, 2.4]
        for kp in kps:
            for cc in [6.0e6, 9.0e6, 12.0e6]:
                q = dict(p, kp=kp, c_cell=cc)
                res, _, _ = run_case("market", q, tl=600, label=f"kp{kp}_cc{cc/1e6:.0f}", log=log)
                res.update(kp=kp, c_cell=cc)
                allres.append(res)
                (OUT / f"sens_results_kp{kps[0] if len(kps)==1 else 'all'}.json").write_text(json.dumps(allres, indent=1, default=float))
        log("SENS DONE")
    elif mode == "capex":
        allres = []
        for ce in [11.0e6, 9.0e6, 7.0e6]:
            q = dict(p, c_en=ce, c_cell=ce * 9.0 / 14.0, c_pow=p["c_pow"] * ce / 14.0e6)
            res, keep, data = run_case("market", q, tl=900, label=f"capex{ce/1e6:.0f}", log=log)
            res.update(c_en=ce, c_cell=q["c_cell"], c_pow=q["c_pow"])
            allres.append(res)
            with open(OUT / f"dispatch_capex{ce/1e6:.0f}.pkl", "wb") as fh:
                pickle.dump(dict(keep={k: dict(plan=v["r"]["plan"], ch=v["r"]["ch"], dis=v["r"]["dis"],
                                               soc=v["r"]["soc"], cand=v["r"]["cand"], V=v["a"]["V"],
                                               depth_log=v["a"]["depth_log"]) for k, v in keep.items()},
                                 price=data["price"], lbl=data["lbl"], w=data["w"]), fh)
            (OUT / "capex_results.json").write_text(json.dumps(allres, indent=1, default=float))
        log("CAPEX DONE")
    elif mode == "fullcand":
        net = load_case(); data = build(net, price_case="market")
        for m in ["M0", "M2"]:
            r = solve(m, net, data, p, time_limit=1800)
            a = audit(r, net, data, p) if "plan" in r else None
            plan1 = {b + 1: v for b, v in r.get("plan", {}).items()}
            tot = a["total_ref"] / 1e6 if a else float("nan")
            log(f"[fullcand/{m}] status {r['status']} gap {r['mip_gap']} obj {r['obj']/1e6:.4f} "
                f"plan(np,ne) {plan1} total_ref {tot:.4f}")
        log("FULLCAND DONE")

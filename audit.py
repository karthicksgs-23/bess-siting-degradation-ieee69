"""Ex-post audit of a plan + dispatch: rainflow cycle ageing and exact AC (BFS) power flow."""
import numpy as np
import rainflow
from net69 import bfs
from model import phi, crf


def rainflow_damage(soc_day, E, p):
    """Fraction of cycle life consumed in one (cyclic) day. soc_day: 24 values (MWh)."""
    if E <= 0:
        return 0.0, []
    x = np.asarray(soc_day, float)
    k = int(np.argmax(x))
    seq = np.concatenate([x[k:], x[:k], [x[k]]])  # start/end at the maximum -> closed cycles
    dmg, depths = 0.0, []
    for rng, _mean, cnt, _i0, _i1 in rainflow.extract_cycles(seq):
        d = rng / E
        dmg += cnt * phi(d, p)
        depths.append((d, cnt))
    return dmg, depths


def audit(res, net, data, p):
    plan = res["plan"]
    cand = list(res["cand"])
    T = data["P"].shape[0]
    days = T // 24
    w = data["w"]
    # ---- costs of assets
    E = {b: ne * p["e_mod"] for b, (npw, ne) in plan.items()}
    Pw = {b: npw * p["p_mod"] for b, (npw, ne) in plan.items()}
    capex = sum(p["c_site"] + p["c_pow"] * Pw[b] + p["c_en"] * E[b] for b in plan)
    ann_capex = capex * (crf(p) + p["om"])
    cal = sum(p["c_cell"] * E[b] / p["L_cal"] for b in plan)
    # ---- rainflow cycle ageing
    cyc_cost, D_cyc, depth_log, fec = 0.0, {}, {}, {}
    for b in plan:
        i = cand.index(b)
        Db, dl, thr = 0.0, [], 0.0
        for d in range(days):
            sl = slice(d * 24, d * 24 + 24)
            dmg, depths = rainflow_damage(res["soc"][sl, i], E[b], p)
            wd = w[d * 24]
            Db += wd * dmg
            dl += [(dd, c * wd) for dd, c in depths]
            thr += wd * res["dis"][sl, i].sum()
        D_cyc[b] = Db
        depth_log[b] = dl
        fec[b] = thr / E[b] if E[b] > 0 else 0.0  # full equivalent cycles per year
        cyc_cost += p["c_cell"] * E[b] * Db
    # ---- exact AC power flow with battery injections
    inj = np.zeros_like(data["P"])
    for b in plan:
        i = cand.index(b)
        inj[:, b] = res["ch"][:, i] - res["dis"][:, i]
    V, lb, lt = bfs(net, data["P"] - data["PV"] + inj, data["Q"], v0=p["v0"])
    head = (data["P"] - data["PV"] + inj).sum(axis=1) + lt
    energy = float(np.sum(w * data["price"] * head))
    loss_mwh = float(np.sum(w * lt))
    vmin = V[:, 1:].min()
    n_viol = int((V[:, 1:].min(axis=1) < p["vmin"] - 1e-4).sum())
    life = {b: 1.0 / (1.0 / p["L_cal"] + D_cyc[b]) for b in plan}
    # simultaneous charge/discharge check
    simul = float(np.minimum(res["ch"], res["dis"]).max()) if len(plan) else 0.0
    total = ann_capex + cal + energy + cyc_cost
    return dict(ann_capex=ann_capex, cal=cal, energy=energy, cyc_ref=cyc_cost, total_ref=total,
                loss_mwh=loss_mwh, vmin=vmin, n_viol=n_viol, life=life, fec=fec, D_cyc=D_cyc,
                depth_log=depth_log, simul=simul, V=V, loss_t=lt,
                E_tot=sum(E.values()), P_tot=sum(Pw.values()))


def proxy_cycle_cost(res, data, p):
    """Cycle-ageing cost as the planning proxy itself would estimate it for the dispatch."""
    w = data["w"]
    m = res["model"]
    if m == "M0":
        return 0.0
    thr = float(np.sum(w[:, None] * res["dis"]))  # MWh/yr discharged
    if m == "M1":
        return p["c_cell"] * phi(p["d_ref"], p) / p["d_ref"] * thr
    return res.get("seg_cost", np.nan)

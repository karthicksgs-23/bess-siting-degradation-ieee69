"""Degradation-aware BESS siting and sizing MILP on the IEEE 69-bus feeder.

Network: LinDistFlow (squared-voltage) with a fixed per-bus/per-hour correction taken
from the exact BFS base case, piecewise-linear (tangent) outer approximation of
branch active-power losses.  Solver: HiGHS through scipy.optimize.milp.

Degradation formulations (planning proxies):
  M0  degradation-blind      : calendar ageing only
  M1  linear throughput      : cost per MWh discharged, calibrated at 80 % DoD
  M2  cycle-depth segments   : J SOC segments with convex marginal cost
                               c_j = C_cell*J*(Phi(j/J)-Phi((j-1)/J))   (Xu et al. 2018 idea)
Reference audit (not a planning model): rainflow counting of the SOC trajectory,
Phi(d) = d^kp / N100 per full cycle, plus calendar ageing.
"""
import time
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

from net69 import load_case, bfs, S_BASE_MVA

# ----------------------------------------------------------------- parameters
DEFAULT = dict(
    v0=1.02, vmin=0.95, vmax=1.05, vmin_plan=0.951,
    p_mod=0.1, e_mod=0.25,          # MW per power module, MWh per energy module
    np_max=20, ne_max=24, n_sites=3,
    e2p_min=1.0, e2p_max=4.0,       # hours
    c_pow=6.0e6, c_en=14.0e6, c_site=2.0e6,   # INR/MW, INR/MWh, INR/site
    c_cell=9.0e6,                    # INR/MWh cell replacement cost (part of c_en)
    om=0.02, rate=0.08, life=20,     # O&M share of capex, discount rate, project years
    L_cal=15.0,                      # calendar life (years to 80 % SoH)
    N100=4000.0, kp=1.8, d_ref=0.8,  # cycle life at 100 % DoD, Wohler exponent, M1 calibration DoD
    eta_c=0.95, eta_d=0.95,
    J=4,                             # SOC segments for M2
    n_tan=9,                         # loss tangents
    pen_v=5.0e7,                     # INR per p.u.^2-hour of undervoltage slack
)


def phi(d, p):
    return np.power(np.clip(d, 0, 1), p["kp"]) / p["N100"]


def crf(p):
    r, n = p["rate"], p["life"]
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


class VarIdx:
    def __init__(self):
        self.n = 0
        self.lb, self.ub, self.integ, self.c = [], [], [], []

    def add(self, shape, lb=0.0, ub=np.inf, integ=0, cost=0.0):
        size = int(np.prod(shape))
        idx = np.arange(self.n, self.n + size).reshape(shape)
        self.n += size
        self.lb.append(np.broadcast_to(lb, shape).ravel().astype(float))
        self.ub.append(np.broadcast_to(ub, shape).ravel().astype(float))
        self.integ.append(np.full(size, integ))
        self.c.append(np.broadcast_to(cost, shape).ravel().astype(float))
        return idx

    def arrays(self):
        return (np.concatenate(self.lb), np.concatenate(self.ub),
                np.concatenate(self.integ), np.concatenate(self.c))


class Rows:
    def __init__(self):
        self.r, self.c, self.v, self.lo, self.hi = [], [], [], [], []
        self.m = 0

    def add(self, cols_vals, lo, hi):
        """cols_vals: list of (col_index_array, coeff_array_or_scalar) with equal row count."""
        nrow = None
        for cols, vals in cols_vals:
            cols = np.atleast_1d(cols)
            nrow = len(cols) if nrow is None else nrow
        rows = np.arange(self.m, self.m + nrow)
        for cols, vals in cols_vals:
            cols = np.atleast_1d(cols)
            vals = np.broadcast_to(vals, cols.shape)
            self.r.append(rows); self.c.append(cols); self.v.append(vals.astype(float))
        self.lo.append(np.broadcast_to(lo, (nrow,)).astype(float))
        self.hi.append(np.broadcast_to(hi, (nrow,)).astype(float))
        self.m += nrow

    def matrix(self, nvar):
        A = sp.csr_matrix((np.concatenate(self.v), (np.concatenate(self.r), np.concatenate(self.c))),
                          shape=(self.m, nvar))
        return A, np.concatenate(self.lo), np.concatenate(self.hi)


def prepare(net, data, p):
    """Base-case quantities: LinDistFlow correction, flow ranges for tangents."""
    Pn = (data["P"] - data["PV"]) / S_BASE_MVA  # pu net demand, T x n
    Qn = data["Q"] / S_BASE_MVA
    A = net["A"]
    Pk0 = Pn @ A.T  # T x nb lossless branch flows
    Qk0 = Qn @ A.T
    # lossless LinDistFlow squared voltages
    drop = 2 * (Pk0 * net["r"] + Qk0 * net["x"])  # T x nb
    vlin = p["v0"] ** 2 - drop @ A  # T x n (A[k,j]=1 if branch k on path to j)
    Vb, _, lt = bfs(net, data["P"] - data["PV"], data["Q"], v0=p["v0"])
    corr = Vb ** 2 - vlin  # added to LinDistFlow estimate
    return dict(Pn=Pn, Qn=Qn, Qk0=Qk0, Pk0=Pk0, corr=corr, Vbase=Vb, loss_base=lt)


def solve(model, net, data, p, fixed=None, time_limit=1800, verbose=False, cand=None, relax=False):
    """model in {'M0','M1','M2'}; fixed = dict(bus->(np, ne)) to fix the plan (operation only)."""
    t0 = time.time()
    pre = prepare(net, data, p)
    T, n = data["P"].shape
    nb = len(net["f"])
    if cand is None:
        cand = np.arange(1, n)  # all non-root buses (0-indexed)
    C = len(cand)
    J = p["J"] if model == "M2" else 1
    w = data["w"]; price = data["price"]
    Sb = S_BASE_MVA
    cr = crf(p)

    vi = VarIdx()
    # ---- planning variables
    if fixed is None:
        y = vi.add((C,), 0, 1, 1, p["c_site"] * (cr + p["om"]))
        npw = vi.add((C,), 0, p["np_max"], 1, p["c_pow"] * p["p_mod"] * (cr + p["om"]))
        ne = vi.add((C,), 0, p["ne_max"], 1,
                    p["c_en"] * p["e_mod"] * (cr + p["om"]) + p["c_cell"] * p["e_mod"] / p["L_cal"])
    else:
        npf = np.array([fixed.get(int(b), (0, 0))[0] for b in cand], float)
        nef = np.array([fixed.get(int(b), (0, 0))[1] for b in cand], float)
        y = vi.add((C,), (npf > 0).astype(float), (npf > 0).astype(float), 0, 0.0)
        npw = vi.add((C,), npf, npf, 0, 0.0)
        ne = vi.add((C,), nef, nef, 0, 0.0)

    # ---- operating variables
    Pk = vi.add((T, nb), -np.inf, np.inf, 0, 0.0)
    v = vi.add((T, n - 1), 0.0, np.inf, 0, 0.0)          # squared voltage, buses 1..n-1
    s = vi.add((T, n - 1), 0.0, np.inf, 0, p["pen_v"] * w[:, None])  # undervoltage slack
    wl = vi.add((T, nb), 0.0, np.inf, 0, 0.0)              # P_k^2 aux
    ch = vi.add((T, C, J), 0.0, np.inf, 0, 0.0)            # pu
    dis = vi.add((T, C, J), 0.0, np.inf, 0, 0.0)
    e = vi.add((T, C, J), 0.0, np.inf, 0, 0.0)             # pu-h

    lb, ub, integ, c = vi.arrays()
    c = c.copy()
    # energy purchase at the head + losses: price * Sb * (P_head + sum r_k (wl + Q^2))
    head = np.where(net["f"] == 0)[0]
    for t in range(T):
        coef = w[t] * price[t] * Sb
        c[Pk[t, head]] += coef
        c[wl[t, :]] += coef * net["r"]
    const = float(np.sum(w * price * Sb * ((pre["Qk0"] ** 2) @ net["r"])))
    # degradation proxies (per pu-h discharged)
    cdeg = np.zeros_like(c)
    if model == "M1":
        c_thr = p["c_cell"] * phi(p["d_ref"], p) / p["d_ref"]  # INR per MWh discharged
        for t in range(T):
            cdeg[dis[t].ravel()] += w[t] * c_thr * Sb
    elif model == "M2":
        jj = np.arange(1, J + 1)
        cj = p["c_cell"] * J * (phi(jj / J, p) - phi((jj - 1) / J, p))
        for t in range(T):
            cdeg[dis[t]] += w[t] * Sb * cj[None, :]
    c = c + cdeg

    R = Rows()
    # (1) flow balance at each non-root bus j: P_parent(j) - sum P_children = Pn_j + ch_j - dis_j
    child_of = {j: [] for j in range(n)}
    for k in range(nb):
        child_of[net["f"][k]].append(k)
    cpos = {int(b): i for i, b in enumerate(cand)}
    for j in range(1, n):
        kpar = net["parent_br"][j]
        for t in range(T):
            cols = [(Pk[t, kpar], 1.0)]
            for kc in child_of[j]:
                cols.append((Pk[t, kc], -1.0))
            if j in cpos:
                i = cpos[j]
                cols.append((ch[t, i, :], -np.ones(J)))
                cols.append((dis[t, i, :], np.ones(J)))
            cc = np.concatenate([np.atleast_1d(a) for a, _ in cols])
            vv = np.concatenate([np.broadcast_to(b, np.atleast_1d(a).shape) for a, b in cols])
            _row_add(R, cc, vv, pre["Pn"][t, j], pre["Pn"][t, j])
    # (2) voltage drop: v_to - v_from + 2 r P_k = -2 x Q_k0
    for k in range(nb):
        fr, to = net["f"][k], net["t"][k]
        for t in range(T):
            rhs = -2 * net["x"][k] * pre["Qk0"][t, k]
            if fr == 0:
                _row_add(R, np.array([v[t, to - 1], Pk[t, k]]), np.array([1.0, 2 * net["r"][k]]),
                         rhs + p["v0"] ** 2, rhs + p["v0"] ** 2)
            else:
                _row_add(R, np.array([v[t, to - 1], v[t, fr - 1], Pk[t, k]]),
                         np.array([1.0, -1.0, 2 * net["r"][k]]), rhs, rhs)
    # (3) voltage limits with correction: vmin^2 <= v + corr + s ; v + corr <= vmax^2
    corr = pre["corr"][:, 1:]
    vv_idx = v.ravel(); s_idx = s.ravel()
    R.add([(vv_idx, 1.0), (s_idx, 1.0)], (p.get("vmin_plan", p["vmin"]) ** 2 - corr).ravel(), np.inf)
    R.add([(vv_idx, 1.0)], -np.inf, (p["vmax"] ** 2 - corr).ravel())
    # (4) loss tangents: wl >= 2 a P - a^2
    pmax = np.abs(pre["Pk0"]).max(axis=0) + 0.25  # pu
    for q in np.linspace(-1, 1, p["n_tan"]):
        a = q * pmax  # nb
        A_ = np.broadcast_to(a, (T, nb))
        R.add([(wl.ravel(), 1.0), (Pk.ravel(), (-2 * A_).ravel())], (-(A_ ** 2)).ravel(), np.inf)
    # (5) battery power and energy limits
    pm = p["p_mod"] / Sb; em = p["e_mod"] / Sb
    ch_sum_rows(R, ch, npw, pm, T, C, J)
    ch_sum_rows(R, dis, npw, pm, T, C, J)
    for j in range(J):
        R.add([(e[:, :, j].ravel(), 1.0), (np.tile(ne, T), -em / J)], -np.inf, 0.0)
    # (6) SOC dynamics, cyclic per representative day
    days = T // 24
    for d in range(days):
        for h in range(24):
            t = d * 24 + h
            tp = d * 24 + (h - 1) % 24
            R.add([(e[t].ravel(), 1.0), (e[tp].ravel(), -1.0),
                   (ch[t].ravel(), -p["eta_c"]), (dis[t].ravel(), 1.0 / p["eta_d"])], 0.0, 0.0)
    # (7) sizing logic
    if fixed is None:
        R.add([(npw, 1.0), (y, -p["np_max"])], -np.inf, 0)
        R.add([(ne, 1.0), (y, -p["ne_max"])], -np.inf, 0)
        R.add([(npw, 1.0), (y, -1.0)], 0, np.inf)  # site used -> at least one module
        R.add([(ne, p["e_mod"]), (npw, -p["e2p_min"] * p["p_mod"])], 0, np.inf)
        R.add([(ne, p["e_mod"]), (npw, -p["e2p_max"] * p["p_mod"])], -np.inf, 0)
        _row_add(R, y, np.ones(C), -np.inf, p["n_sites"])

    A, lo, hi = R.matrix(vi.n)
    t_build = time.time() - t0
    SCALE = 1e6  # solve in INR million for numerical conditioning
    if relax:
        from scipy.optimize import linprog
        eq = np.isclose(lo, hi)
        Ae, be = A[eq], hi[eq]
        hfin = (~eq) & np.isfinite(hi)
        lfin = (~eq) & np.isfinite(lo)
        Au = sp.vstack([A[hfin], -A[lfin]]).tocsr()
        bu = np.concatenate([hi[hfin], -lo[lfin]])
        res = linprog(c / SCALE, A_ub=Au, b_ub=bu, A_eq=Ae, b_eq=be,
                      bounds=np.column_stack([lb, ub]), method="highs-ipm",
                      options=dict(time_limit=time_limit, disp=verbose))
    else:
      res = milp(c=c / SCALE, constraints=LinearConstraint(A, lo, hi), integrality=integ,
               bounds=Bounds(lb, ub),
               options=dict(time_limit=time_limit, disp=verbose, mip_rel_gap=1e-4))
    if not hasattr(res, "mip_gap"):
        res.mip_gap = 0.0
    out = dict(model=model, status=res.status, message=res.message, obj=(res.fun * SCALE if res.fun is not None else np.nan) + const,
               t_build=t_build, t_solve=time.time() - t0 - t_build, nvar=vi.n, ncon=A.shape[0],
               nint=int(integ.sum()), mip_gap=getattr(res, "mip_gap", np.nan))
    if res.x is None:
        return out
    x = res.x
    out["proxy_cyc"] = float(cdeg @ x)
    out["plan"] = {int(cand[i]): (int(round(x[npw[i]])), int(round(x[ne[i]])))
                   for i in range(C) if x[npw[i]] > 0.5}
    out["relaxed_E"] = {int(cand[i]): float(x[ne[i]]) * p["e_mod"] for i in range(C)}
    out["ch"] = x[ch].sum(axis=2) * Sb  # T x C MW
    out["dis"] = x[dis].sum(axis=2) * Sb
    out["soc"] = x[e].sum(axis=2) * Sb  # MWh
    out["slack"] = x[s].max()
    out["cand"] = cand
    out["pre"] = pre
    return out


def _row_add(R, cols, vals, lo, hi):
    cols = np.atleast_1d(cols)
    rows = np.full(cols.shape, R.m)
    R.r.append(rows); R.c.append(cols); R.v.append(np.broadcast_to(vals, cols.shape).astype(float))
    R.lo.append(np.array([lo], float)); R.hi.append(np.array([hi], float))
    R.m += 1


def ch_sum_rows(R, var, npw, pm, T, C, J):
    """sum_j var[t,i,j] <= pm * np_i  for all t,i."""
    rows0 = R.m
    nrow = T * C
    rows = np.arange(rows0, rows0 + nrow)
    # var part
    vcols = var.reshape(T * C, J)
    R.r.append(np.repeat(rows, J)); R.c.append(vcols.ravel()); R.v.append(np.ones(nrow * J))
    R.r.append(rows); R.c.append(np.tile(npw, T)); R.v.append(np.full(nrow, -pm))
    R.lo.append(np.full(nrow, -np.inf)); R.hi.append(np.zeros(nrow))
    R.m += nrow

"""IEEE 69-bus radial feeder (Baran & Wu 1989, MATPOWER case69.m) and an
exact backward/forward-sweep (BFS) AC power flow used for ex-post checks."""
import re
from pathlib import Path
import numpy as np

DATA = Path(__file__).parent / "data" / "case69.m"
V_BASE_KV = 12.66
S_BASE_MVA = 10.0
Z_BASE = V_BASE_KV**2 / S_BASE_MVA  # ohm


def _block(txt, name):
    m = re.search(r"mpc\." + name + r"\s*=\s*\[(.*?)\];", txt, re.S)
    rows = []
    for line in m.group(1).splitlines():
        line = line.split("%")[0].strip().rstrip(";")
        if line:
            rows.append([float(v) for v in line.split()])
    return np.array(rows)


def load_case():
    txt = DATA.read_text()
    bus = _block(txt, "bus")
    br = _block(txt, "branch")
    n = bus.shape[0]
    net = {
        "n": n,
        "Pd": bus[:, 2] / 1e3,  # MW (file gives kW)
        "Qd": bus[:, 3] / 1e3,  # MVAr
        "f": br[:, 0].astype(int) - 1,
        "t": br[:, 1].astype(int) - 1,
        "r": br[:, 2] / Z_BASE,  # p.u.
        "x": br[:, 3] / Z_BASE,
    }
    nb = len(net["f"])
    # parent branch of every non-root bus, and children lists
    parent_br = -np.ones(n, dtype=int)
    for k in range(nb):
        parent_br[net["t"][k]] = k
    net["parent_br"] = parent_br
    # topological order (root first)
    children = {i: [] for i in range(n)}
    for k in range(nb):
        children[net["f"][k]].append(net["t"][k])
    order, stack = [], [0]
    while stack:
        i = stack.pop()
        order.append(i)
        stack.extend(children[i])
    net["order"] = np.array(order)
    # path matrix: A[k, j] = 1 if bus j is downstream of (or equal to) the to-bus of branch k
    A = np.zeros((nb, n))
    for j in range(1, n):
        b = j
        while b != 0:
            k = parent_br[b]
            A[k, j] = 1.0
            b = net["f"][k]
    net["A"] = A  # branch flow = A @ net injection demand
    # R/X path sums for LinDistFlow voltage sensitivities: V2_j = 1 - 2*sum_{k in path(j)} (r_k P_k + x_k Q_k)
    net["Ppath"] = A.T  # Ppath[j, k] = 1 if branch k on path root->j
    return net


def bfs(net, P_mw, Q_mvar, v0=1.0, tol=1e-10, itmax=200):
    """Backward/forward sweep. P_mw, Q_mvar: net demand per bus (MW, MVAr),
    shape (n,) or (T, n). Returns V (p.u. magnitude), branch losses (MW), total loss (MW)."""
    P = np.atleast_2d(P_mw) / S_BASE_MVA
    Q = np.atleast_2d(Q_mvar) / S_BASE_MVA
    T, n = P.shape
    f, t, r, x = net["f"], net["t"], net["r"], net["x"]
    z = r + 1j * x
    S = P + 1j * Q
    V = np.ones((T, n), dtype=complex) * v0
    order = net["order"]
    rev = order[::-1]
    for _ in range(itmax):
        I_load = np.conj(S / V)
        I_br = np.zeros((T, len(f)), dtype=complex)
        Ibus = I_load.copy()
        for j in rev:
            if j == 0:
                continue
            k = net["parent_br"][j]
            I_br[:, k] = Ibus[:, j]
            Ibus[:, f[k]] += Ibus[:, j]
        Vn = V.copy()
        for j in order:
            if j == 0:
                continue
            k = net["parent_br"][j]
            Vn[:, j] = Vn[:, f[k]] - z[k] * I_br[:, k]
        err = np.max(np.abs(Vn - V))
        V = Vn
        if err < tol:
            break
    loss_br = (np.abs(I_br) ** 2) * r * S_BASE_MVA
    return np.abs(V), loss_br, loss_br.sum(axis=1)


if __name__ == "__main__":
    net = load_case()
    V, lb, lt = bfs(net, net["Pd"], net["Qd"])
    print(f"Total load P = {net['Pd'].sum()*1e3:.1f} kW, Q = {net['Qd'].sum()*1e3:.1f} kVAr")
    print(f"Base-case loss = {lt[0]*1e3:.2f} kW (literature 224.9-225.0 kW)")
    j = V[0].argmin()
    print(f"Vmin = {V[0, j]:.4f} p.u. at bus {j+1} (literature 0.9092 @ 65)")

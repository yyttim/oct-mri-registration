"""Joint placement of all slabs and their mosaic tiles from the correspondences (docs/TILESTACK.md, Placement). The
in-plane and the depth residuals enter two robust least-squares problems with one design, the L1 penalties by reweighted
least squares. The mean translation of the slabs is held at zero, and the ridge settles the field they share.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from . import data

FIELD = int(round(2.56 / data.VOX))     # control spacing of the field, pixels
HUBER = round(0.005 / data.VOX, 6)      # robust threshold of the residuals, pixels and planes
EPS = round(0.001 / data.VOX, 6)        # smoothing of the L1 penalties
LAM_TILE = round(0.0006 / data.VOX, 6)  # L1 weights, 0.0006 mm on the tiles and 0.015 mm between neighbouring tiles: in
LAM_FUSE = round(0.015 / data.VOX, 6)   # pixels both scale as 1 / VOX against the squared residuals, at any voxel
LAM_FIELD = 0.1                         # ridge on the field's control values


def _b3(u):
    u2, u3 = u * u, u * u * u
    return np.stack([(1 - u) ** 3, 3 * u3 - 6 * u2 + 4, -3 * u3 + 3 * u2 + 3 * u + 1, u3], -1) / 6.0


class Spline:
    """Cubic B-spline on the slab field with control spacing G pixels (Rueckert et al. 1999)."""

    def __init__(self, G):
        self.G = G
        self.gy, self.gx = int(np.ceil(data.NY / G)) + 3, int(np.ceil(data.NX / G)) + 3
        self.n = self.gy * self.gx

    def weights(self, y, x):
        fy, fx = np.clip(y, 0, data.NY - 1e-3) / self.G, np.clip(x, 0, data.NX - 1e-3) / self.G
        jy, jx = np.floor(fy).astype(int), np.floor(fx).astype(int)
        by, bx = _b3(fy - jy), _b3(fx - jx)
        idx = ((jy[:, None] + np.arange(4))[:, :, None] * self.gx + (jx[:, None] + np.arange(4))[:, None, :]).reshape(-1, 16)
        return idx, (by[:, :, None] * bx[:, None, :]).reshape(-1, 16)


def _design(m, lay, slabs, spl, na):
    """The design shared by the three axes and the right-hand sides. Columns per slab: the translation (and the two
    linear terms without a field, na 3), the tiles, the field."""
    kof = {s: i for i, s in enumerate(slabs)}
    K, n = len(slabs), lay.n
    per = na + n + (spl.n if spl else 0)
    R, C, V = [], [], []
    off = 0
    for c in m:
        N = len(c["xB"])
        r = np.arange(off, off + N)
        for k, x, y, sg in ((kof[int(c["t"])], c["xB"], c["yB"], 1.0), (kof[int(c["s"])], c["xA"], c["yA"], -1.0)):
            cell = lay.cell(x, y, slabs[k])
            base = k * per
            R += [r, r]; C += [np.full(N, base), base + na + cell]; V += [np.full(N, sg), np.full(N, sg)]
            if na == 3:
                R += [r, r]; C += [np.full(N, base + 1), np.full(N, base + 2)]
                V += [sg * (x - data.XC) / 1000, sg * (y - data.YC) / 1000]
            if spl:
                idx, w = spl.weights(y, x)
                R.append(np.repeat(r, 16)); C.append((base + na + n + idx).ravel()); V.append(sg * w.ravel())
        off += N
    cat = np.concatenate
    A = sp.csr_matrix((cat(V), (cat(R), cat(C))), shape=(off, K * per))
    b = [cat([c[f"{a}A"] - c[f"{a}B"] for c in m]) for a in "xyz"]
    return A, b, per, na


def _mean_zero(groups, size, w=1e6):
    """w (sum of the columns of a group)^2 for each group of columns: holds the mean of each group at zero."""
    g = [np.asarray(c) for c in groups]
    r = np.concatenate([np.repeat(c, len(c)) for c in g]); c = np.concatenate([np.tile(c, len(c)) for c in g])
    return sp.csr_matrix((np.full(len(r), w), (r, c)), shape=(size, size))


def solve(m, lay, slabs, huber=HUBER, lam_tile=LAM_TILE, lam_fuse=LAM_FUSE, eps=EPS, G=FIELD, lam_field=LAM_FIELD,
          iters=60, tol=0.01):
    """m: correspondence sets (dicts with zA, yA, xA, zB, yB, xB, s, t), in pixels and planes. G: control spacing of the
    field in pixels, None for two linear terms instead. huber, eps and tol in pixels."""
    slabs = list(slabs)
    K, n = len(slabs), lay.n
    if G and lam_field <= 0:
        raise ValueError("the field needs its ridge, lam_field > 0")
    spl = Spline(G) if G else None
    A, (bx, by, bz), per, na = _design(m, lay, slabs, spl, 1 if spl else 3)
    tcol = np.concatenate([k * per + na + np.arange(n) for k in range(K)])
    reg = np.full(K * per, 1e-6)
    for k in range(K):
        reg[k * per + na + n: (k + 1) * per] = lam_field
    RG = _mean_zero([[k * per + j for k in range(K)] for j in range(na)], K * per)
    fr, fc, fv = [], [], []                             # neighbour differences of tile translations
    for k, s in enumerate(slabs):
        present = set(lay.exists[s])
        for t in sorted(present):
            for d in [lay.cols] + ([1] if (t + 1) % lay.cols else []):
                if t + d in present:
                    r = len(fr) // 2
                    fr += [r, r]; fc += [t + k * n, t + d + k * n]; fv += [1.0, -1.0]
    nf = len(fr) // 2
    Dt = sp.csr_matrix((fv, (fr, fc)), shape=(nf, K * n))
    D = (Dt @ sp.csr_matrix((np.ones(K * n), (np.arange(K * n), tcol)), shape=(K * n, K * per))).tocsr()
    use_f = lam_fuse > 0
    wx, wz = np.ones(A.shape[0]), np.ones(A.shape[0])
    lx = np.full(K * n, lam_tile / eps)
    lz = lx.copy()
    fx_w = np.full(nf, lam_fuse / eps); fz_w = fx_w.copy()
    prev = None
    for _ in range(iters):
        R = reg.copy(); R[tcol] += lx
        M = A.T @ sp.diags(wx) @ A + RG + sp.diags(R)
        if use_f:
            M = M + D.T @ sp.diags(fx_w) @ D
        lu = splu(M.tocsc())
        px = lu.solve(A.T @ (wx * bx)); py = lu.solve(A.T @ (wx * by))
        res = np.hypot(A @ px - bx, A @ py - by)
        wx = np.where(res <= huber, 1.0, huber / np.maximum(res, 1e-9))
        Rz = reg.copy(); Rz[tcol] += lz
        Mz = A.T @ sp.diags(wz) @ A + RG + sp.diags(Rz)
        if use_f:
            Mz = Mz + D.T @ sp.diags(fz_w) @ D
        pz = splu(Mz.tocsc()).solve(A.T @ (wz * bz))
        rz = np.abs(A @ pz - bz)
        wz = np.where(rz <= huber, 1.0, huber / np.maximum(rz, 1e-9))
        lx = lam_tile / (np.hypot(px[tcol], py[tcol]) + eps)
        fx_w = lam_fuse / (np.hypot(D @ px, D @ py) + eps)
        lz = lam_tile / (np.abs(pz[tcol]) + eps)
        fz_w = lam_fuse / (np.abs(D @ pz) + eps)
        cur = np.r_[px, py, pz]
        if prev is not None and np.abs(cur - prev).max() < tol:
            break
        prev = cur
    P = np.stack([px.reshape(K, per), py.reshape(K, per), pz.reshape(K, per)], 1)                 # (K, 3, per)
    aff = np.zeros((K, 3, 3))                                                                       # [a, b_x, b_y] per axis
    aff[:, :, :na] = P[:, :, :na]
    return dict(slabs=slabs, aff=aff, field=P[:, :, na + n:] if spl else None, G=G, tiles=np.transpose(P[:, :, na:na + n], (0, 2, 1)))


def displacement(sol, lay, s, x, y):
    """Displacement (u, v, w) of raw points of slab s, w the depth offset: pixels and planes."""
    k = sol["slabs"].index(s)
    x, y = np.asarray(x, float), np.asarray(y, float)
    a = sol["aff"][k]; e = sol["tiles"][k][lay.cell(x, y, s)]
    dx, dy = (x - data.XC) / 1000, (y - data.YC) / 1000
    d = [a[i, 0] + a[i, 1] * dx + a[i, 2] * dy + e[..., i] for i in range(3)]
    if sol["field"] is not None:
        idx, w = Spline(sol["G"]).weights(y.ravel(), x.ravel())
        d = [d[i] + (sol["field"][k, i][idx] * w).sum(1).reshape(x.shape) for i in range(3)]
    return d


def residuals(sol, lay, c):
    """Common-frame mismatch (B minus A) of a correspondence set: x, y in pixels, z in planes."""
    uB, vB, wB = displacement(sol, lay, int(c["t"]), c["xB"], c["yB"])
    uA, vA, wA = displacement(sol, lay, int(c["s"]), c["xA"], c["yA"])
    return c["xB"] + uB - c["xA"] - uA, c["yB"] + vB - c["yA"] - vA, c["zB"] + wB - c["zA"] - wA

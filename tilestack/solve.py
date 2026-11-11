"""Joint placement of all slabs and their mosaic tiles from the overlap correspondences.

Slab s maps its raw voxel (z, y, x) in mosaic tile k = cell(x, y) to the common frame by
    (X, Y, Z) = (x, y, z - beta C_s(z, y, x)) + a_s + f_s(x, y) + e_s[k]:
a translation a_s, a smooth deformation f_s (a cubic B-spline, which carries the slab's tilts) and a translation e_s[k]
per mosaic tile, each in-plane and in depth, and along depth the optical-path term beta C (optical.py). A correspondence
says that its two ends land on the same point. The in-plane and the depth residuals enter two robust (Huber)
least-squares problems with one design, and beta is one more unknown of the depth problem. The tile translations carry a
group-L1 penalty and a fused-L1 penalty between neighbouring tiles of a slab (tile errors come in rows and columns;
Tibshirani et al. 2005), by reweighted least squares, and the field a weak ridge. The common frame is the mean slab: the
mean translation over the slabs is held at zero, and the ridge settles the field all slabs share.
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

    def bending(self):
        idx = np.arange(self.n).reshape(self.gy, self.gx)
        rows, cols, vals, r = [], [], [], 0
        for a, b, c in ((idx[:-2], idx[1:-1], idx[2:]), (idx[:, :-2], idx[:, 1:-1], idx[:, 2:])):
            for i, j, k in zip(a.ravel(), b.ravel(), c.ravel()):
                rows += [r] * 3; cols += [i, j, k]; vals += [1.0, -2.0, 1.0]; r += 1
        s2 = np.sqrt(2.0)
        for i, j, k, l in zip(idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()):
            rows += [r] * 4; cols += [i, j, k, l]; vals += [s2, -s2, -s2, s2]; r += 1
        D = sp.csr_matrix((vals, (rows, cols)), shape=(r, self.n))
        return D.T @ D


def _design(m, lay, slabs, spl, na):
    """The design shared by the three axes, the right-hand sides, the optical column and the tiles each slab has
    matches in. Columns per slab: the translation (and the two linear terms without a field, na 3), the tiles, the field."""
    kof = {s: i for i, s in enumerate(slabs)}
    K, n = len(slabs), lay.n
    per = na + n + (spl.n if spl else 0)
    R, C, V = [], [], []
    has = np.zeros((K, n), bool)
    off = 0
    for c in m:
        N = len(c["xB"])
        r = np.arange(off, off + N)
        for k, x, y, sg in ((kof[int(c["t"])], c["xB"], c["yB"], 1.0), (kof[int(c["s"])], c["xA"], c["yA"], -1.0)):
            cell = lay.cell(x, y, slabs[k])
            has[k, np.unique(cell)] = True
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
    return A, b, cat([c["cA"] - c["cB"] for c in m]), per, na, has


def _mean_zero(groups, size, w=1e6):
    """w (sum of the columns of a group)^2 for each group of columns: holds the mean of each group at zero."""
    g = [np.asarray(c) for c in groups]
    r = np.concatenate([np.repeat(c, len(c)) for c in g]); c = np.concatenate([np.tile(c, len(c)) for c in g])
    return sp.csr_matrix((np.full(len(r), w), (r, c)), shape=(size, size))


def solve(m, lay, slabs, frame=None, huber=HUBER, penalty="l1", lam_tile=LAM_TILE, lam_fuse=LAM_FUSE, eps=EPS, G=FIELD,
          lam_bend=0.0, lam_field=LAM_FIELD, fuse="both", optical=True, tiles=True, iters=60, tol=0.01):
    """m: correspondence sets (dicts with zA, yA, xA, cA, zB, yB, xB, cB, s, t), in pixels and planes.
    frame: the slabs whose mean placement is the common frame (default all). penalty: "l1" (group-L1 lam_tile and
    fused-L1 lam_fuse between neighbouring tiles, reweighted with eps) or "l2" (ridge lam_tile) on the tile
    translations. fuse: "both" lattice directions or "columns". G: control spacing (pixels) of the smooth field, None
    for no field. lam_field: the ridge on the field; lam_bend: a bending penalty on it (none by default; the control
    spacing sets the smoothness). optical: beta solved with the rest (True), fixed at a given value, or none (False).
    tiles: False holds every tile translation at zero. huber, eps and tol in pixels."""
    slabs = list(slabs)
    K, n = len(slabs), lay.n
    if G and lam_field <= 0:
        raise ValueError("the field needs its ridge, lam_field > 0")
    spl = Spline(G) if G else None
    A, (bx, by, bz), dc, per, na, has = _design(m, lay, slabs, spl, 1 if spl else 3)
    free = optical is True
    nz = K * per + int(free)                              # depth unknowns: the slabs' and beta when it is solved
    Az = sp.hstack([A, sp.csr_matrix(dc[:, None])]).tocsr() if free else A
    if not free:
        bz = bz - float(optical) * dc
    fk = [slabs.index(s) for s in (slabs if frame is None else frame)]
    tcol = np.concatenate([k * per + na + np.arange(n) for k in range(K)])
    reg = np.full(K * per, 1e-6)
    for k in range(K):
        reg[k * per + na + n: (k + 1) * per] = lam_field
    Gm = _mean_zero([[k * per + j for k in fk] for j in range(na)], K * per)
    Rb = sp.csr_matrix((K * per, K * per))
    if spl and lam_bend:
        Bd = spl.bending()
        Rb = sp.block_diag([sp.csr_matrix((na + n, na + n)) if i % 2 == 0 else lam_bend * Bd for i in range(2 * K)]).tocsr()
    RG = (Rb + Gm).tocsr()
    RGz = sp.block_diag([RG, sp.csr_matrix((1, 1))]).tocsr() if free else RG
    fr, fc, fv = [], [], []                             # neighbour differences of tile translations
    for k, s in enumerate(slabs):
        present = set(lay.exists[s])
        for t in sorted(present):
            for d in [lay.cols] + ([1] if fuse == "both" and (t + 1) % lay.cols else []):
                if t + d in present:
                    r = len(fr) // 2
                    fr += [r, r]; fc += [t + k * n, t + d + k * n]; fv += [1.0, -1.0]
    nf = len(fr) // 2
    Dt = sp.csr_matrix((fv, (fr, fc)), shape=(nf, K * n))
    Dxy = (Dt @ sp.csr_matrix((np.ones(K * n), (np.arange(K * n), tcol)), shape=(K * n, K * per))).tocsr()
    Dz = (Dt @ sp.csr_matrix((np.ones(K * n), (np.arange(K * n), tcol)), shape=(K * n, nz))).tocsr()
    use_f = penalty == "l1" and lam_fuse > 0 and tiles
    l1 = penalty == "l1" and tiles
    wx, wz = np.ones(A.shape[0]), np.ones(A.shape[0])
    lx = np.full(K * n, 1e9 if not tiles else (lam_tile / eps if penalty == "l1" else lam_tile))
    lz = lx.copy()
    fx_w = np.full(nf, lam_fuse / eps); fz_w = fx_w.copy()
    prev = None
    for it in range(iters):
        R = reg.copy(); R[tcol] += lx
        M = A.T @ sp.diags(wx) @ A + RG + sp.diags(R)
        if use_f:
            M = M + Dxy.T @ sp.diags(fx_w) @ Dxy
        lu = splu(M.tocsc())
        px = lu.solve(A.T @ (wx * bx)); py = lu.solve(A.T @ (wx * by))
        res = np.hypot(A @ px - bx, A @ py - by)
        wx = np.where(res <= huber, 1.0, huber / np.maximum(res, 1e-9))
        Rz = np.r_[reg, np.zeros(int(free))]; Rz[tcol] += lz
        Mz = Az.T @ sp.diags(wz) @ Az + RGz + sp.diags(Rz)
        if use_f:
            Mz = Mz + Dz.T @ sp.diags(fz_w) @ Dz
        pz = splu(Mz.tocsc()).solve(Az.T @ (wz * bz))
        rz = np.abs(Az @ pz - bz)
        wz = np.where(rz <= huber, 1.0, huber / np.maximum(rz, 1e-9))
        if l1:
            lx = lam_tile / (np.hypot(px[tcol], py[tcol]) + eps)
            fx_w = lam_fuse / (np.hypot(Dxy @ px, Dxy @ py) + eps)
            lz = lam_tile / (np.abs(pz[tcol]) + eps)
            fz_w = lam_fuse / (np.abs(Dz @ pz) + eps)
        cur = np.r_[px, py, pz]
        if prev is not None and np.abs(cur - prev).max() < tol:
            break
        prev = cur
    P = np.stack([px.reshape(K, per), py.reshape(K, per), pz[:K * per].reshape(K, per)], 1)       # (K, 3, per)
    aff = np.zeros((K, 3, 3))                                                                       # [a, b_x, b_y] per axis
    aff[:, :, :na] = P[:, :, :na]
    return dict(slabs=slabs, aff=aff, field=P[:, :, na + n:] if spl else None, G=G, tiles=np.transpose(P[:, :, na:na + n], (0, 2, 1)),
                has=has, beta=float(pz[-1]) if free else float(optical), iterations=it + 1,
                settings=dict(huber=huber, penalty=penalty, lam_tile=lam_tile, lam_fuse=lam_fuse, eps=eps, G=G,
                              lam_bend=lam_bend, lam_field=lam_field, fuse=fuse, optical=optical, tiles=tiles))


def displacement(sol, lay, s, x, y):
    """Displacement (u, v, w) of raw points of slab s, w the depth offset without the optical term: pixels and planes."""
    k = sol["slabs"].index(s)
    x, y = np.asarray(x, float), np.asarray(y, float)
    a = sol["aff"][k]; e = sol["tiles"][k][lay.cell(x, y, s)]
    dx, dy = (x - data.XC) / 1000, (y - data.YC) / 1000
    d = [a[i, 0] + a[i, 1] * dx + a[i, 2] * dy + e[..., i] for i in range(3)]
    if sol["field"] is not None:
        idx, w = Spline(sol["G"]).weights(y.ravel(), x.ravel())
        d = [d[i] + (sol["field"][k, i][idx] * w).sum(1).reshape(x.shape) for i in range(3)]
    return d


def inplane(sol, lay, s, x, y):
    """In-plane displacement (u, v) of raw points of slab s, pixels."""
    return displacement(sol, lay, s, x, y)[:2]


def depth(sol, lay, s, x, y, c=0.0):
    """Common-frame depth offset (planes) of raw points of slab s with optical covariate c."""
    return displacement(sol, lay, s, x, y)[2] - sol["beta"] * np.asarray(c)


def residuals(sol, lay, c):
    """Common-frame mismatch (B minus A) of a correspondence set: x, y in pixels, z in planes."""
    uB, vB, wB = displacement(sol, lay, int(c["t"]), c["xB"], c["yB"])
    uA, vA, wA = displacement(sol, lay, int(c["s"]), c["xA"], c["yA"])
    return (c["xB"] + uB - c["xA"] - uA, c["yB"] + vB - c["yA"] - vA,
            c["zB"] - sol["beta"] * c["cB"] + wB - c["zA"] + sol["beta"] * c["cA"] - wA)

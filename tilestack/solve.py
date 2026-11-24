"""Joint placement of all slabs and their mosaic tiles from the overlap correspondences (docs/TILESTACK.md, Placement):
    (X, Y, Z) = (x, y, z) + a_s + f_s(x, y) + e_s[k], with k = cell(x, y) the mosaic tile of the raw voxel (z, y, x).
The in-plane and the depth residuals enter two robust (Huber) least-squares problems with one design. The L1 penalties
on the tile translations are handled by reweighted least squares. The mean translation over the slabs is held at zero,
and the ridge settles the field all slabs share.
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
    """The four cubic B-spline weights at the fractions u, numpy arrays or torch tensors."""
    u2, u3 = u * u, u * u * u
    return (1 - u) ** 3 / 6, (3 * u3 - 6 * u2 + 4) / 6, (-3 * u3 + 3 * u2 + 3 * u + 1) / 6, u3 / 6


class Spline:
    """Cubic B-spline on the slab field with control spacing G pixels (Rueckert et al. 1999)."""

    def __init__(self, G):
        self.G = G
        self.gy, self.gx = int(np.ceil(data.NY / G)) + 3, int(np.ceil(data.NX / G)) + 3
        self.n = self.gy * self.gx

    def weights(self, y, x):
        fy, fx = np.clip(y, 0, data.NY - 1e-3) / self.G, np.clip(x, 0, data.NX - 1e-3) / self.G
        jy, jx = np.floor(fy).astype(int), np.floor(fx).astype(int)
        by, bx = np.stack(_b3(fy - jy), -1), np.stack(_b3(fx - jx), -1)
        idx = ((jy[:, None] + np.arange(4))[:, :, None] * self.gx + (jx[:, None] + np.arange(4))[:, None, :]).reshape(-1, 16)
        return idx, (by[:, :, None] * bx[:, None, :]).reshape(-1, 16)


def _design(m, lay, slabs, spl):
    """The design shared by the three axes and the right-hand sides. Columns per slab: the translation, the tiles, the
    field."""
    kof = {s: i for i, s in enumerate(slabs)}
    K, n = len(slabs), lay.n
    per = 1 + n + spl.n
    R, C, V = [], [], []
    off = 0
    for c in m:
        N = len(c["xB"])
        r = np.arange(off, off + N)
        for k, x, y, sg in ((kof[int(c["t"])], c["xB"], c["yB"], 1.0), (kof[int(c["s"])], c["xA"], c["yA"], -1.0)):
            cell = lay.cell(x, y, slabs[k])
            base = k * per
            R += [r, r]; C += [np.full(N, base), base + 1 + cell]; V += [np.full(N, sg), np.full(N, sg)]
            idx, w = spl.weights(y, x)
            R.append(np.repeat(r, 16)); C.append((base + 1 + n + idx).ravel()); V.append(sg * w.ravel())
        off += N
    cat = np.concatenate
    A = sp.csr_matrix((cat(V), (cat(R), cat(C))), shape=(off, K * per))
    b = [cat([c[f"{a}A"] - c[f"{a}B"] for c in m]) for a in "xyz"]
    return A, b, per


def solve(m, lay, slabs):
    """m: correspondence sets (dicts with zA, yA, xA, zB, yB, xB, s, t), in pixels and planes. -> the placement in pixels
    and planes, (x, y, depth): shift (slab, 3), tiles (slab, tile, 3) and the field's control values (slab, 3, control)."""
    K, n = len(slabs), lay.n
    spl = Spline(FIELD)
    A, (bx, by, bz), per = _design(m, lay, slabs, spl)
    tcol = np.concatenate([k * per + 1 + np.arange(n) for k in range(K)])
    reg = np.full(K * per, 1e-6)
    for k in range(K):
        reg[k * per + 1 + n: (k + 1) * per] = LAM_FIELD
    acol = np.arange(K) * per                           # 1e6 (sum of the translations)^2 holds their mean at zero
    Gm = sp.csr_matrix((np.full(K * K, 1e6), (np.repeat(acol, K), np.tile(acol, K))), shape=(K * per, K * per))
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
    wx, wz = np.ones(A.shape[0]), np.ones(A.shape[0])
    lx = np.full(K * n, LAM_TILE / EPS)
    lz = lx.copy()
    fx_w = np.full(nf, LAM_FUSE / EPS); fz_w = fx_w.copy()
    prev = None
    for it in range(60):
        R = reg.copy(); R[tcol] += lx
        M = A.T @ sp.diags(wx) @ A + Gm + sp.diags(R) + D.T @ sp.diags(fx_w) @ D
        lu = splu(M.tocsc())
        px = lu.solve(A.T @ (wx * bx)); py = lu.solve(A.T @ (wx * by))
        res = np.hypot(A @ px - bx, A @ py - by)
        wx = np.where(res <= HUBER, 1.0, HUBER / np.maximum(res, 1e-9))
        Rz = reg.copy(); Rz[tcol] += lz
        Mz = A.T @ sp.diags(wz) @ A + Gm + sp.diags(Rz) + D.T @ sp.diags(fz_w) @ D
        pz = splu(Mz.tocsc()).solve(A.T @ (wz * bz))
        rz = np.abs(A @ pz - bz)
        wz = np.where(rz <= HUBER, 1.0, HUBER / np.maximum(rz, 1e-9))
        lx = LAM_TILE / (np.hypot(px[tcol], py[tcol]) + EPS)
        fx_w = LAM_FUSE / (np.hypot(D @ px, D @ py) + EPS)
        lz = LAM_TILE / (np.abs(pz[tcol]) + EPS)
        fz_w = LAM_FUSE / (np.abs(D @ pz) + EPS)
        cur = np.r_[px, py, pz]
        if prev is not None and np.abs(cur - prev).max() < 0.01:        # pixels and planes
            break
        prev = cur
    P = np.stack([px.reshape(K, per), py.reshape(K, per), pz.reshape(K, per)], 1)                 # (K, 3, per)
    return dict(slabs=slabs, shift=P[:, :, 0], field=P[:, :, 1 + n:], G=FIELD, tiles=np.transpose(P[:, :, 1:1 + n], (0, 2, 1)),
                iterations=it + 1,
                settings=dict(huber=HUBER, lam_tile=LAM_TILE, lam_fuse=LAM_FUSE, eps=EPS, G=FIELD, lam_field=LAM_FIELD))


def displacement(sol, lay, s, x, y):
    """Displacement (u, v, w) of raw points of slab s, w the depth offset: pixels and planes."""
    k = sol["slabs"].index(s)
    x, y = np.asarray(x, float), np.asarray(y, float)
    e = sol["tiles"][k][lay.cell(x, y, s)]
    idx, w = Spline(sol["G"]).weights(y.ravel(), x.ravel())
    return [sol["shift"][k, i] + e[..., i] + (sol["field"][k, i][idx] * w).sum(1).reshape(x.shape) for i in range(3)]


def residuals(sol, lay, c):
    """Common-frame mismatch (B minus A) of a correspondence set: x, y in pixels, z in planes."""
    uB, vB, wB = displacement(sol, lay, int(c["t"]), c["xB"], c["yB"])
    uA, vA, wA = displacement(sol, lay, int(c["s"]), c["xA"], c["yA"])
    return c["xB"] + uB - c["xA"] - uA, c["yB"] + vB - c["yA"] - vA, c["zB"] + wB - c["zA"] - wA

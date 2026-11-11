"""Correspondences between neighbouring slabs inside their depth overlap (batched GPU phase correlation).

Slab s (A, older) and slab t (B, newer) image the same tissue: B's raw plane q shows what A shows near plane q + r(x, y).
A block of B (3 planes of log mu_s at a fixed depth below B's cut) is compared with candidate 3-plane stacks of A within
+-SEARCH planes of q + r by phase correlation (tiles can sit a few planes off the pair's plane). The in-plane peak is
refined on an upsampled DFT (Guizar-Sicairos et al. 2008), the plane on stacks of A interpolated at quarter planes
around the best whole plane, with a Gaussian through the highest of the seven and its two neighbours. A match says that
B's block centre (zB, yB, xB) and A's point (zA, yA, xA) are the same tissue.

Two passes per pair: a sparse coarse pass (large blocks, wide plane search) finds the depth offset plane r(x, y) and
the in-plane translation; the dense pass (small blocks) is pre-shifted by them. A pair is dropped (NoOverlap) when fewer
than MIN_BLOCKS coarse blocks agree on a plane offset within AGREE, or the plane lies outside the older slab's window.
"""
import os

import numpy as np
import torch

from . import data

dev = "cuda"
BLOCK = int(round(1.28 / data.VOX))                   # dense pass: block side and lattice (half the block)
COARSE = int(round(2.56 / data.VOX))                  # coarse pass: block side, lattice three quarters of it
DEPTHS = tuple(int(round(d / data.VOX)) for d in data.MATCH_MM)     # B's depths under its cut
SEARCH = int(round(0.10 / data.VOX))                  # planes either side of the predicted plane
PEAK = 0.12                                           # smallest correlation peak kept
MIN_BLOCKS = 8                                        # coarse blocks that must agree on the pair's plane offset
AGREE = 0.04 / data.VOX                               # planes within which they agree


class NoOverlap(Exception):
    """The two slabs share no tissue the matcher can use."""


def _parabola(a, b, c):
    den = a - 2.0 * b + c
    safe = torch.where(den < 0, den, torch.full_like(den, -1.0))
    return torch.where(den < 0, 0.5 * (a - c) / safe, torch.zeros_like(den))


def _gauss(a, c0, c):
    """Sub-sample peak offset of three positive samples: a parabola through their logarithms."""
    return _parabola(*(torch.log(torch.clamp(v, min=1e-6)) for v in (a, c0, c)))


def _stacks(L, z0, y0, x0, b):
    """L[z0:z0+3, y0:y0+b, x0:x0+b] for every item (z0 may have a trailing candidate axis)."""
    k3 = torch.arange(3, device=dev); kb = torch.arange(b, device=dev)
    zi = z0[..., None] + k3                                   # (..., 3)
    yi = y0[:, None] + kb; xi = x0[:, None] + kb              # (N, b)
    extra = zi.dim() - 2
    yi = yi.view(yi.shape[0], *([1] * extra), 1, b, 1); xi = xi.view(xi.shape[0], *([1] * extra), 1, 1, b)
    return L[zi[..., None, None], yi, xi]


def _upsampled(Rs, iy, ix, b, up, half):
    """Correlation on a fine grid (step 1/up, +-half pixels) around integer peaks (iy, ix), computed directly from the
    half spectra Rs (n, b, b // 2 + 1) by matrix-multiply DFT (Guizar-Sicairos 2008): (n, m, m) and the offsets t."""
    t = torch.arange(-half, half + 1e-6, 1.0 / up, device=dev)
    ky = torch.fft.fftfreq(b, device=dev) * b
    kx = torch.arange(b // 2 + 1, device=dev).float()
    wx = torch.where((kx == 0) | (kx == b // 2), 1.0, 2.0)
    Ey = torch.exp(2j * np.pi * (iy[:, None, None] + t[None, :, None]) * ky[None, None, :] / b)
    Ex = wx[None, :, None] * torch.exp(2j * np.pi * kx[None, :, None] * (ix[:, None, None] + t[None, None, :]) / b)
    return (Ey @ Rs @ Ex).real / (b * b * 3), t


def correlate(LA, LB, zB, yB, xB, cand, yA, xA, b, chunk=512):
    """Phase correlation of B stacks (first planes zB, corners yB, xB) with A stacks (first planes cand (N, P), -1 =
    invalid, corners yA, xA). Returns, per item, the fractional best candidate index, the in-plane peak (dy, dx)
    relative to the A corner offset, the peak height and whether the best candidate is interior.
    The in-plane peak is refined on an upsampled DFT (no pull towards whole pixels), the candidate index by a Gaussian
    through the neighbouring candidates' peak heights at that position."""
    win = torch.hann_window(b, device=dev)
    win = win[:, None] * win[None, :]
    out = []
    for i0 in range(0, len(zB), chunk):
        sl = slice(i0, i0 + chunk)
        c = cand[sl]; ok = c >= 0
        B = _stacks(LB, zB[sl], yB[sl], xB[sl], b)                          # (n, 3, b, b)
        A = _stacks(LA, torch.where(ok, c, torch.zeros_like(c)), yA[sl], xA[sl], b)   # (n, P, 3, b, b)
        B = (B - B.mean((1, 2, 3), keepdim=True)) / (B.std((1, 2, 3), keepdim=True) + 1e-6)
        A = (A - A.mean((2, 3, 4), keepdim=True)) / (A.std((2, 3, 4), keepdim=True) + 1e-6)
        FB = torch.fft.rfft2(B * win).conj()
        R = torch.fft.rfft2(A * win) * FB[:, None]
        Rs = (R / (R.abs() + 1e-6)).sum(2)                                   # (n, P, b, b // 2 + 1)
        del A, B, R
        cc = torch.fft.irfft2(Rs, s=(b, b)) / 3                              # (n, P, b, b)
        n, P = c.shape
        pk, idx = cc.flatten(2).max(2)
        pk = torch.where(ok, pk, torch.full_like(pk, -1.0))
        best = pk.argmax(1)
        ar = torch.arange(n, device=dev)
        iyl, ixl = idx[ar, best] // b, idx[ar, best] % b
        iy, ix = iyl.float(), ixl.float()
        C, t = _upsampled(Rs[ar, best], iy, ix, b, 20, 1.2)
        m = len(t)
        k = C.flatten(1).argmax(1); ky, kx = k // m, k % m
        kyc, kxc = ky.clamp(1, m - 2), kx.clamp(1, m - 2)
        fy = iy + t[ky] + torch.where(ky == kyc, _parabola(C[ar, kyc - 1, kx], C[ar, kyc, kx], C[ar, kyc + 1, kx]), torch.zeros_like(iy)) / 20
        fx = ix + t[kx] + torch.where(kx == kxc, _parabola(C[ar, ky, kxc - 1], C[ar, ky, kxc], C[ar, ky, kxc + 1]), torch.zeros_like(ix)) / 20
        fy = torch.where(fy > b / 2, fy - b, fy); fx = torch.where(fx > b / 2, fx - b, fx)
        lo = torch.clamp(best - 1, min=0); hi = torch.clamp(best + 1, max=P - 1)
        inner = (best > 0) & (best < P - 1) & ok[ar, lo] & ok[ar, hi]
        # sub-plane: neighbouring planes are nearly uncorrelated, so the correlation over whole planes is a narrow
        # triangle that no three-point fit follows; over A stacks interpolated linearly it peaks at the true fraction
        offs = torch.arange(-3, 4, device=dev).float() / 4
        z0f = c[ar, best].float()[:, None] + offs[None]
        zf = torch.clamp(z0f.floor().long(), 0, LA.shape[0] - 4)
        a = (z0f - zf.float()).clamp(0, 1)[..., None, None, None]
        Af = (1 - a) * _stacks(LA, zf, yA[sl], xA[sl], b) + a * _stacks(LA, zf + 1, yA[sl], xA[sl], b)
        Af = (Af - Af.mean((2, 3, 4), keepdim=True)) / (Af.std((2, 3, 4), keepdim=True) + 1e-6)
        Rf = torch.fft.rfft2(Af * win) * FB[:, None]
        Rsf = (Rf / (Rf.abs() + 1e-6)).sum(2)
        del Af, Rf
        h = torch.stack([_upsampled(Rsf[:, j], iy, ix, b, 10, 0.6)[0].flatten(1).max(1).values for j in range(7)], 1)
        jb = h.argmax(1); jc = jb.clamp(1, 5)
        dq = offs[jb] + torch.where(jb == jc, _gauss(h[ar, jc - 1], h[ar, jc], h[ar, jc + 1]), torch.zeros_like(fy)) / 4
        fp = best + torch.where(inner, dq, torch.zeros_like(fy))
        del Rsf
        out.append(torch.stack([fp, fy, fx, pk[ar, best], inner.float()], 1).cpu())
        del Rs, cc, C
    return torch.cat(out).numpy() if out else np.zeros((0, 5))


def _block_grid(vA, vB, b, step, pre=(0, 0)):
    ny, nx = vB.shape
    py, px = pre
    ys, xs = [], []
    for y0 in range(0, ny - b + 1, step):
        for x0 in range(0, nx - b + 1, step):
            ya, xa = y0 + py, x0 + px
            if ya < 0 or xa < 0 or ya + b > ny or xa + b > nx:
                continue
            if vB[y0:y0 + b, x0:x0 + b].mean() > 0.98 and vA[ya:ya + b, xa:xa + b].mean() > 0.98:
                ys.append(y0); xs.append(x0)
    return np.array(ys, int), np.array(xs, int)


def _robust_plane(x, y, v):
    """Plane v = c0 + c1 (x - xc)/1000 + c2 (y - yc)/1000, refitted three times without outliers."""
    A = np.c_[np.ones_like(x), (x - data.XC) / 1000, (y - data.YC) / 1000]
    k = np.ones(len(v), bool)
    for _ in range(4):
        c = np.linalg.lstsq(A[k], v[k], rcond=None)[0]
        r = np.abs(v - A @ c)
        k = r < 3 * np.median(r[k]) + 0.006 / data.VOX           # 6 um
    return c


def good(m, peak=PEAK):
    """Matches to keep: a clear correlation peak, not at the edge of the plane search."""
    return (m["peak"] > peak) & (m["interior"] > 0)


def keep(c, g):
    """A correspondence set reduced to the matches g (its scalar entries kept)."""
    return {q: (v[g] if np.ndim(v) and len(v) == len(g) else v) for q, v in c.items()}


def match_pair(s, t, LA=None, LB=None, out=None):
    """Dense correspondences between slab s (A) and slab t > s (B). LA, LB: log mu_s on the GPU (loaded if None).
    Raises NoOverlap when the two slabs share no tissue."""
    if out and os.path.exists(out):
        return dict(np.load(out))
    LA = torch.from_numpy(data.load_log(s)).to(dev) if LA is None else LA
    LB = torch.from_numpy(data.load_log(t)).to(dev) if LB is None else LB
    nz = LA.shape[0]
    sA, vA = data.geometry(s); sB, vB = data.geometry(t)
    # coarse pass: B at its first two match depths (shallow, so that the deep overlap after an over-cut is still
    # reached), every deeper plane of A
    bc = COARSE
    y1, x1 = _block_grid(vA, vB, bc, bc * 3 // 4)
    ys, xs = np.r_[y1, y1], np.r_[x1, x1]
    q = np.array([int(round(np.median(sB[y:y + bc, x:x + bc]))) for y, x in zip(ys, xs)]) + np.repeat(DEPTHS[:2], len(y1))
    keep_ = q + 2 <= data.DEEP
    ys, xs, q = ys[keep_], xs[keep_], q[keep_]
    P = nz - 2
    cand = np.tile(np.arange(P), (len(q), 1))
    lowA = np.array([np.median(sA[y:y + bc, x:x + bc]) for y, x in zip(ys, xs)]) + data.MARGIN
    cand = np.where((cand >= q[:, None] - 2 * DEPTHS[0]) & (cand >= lowA[:, None]) & (cand + 2 <= data.DEEP_A), cand, -1)
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)  # noqa: E731
    r = correlate(LA, LB, T(q), T(ys), T(xs), T(cand), T(ys), T(xs), bc, chunk=32)
    g = good(dict(peak=r[:, 3], interior=r[:, 4]))
    if g.sum() < MIN_BLOCKS:
        raise NoOverlap(f"pair {s}-{t}: only {g.sum()} coarse matches")
    xc, yc = xs + bc / 2, ys + bc / 2
    zr = r[:, 0] - q                                                         # raw plane offset zA - zB
    rplane = _robust_plane(xc[g], yc[g], zr[g])
    agree = int((np.abs(zr[g] - rplane[0] - rplane[1] * (xc[g] - data.XC) / 1000 - rplane[2] * (yc[g] - data.YC) / 1000) < AGREE).sum())
    if agree < MIN_BLOCKS or not 0 < rplane[0] < data.DEEP_A:
        raise NoOverlap(f"pair {s}-{t}: {agree} coarse matches agree on the plane offset {np.round(rplane, 2)}")
    pre = (int(round(np.median(r[g, 1]))), int(round(np.median(r[g, 2]))))
    # dense pass
    b = BLOCK
    ys, xs = _block_grid(vA, vB, b, b // 2, pre)
    rows = []
    for d in DEPTHS:
        q = np.array([int(round(np.median(sB[y:y + b, x:x + b]))) + d for y, x in zip(ys, xs)])
        xc, yc = xs + b / 2, ys + b / 2
        r0 = rplane[0] + rplane[1] * (xc - data.XC) / 1000 + rplane[2] * (yc - data.YC) / 1000
        base = np.rint(q + r0).astype(int) - SEARCH
        cand = base[:, None] + np.arange(2 * SEARCH + 1)[None]
        lowA = np.array([np.median(sA[y + pre[0]:y + pre[0] + b, x + pre[1]:x + pre[1] + b]) for y, x in zip(ys, xs)])
        ok = (cand >= lowA[:, None] + data.MARGIN) & (cand + 2 <= data.DEEP_A)
        cand = np.where(ok, cand, -1)
        use = (q + 2 <= data.DEEP) & (ok.sum(1) >= 3)
        if not use.any():
            continue
        u = np.nonzero(use)[0]
        rr = correlate(LA, LB, T(q[u]), T(ys[u]), T(xs[u]), T(cand[u]), T(ys[u] + pre[0]), T(xs[u] + pre[1]), b)
        zA = base[u] + rr[:, 0] + 1
        rows.append(np.c_[q[u] + 1, yc[u], xc[u], zA, yc[u] + pre[0] + rr[:, 1], xc[u] + pre[1] + rr[:, 2],
                          rr[:, 3], np.full(len(u), d), rr[:, 4]])
    a = np.concatenate(rows) if rows else np.zeros((0, 9))
    res = dict(zB=a[:, 0], yB=a[:, 1], xB=a[:, 2], zA=a[:, 3], yA=a[:, 4], xA=a[:, 5], peak=a[:, 6], d=a[:, 7],
               interior=a[:, 8], s=np.array(s), t=np.array(t), rplane=rplane, pre=np.array(pre), block=np.array(b))
    if out:
        np.savez_compressed(out, **res)
    return res

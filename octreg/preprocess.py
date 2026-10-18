"""Masks (docs/METHOD.md §1): the MRI foreground and the OCT specimen mask from isotropic texture. numpy/scipy on the CPU,
arrays [D, H, W] on isotropic grids of voxel_mm (mm). Exact zeros in the OCT are missing data (black tiles): never specimen."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from skimage.filters import threshold_otsu

from .params import Params

_CHUNK = 64          # planes per chunk along axis 0: memory only, results do not depend on it


def _fill_planes(m):
    """Holes filled in every array plane (union over the three axes): background enclosed by specimen within a plane is
    specimen, also where the hole reaches a cut face of the block and so is not enclosed in 3-D."""
    out = m.copy()
    for ax in range(3):
        s = np.zeros((3, 3, 3), bool)
        s[tuple(1 if d == ax else slice(None) for d in range(3))] = ndimage.generate_binary_structure(2, 1)
        out |= ndimage.binary_fill_holes(m, structure=s)
    return out


def _components(m, min_fraction):
    """The 6-connected components of m holding >= min_fraction of its voxels."""
    lab, _ = ndimage.label(m)
    size = np.bincount(lab.ravel())
    keep = size >= min_fraction * size[1:].sum()
    keep[0] = False
    return keep[lab]


def _pool(a, k):
    """Sums over k^3 blocks of a [D, H, W] zero-padded to multiples of k (block j covers voxels j k .. j k + k - 1)."""
    a = np.pad(a, [(0, -s % k) for s in a.shape])
    D, H, W = (s // k for s in a.shape)
    return a.reshape(D, k, H, k, W, k).sum(axis=(1, 3, 5))


def _upsample(a, k, shape, rows=None):
    """Linear interpolation of a block grid a (block centres at j k + (k - 1) / 2 voxels, clamped beyond the outer centres) at
    the voxels of a grid of `shape`. rows = (lo, hi) returns only rows lo:hi along axis 0. -> float32."""
    x = np.asarray(a, np.float32)
    for ax, n in enumerate(shape):
        c = np.clip((np.arange(n) - (k - 1) / 2) / k, 0, a.shape[ax] - 1)[slice(*rows) if ax == 0 and rows else slice(None)]
        i = np.floor(c).astype(int)
        f = (c - i).astype(np.float32).reshape([-1 if d == ax else 1 for d in range(3)])
        x = np.take(x, i, axis=ax) * (1 - f) + np.take(x, np.minimum(i + 1, a.shape[ax] - 1), axis=ax) * f
    return x


# ----------------------------------------------------------------------------- MRI foreground (§1)
def foreground(arr, voxel_mm, params: Params = Params()):
    """The MRI foreground: the voxels above the first histogram valley, walking down from the brightest peak, that holds less
    than valley_ratio of the smaller of its two peaks and whose darker class lies mostly outside the brighter side: less than
    half of the voxels between the next lower valley (or zero) and this one lie inside the convex hull of the candidate
    foreground (medium lies outside the specimen, while a darker tissue class such as white matter lies within the hull of
    cortex). Without such a valley, every positive voxel. Histogram of 256 bins of the positive values, peaks of the square
    root of the counts (so a narrow background peak cannot hide the tissue peak) with a prominence >= 5 % of the maximum and
    >= 5 bins apart. Holes filled, components >= min_component of the mask kept.
    arr: [D, H, W], voxel_mm: spacing. -> (bool [D, H, W], {threshold (arr units), volume_cm3})."""
    v = np.asarray(arr, np.float32)
    s = v[np.isfinite(v) & (v > 0)]
    if s.size == 0:
        raise ValueError("MRI foreground: the image has no positive values")
    h, e = np.histogram(s, bins=256)
    c = 0.5 * (e[1:] + e[:-1])
    pk = find_peaks(np.sqrt(h), prominence=0.05 * np.sqrt(h.max()), distance=5)[0]
    vals = [lo + int(np.argmin(h[lo:hi + 1])) for lo, hi in zip(pk[:-1], pk[1:])]          # between neighbouring peaks
    for i in range(len(vals) - 1, -1, -1):
        val = vals[i]
        if h[val] < params.valley_ratio * min(h[pk[i]], h[pk[i + 1]]):
            m = _components(ndimage.binary_fill_holes(v > c[val]), params.min_component)
            if _enclosed(m, (v > (c[vals[i - 1]] if i else 0)) & (v <= c[val])) < 0.5:
                t = float(c[val])
                break
    else:
        t, m = 0.0, _components(ndimage.binary_fill_holes(v > 0), params.min_component)
    return m, {"threshold": t, "volume_cm3": float(m.sum()) * voxel_mm ** 3 / 1e3}


def _enclosed(m, sel):
    """The share of the voxels of sel (every second voxel along each axis) that lie inside the convex hull of the mask m.
    1.0 when m has fewer than four surface voxels or they span a flat hull, 0.0 when sel has no sampled voxel."""
    from scipy.spatial import ConvexHull, Delaunay, QhullError
    pts = np.argwhere(m & ~ndimage.binary_erosion(m))
    q = np.argwhere(sel[::2, ::2, ::2]) * 2
    if len(pts) < 4:
        return 1.0
    if len(q) == 0:
        return 0.0
    try:
        hull = Delaunay(pts[ConvexHull(pts).vertices])
    except QhullError:                                       # a flat point set
        return 1.0
    return float((hull.find_simplex(q) >= 0).mean())


# ----------------------------------------------------------------------------- specimen mask (§1)
def threshold(counts, centres):
    """t = min(Otsu, low-side triangle) of a histogram (counts, bin centres). With two modes Otsu falls between them. With one
    main mode Otsu would split it, and the foot of that mode on its low side (the triangle threshold of Zack et al. 1977, taken
    on the low side) lies lower."""
    h = np.asarray(counts, np.float64)
    p, lo = int(np.argmax(h)), int(np.flatnonzero(h)[0])
    w = p - lo
    foot = centres[lo + int(np.argmax(h[p] * np.arange(w) - w * h[lo:p]))] if w else centres[lo]
    return min(float(threshold_otsu(hist=(h, centres))), float(foot))


def _hist(chunks, nbins=256):
    """Histogram of the values of the arrays from chunks() (called twice) over their range. -> (counts, bin centres)."""
    r = [(float(x.min()), float(x.max())) for x in chunks() if x.size]
    lo, hi = min(a for a, _ in r), max(b for _, b in r)
    counts = sum(np.histogram(x, nbins, (lo, hi))[0] for x in chunks())
    e = np.linspace(lo, hi, nbins + 1)
    return counts, 0.5 * (e[1:] + e[:-1])


def specimen_mask(fine, voxel_mm, params: Params = Params()):
    """§1 specimen mask: a voxel is specimen when it has signal and isotropic texture, each decided by `threshold`.
    Signal: log intensity of the measured voxels (fine > 0) above t. Voxels below t have no signal: they are left out of the
    texture and become specimen only where the hole filling encloses them. Every voxel fine > 0 counts as measured.
    Texture on the signal voxels m: the normalised Gaussian smoothing b = G(x m) / G(m) (sigma texture_bandpass_mm), per array
    axis the running coefficient of variation of b over texture_window_mm (counted where >= half the window has signal),
    C = the minimum over the axes, averaged over blocks of ~texture_grid_mm (blocks >= half signal), log C smoothed over the
    blocks (normalised Gaussian, sigma texture_smooth_mm), above t, linearly interpolated to the fine grid > 0.5. Signal AND
    texture, components >= min_component of the mask kept, holes filled in every array plane, AND measured.
    fine: [D, H, W] OCT intensities. voxel_mm: spacing (mm).
    -> (bool [D, H, W], {signal_threshold (intensity), texture_threshold (C), volume_cm3})."""
    P, D = params, fine.shape[0]
    logs = lambda: (np.log(x[x > 0]) for x in (np.asarray(fine[z:z + _CHUNK], np.float32) for z in range(0, D, _CHUNK)))
    ts = float(np.exp(threshold(*_hist(logs))))
    sig = P.texture_bandpass_mm / voxel_mm
    win = int(round(P.texture_window_mm / voxel_mm)) // 2 * 2 + 1
    k = max(1, int(round(P.texture_grid_mm / voxel_mm)))
    pad, step = int(4 * sig + 0.5) + win // 2, max(1, _CHUNK // k) * k
    num = np.zeros([-(-s // k) for s in fine.shape])
    cnt = np.zeros_like(num)
    with np.errstate(divide="ignore", invalid="ignore"):
        for z0 in range(0, D, step):
            a, z1 = max(0, z0 - pad), min(D, z0 + step)
            x = np.asarray(fine[a:min(D, z1 + pad)], np.float32)
            m = (x > ts).astype(np.float32)
            x = np.divide(ndimage.gaussian_filter(x * m, sig), ndimage.gaussian_filter(m, sig), out=np.zeros_like(x), where=m > 0)
            C = np.full(x.shape, np.inf, np.float32)
            for ax in range(3):
                n = ndimage.uniform_filter1d(m, win, axis=ax)
                mu = ndimage.uniform_filter1d(x, win, axis=ax) / n
                cv = ndimage.uniform_filter1d(x * x, win, axis=ax) / n - mu * mu
                cv = np.sqrt(np.maximum(cv, 0)) / mu
                np.minimum(C, np.where(n >= 0.5, cv, np.inf), out=C)
                del n, mu, cv
            ok = (m > 0) & np.isfinite(C)
            core = slice(z0 - a, z1 - a)
            pc, pn = _pool(np.where(ok, C, 0)[core], k), _pool(ok[core].astype(np.float32), k)
            num[z0 // k:z0 // k + len(pc)], cnt[z0 // k:z0 // k + len(pn)] = pc, pn
            del x, m, C, ok
        real = np.einsum("i,j,k->ijk", *[np.minimum(k, s - k * np.arange(-(-s // k))) for s in fine.shape])
        C = num / cnt
    valid = (cnt >= 0.5 * real) & (C > 0)
    w, s = valid.astype(np.float64), P.texture_smooth_mm / (k * voxel_mm)
    L = np.divide(ndimage.gaussian_filter(np.log(np.where(valid, C, 1.0)) * w, s), ndimage.gaussian_filter(w, s),
                  out=np.zeros_like(w), where=valid)
    if not (valid.sum() > 1 and L[valid].std() > 0):
        raise ValueError("specimen mask: no texture variation in the measured data, supply an OCT mask")
    lt = threshold(*_hist(lambda: [L[valid]]))
    keep = (valid & (L > lt)).astype(np.float32)
    out = np.empty(fine.shape, bool)
    for z0 in range(0, D, _CHUNK):
        z1 = min(D, z0 + _CHUNK)
        out[z0:z1] = (_upsample(keep, k, fine.shape, (z0, z1)) > 0.5) & (np.asarray(fine[z0:z1]) > ts)
    out = _fill_planes(_components(out, P.min_component))
    out &= np.asarray(fine) > 0
    return out, {"signal_threshold": ts, "texture_threshold": float(np.exp(lt)),
                 "volume_cm3": float(out.sum()) * voxel_mm ** 3 / 1e3}

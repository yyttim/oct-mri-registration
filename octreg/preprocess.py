"""Masks (docs/METHOD.md §1): the MRI foreground and the OCT specimen mask from isotropic texture. numpy/scipy on the CPU;
arrays [D, H, W] on isotropic grids of voxel_mm (mm). Exact zeros in the OCT are missing data (black tiles): never specimen."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from skimage.filters import threshold_multiotsu, threshold_otsu

from .params import Params

_CHUNK = 64          # planes per chunk along axis 0: memory only, results do not depend on it


def _close(m, r):
    """Binary closing with a ball of radius r voxels on the edge-padded mask (a face that cuts the tissue is not eroded)."""
    if r < 1:
        return m
    o = np.arange(-r, r + 1) ** 2
    ball = o[:, None, None] + o[None, :, None] + o[None, None, :] <= r * r
    return ndimage.binary_closing(np.pad(m, r, mode="edge"), ball)[r:-r, r:-r, r:-r]


def _fill_planes(m):
    """Holes filled in every array plane (union over the three axes): background enclosed by specimen within a plane is
    specimen, also where the hole reaches a cut face of the block and so is not enclosed in 3-D."""
    out = m.copy()
    for ax in range(3):
        s = np.zeros((3, 3, 3), bool)
        s[tuple(1 if d == ax else slice(None) for d in range(3))] = ndimage.generate_binary_structure(2, 1)
        out |= ndimage.binary_fill_holes(m, structure=s)
    return out


def _components(m, min_fraction=None):
    """6-connected components of m -> (mask of the largest component, or of every component holding >= min_fraction of the
    mask voxels; number of components before the rule)."""
    lab, n = ndimage.label(m)
    size = np.bincount(lab.ravel())
    size[0] = 0
    keep = (np.arange(n + 1) == np.argmax(size)) if min_fraction is None else size >= min_fraction * size.sum()
    keep[0] = False
    return keep[lab], int(n)


def _pool(a, k):
    """Sums over k^3 blocks of a [D, H, W] zero-padded to multiples of k (block j covers voxels j k .. j k + k - 1)."""
    a = np.pad(a, [(0, -s % k) for s in a.shape])
    D, H, W = (s // k for s in a.shape)
    return a.reshape(D, k, H, k, W, k).sum(axis=(1, 3, 5))


def _upsample(a, k, shape, rows=None):
    """Linear interpolation of a block grid a (block centres at j k + (k - 1) / 2 voxels, clamped beyond the outer centres) at
    the voxels of a grid of `shape`; rows = (lo, hi) returns only rows lo:hi along axis 0. -> float32."""
    x = np.asarray(a, np.float32)
    for ax, n in enumerate(shape):
        c = np.clip((np.arange(n) - (k - 1) / 2) / k, 0, a.shape[ax] - 1)[slice(*rows) if ax == 0 and rows else slice(None)]
        i = np.floor(c).astype(int)
        f = (c - i).astype(np.float32).reshape([-1 if d == ax else 1 for d in range(3)])
        x = np.take(x, i, axis=ax) * (1 - f) + np.take(x, np.minimum(i + 1, a.shape[ax] - 1), axis=ax) * f
    return x


# ----------------------------------------------------------------------------- foreground (histogram-valley rule)
def foreground(arr, voxel_mm, params: Params = Params()):
    """Foreground mask by the histogram-valley rule (the MRI foreground). Positive finite values below their
    p99.5, 256-bin histogram smoothed over 5 bins, peaks of its square root (so the narrow background peak of a box-averaged
    grid cannot hide the tissue peak) of prominence >= 5 % of the maximum and >= 5 bins apart; walking left
    from the rightmost peak, t = the first valley < valley_ratio x the smaller of its two peaks whose darker peak lies below
    medium_fraction x the brightest peak (status 'ok'): a valley between two peaks of tissue, such as grey and white matter in a
    crop without medium, is passed over and recorded. Without such a valley t = min(p1, 0.5 x lower cut of a 3-class
    multi-Otsu) (status 'no_valley': the whole field of view). Mask = arr > t, closing (ball of 2 voxels), holes filled,
    components >= min_component of the foreground kept. arr: [D, H, W]; voxel_mm: spacing.
    -> (bool [D, H, W], {threshold (arr units), status, valley_ratio, volume_cm3, n_components, tissue_valleys_passed})."""
    v = np.asarray(arr, np.float32)
    s = v[np.isfinite(v) & (v > 0)]
    s = s[s < np.percentile(s, 99.5)] if s.size else s
    if s.size == 0:
        raise ValueError("foreground: the image has no positive intensity variation")
    h, e = np.histogram(s, bins=256)
    c, hs = 0.5 * (e[1:] + e[:-1]), ndimage.uniform_filter1d(h.astype(float), 5)
    pk = find_peaks(np.sqrt(hs), prominence=0.05 * np.sqrt(hs.max()), distance=5)[0]
    t, status, ratio, passed = None, "no_valley", None, []
    for j, lo in enumerate(pk[:-1][::-1]):
        hi = pk[len(pk) - 1 - j]
        val = lo + int(np.argmin(hs[lo:hi + 1]))
        r = float(hs[val] / max(min(hs[lo], hs[hi]), 1.0))
        if r < params.valley_ratio:
            if c[lo] < params.medium_fraction * c[pk[-1]]:
                t, status, ratio = float(c[val]), "ok", r
                break
            passed.append({"threshold": float(c[val]), "valley_ratio": r, "darker_peak_of_brightest": float(c[lo] / c[pk[-1]])})
    if t is None:
        t = float(min(np.percentile(s, 1), 0.5 * threshold_multiotsu(s, classes=3)[0]))
    m = ndimage.binary_fill_holes(_close(v > t, 2))
    m, n = _components(m, params.min_component)
    return m, {"threshold": t, "status": status, "valley_ratio": ratio, "volume_cm3": float(m.sum()) * voxel_mm ** 3 / 1e3,
               "n_components": n, "tissue_valleys_passed": passed}


# ----------------------------------------------------------------------------- specimen mask (§1)
def threshold(counts, centres):
    """t = min(Otsu, low-side triangle) of a histogram (counts, bin centres). With two modes Otsu falls between them; with one
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
    Signal: log intensity of the measured voxels (fine > 0) above t; voxels below count as missing. Texture on the signal
    voxels: band-pass b = G(x m) / G(m) (sigma texture_bandpass_mm), per array axis the running coefficient of variation of b
    over texture_window_mm (counted where >= half the window has signal), C = the minimum over the axes, averaged over blocks of
    ~texture_grid_mm (blocks >= half signal), log C smoothed over the blocks (normalised Gaussian, sigma texture_smooth_mm),
    above t, linearly interpolated to the fine grid > 0.5. Signal AND texture; components >= min_component of the mask kept,
    holes filled in every array plane, AND measured. fine: [D, H, W] OCT intensities; voxel_mm: spacing (mm).
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
        raise ValueError("specimen mask: no texture variation in the measured data; supply an OCT mask")
    lt = threshold(*_hist(lambda: [L[valid]]))
    keep = (valid & (L > lt)).astype(np.float32)
    out = np.empty(fine.shape, bool)
    for z0 in range(0, D, _CHUNK):
        z1 = min(D, z0 + _CHUNK)
        out[z0:z1] = (_upsample(keep, k, fine.shape, (z0, z1)) > 0.5) & (np.asarray(fine[z0:z1]) > ts)
    out = _fill_planes(_components(out, P.min_component)[0])
    out &= np.asarray(fine) > 0
    return out, {"signal_threshold": ts, "texture_threshold": float(np.exp(lt)),
                 "volume_cm3": float(out.sum()) * voxel_mm ** 3 / 1e3}

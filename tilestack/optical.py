"""The optical covariate of the depth axis. OCT measures optical path, and the apparent depth of a voxel grows with the
refractive index of the material above it, which rises with mu_s. The covariate of a raw voxel is the level above the
embedding gel's own (zero_level) summed down its column from the cut. It is read from a level map of the registered
volume (level_map): there every position holds the slab that imaged it nearest its cut, where the estimate has lost
least with depth, and not a slab's own deep planes. The solve scales it by one parameter, beta.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from . import data, solve
from .compose import BELOW

SMOOTH_MM = 0.08                                       # in-plane smoothing of the level map
SMOOTH = (0.02 / data.VOX, SMOOTH_MM / data.LEVEL_MM, SMOOTH_MM / data.LEVEL_MM)     # planes, working pixels


def zero_level(v):
    """The level the covariate counts from, the embedding gel's: the lowest mode of the levels v (0.01 bins on whole
    hundredths, smoothed over 5 bins, a mode holding at least 5 % of the highest)."""
    e = np.arange(np.floor(v.min() * 100) - 2, np.ceil(v.max() * 100) + 3) / 100
    h = np.convolve(np.histogram(v, bins=e)[0], np.ones(5) / 5, mode="same")
    i = next(i for i in range(2, len(h) - 2) if h[i] >= h[i - 2:i + 3].max() and h[i] >= 0.05 * h.max())
    return float(0.5 * (e[i] + e[i + 1]))


def level_map(vol, z0, src, zero=None):
    """Local level from the registered volume on the working grid (vol (Z, Y, X) mu_s, z0 its first common plane, src
    its source map): log10 mu_s smoothed over the voxels from the reliable planes only, and zero, zero_level over those
    voxels unless given. Where no reliable voxel is near, the map is linear along depth between the levels above and
    below, carried down from the last at the bottom of the stack, and the zero above the data."""
    ok_vox = (vol > 0) & (src >= 0) & (src < BELOW)
    num = gaussian_filter(np.where(ok_vox, np.log10(np.where(ok_vox, vol, 1.0)), 0.0).astype(np.float32), SMOOTH)
    den = gaussian_filter(ok_vox.astype(np.float32), SMOOTH)
    ok = den > 0.2
    M = np.where(ok, num / np.maximum(den, 1e-6), np.nan).astype(np.float32)
    del num, den
    zero = zero_level(M[ok_vox & ok]) if zero is None else float(zero)
    M[~ok] = zero
    nz = M.shape[0]
    zi = np.arange(nz, dtype=np.int32)[:, None, None]
    for y0 in range(0, M.shape[1], 64):                     # linear along depth between the levels above and below
        o, m = ok[:, y0:y0 + 64], M[:, y0:y0 + 64]
        up = np.maximum.accumulate(np.where(o, zi, -1), 0)
        dn = np.flip(np.minimum.accumulate(np.flip(np.where(o, zi, nz), 0), 0), 0)
        a = np.where(dn < nz, (zi - up) / np.maximum(dn - up, 1), 0.0).astype(np.float32)
        mu = np.take_along_axis(m, np.clip(up, 0, nz - 1), 0); md = np.take_along_axis(m, np.clip(dn, 0, nz - 1), 0)
        M[:, y0:y0 + 64] = np.where(~o & (up >= 0), mu * (1 - a) + md * a, m)
    return dict(L=M, z0=z0, zero=zero)


def optical_from_level(s, sol, lay, lev):
    """Optical covariate of slab s on its working grid (planes, y, x): each raw voxel is placed (without the optical
    term), and the level map's level there above the map's zero is summed down the column under the cut."""
    L, z0 = lev["L"], lev["z0"]
    S = data.geometry40(s)[0]
    my, mx = S.shape
    y40, x40 = np.mgrid[0:my, 0:mx]
    x, y = x40 * data.DOWN + (data.DOWN - 1) / 2.0, y40 * data.DOWN + (data.DOWN - 1) / 2.0     # raw centres of the working pixels
    u, v, w = solve.displacement(sol, lay, s, x, y)
    ix = (x + u + data.PAD + 0.5) / data.DOWN - 0.5; iy = (y + v + data.PAD + 0.5) / data.DOWN - 0.5     # the volume's lattice
    dz = w - z0
    C = np.zeros((data.NP, my, mx), np.float32)
    run = np.zeros((my, mx), np.float32)
    for p in range(data.NP):
        lv = map_coordinates(L, [(p + dz).ravel(), iy.ravel(), ix.ravel()], order=1, mode="nearest").reshape(my, mx)
        run += np.clip(p - S + 1.0, 0.0, 1.0) * np.maximum(lv - lev["zero"], 0.0)
        C[p] = run
    return C


def load_level(path, zero):
    """The level map written by export.save_level, as level_map returns it, with its zero (zero_log10 of transforms.json)."""
    import nibabel as nib
    img = nib.load(path)
    return dict(L=np.transpose(np.asarray(img.dataobj, np.float32), (2, 1, 0)), z0=int(round(img.affine[2, 3] / data.VOX)), zero=zero)


def covariate(sol, lay, lev):
    """cov(slab) -> optical covariate from the level map, computed once per slab."""
    cache = {}

    def cov(s):
        if s not in cache:
            cache[s] = optical_from_level(s, sol, lay, lev)
        return cache[s]
    return cov

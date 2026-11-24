"""The block (docs/TILESTACK.md, Usage), its slabs and their cut surfaces."""
import json
import os

import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt, median_filter

from . import BELOW

if not os.environ.get("TILESTACK_BLOCK"):
    raise RuntimeError("TILESTACK_BLOCK is not set: python -m tilestack BLOCK_JSON OUT_DIR sets it to the block's JSON")
BLOCK = json.load(open(os.environ["TILESTACK_BLOCK"]))
SLAB_FILE, WORK, SLABS = BLOCK["slab_file"], BLOCK["work"], [int(s) for s in BLOCK["slabs"]]
assert all(0 <= s < BELOW for s in SLABS), f"slab numbers run from 0 to {BELOW - 1}: the source map adds {BELOW} to them"
VOX, NEIGHBOURS = float(BLOCK["voxel_mm"]), int(BLOCK["neighbours"])
MARGIN, DEEP, DEEP_A = (int(round(float(BLOCK[k]) / VOX)) for k in ("margin_mm", "deep_mm", "deep_a_mm"))
PAD = int(round(0.64 / VOX))                   # margin of the output grid around the raw field
LEVEL_MM = 0.04                                # working grid in-plane: the 40 um volume
SURFACE_TOL = round(0.06 / VOX, 6)             # planes a column may lie off its 3 x 3 neighbourhood's median cut
DOWN = int(round(LEVEL_MM / VOX))              # raw voxels per working voxel, in-plane
assert abs(DOWN * VOX - LEVEL_MM) < 1e-9, f"the working grid ({LEVEL_MM} mm) must hold a whole number of voxels"
os.makedirs(WORK, exist_ok=True)


def _shape():
    path = f"{WORK}/shape.json"
    if not os.path.exists(path):
        shapes = {}
        for s in SLABS:
            with h5py.File(SLAB_FILE.format(s), "r") as f:
                shapes[s] = f["M"].shape
        assert len(set(shapes.values())) == 1, f"the slabs differ in shape: {shapes}"
        json.dump(list(shapes[SLABS[0]]), open(path, "w"))
    return tuple(json.load(open(path)))


NP, NY, NX = _shape()                          # planes of the imaging window, in-plane size of the maps
XC, YC = NX / 2.0, (NY + 1) / 2.0              # centre of the field, the origin of a pair's depth offset plane


def check_work(settings):
    """The caches in WORK hold for the settings they were made with: record them, or stop where they differ."""
    path = f"{WORK}/settings.json"
    settings = json.loads(json.dumps(settings))
    if not os.path.exists(path):
        json.dump(settings, open(path, "w"), indent=1)
        return
    old = json.load(open(path))
    diff = sorted(k for k in set(old) | set(settings) if old.get(k) != settings.get(k))
    if diff:
        raise RuntimeError(f"the caches in {WORK} were made with other settings ({', '.join(diff)}): use an empty work folder")


def load(s):
    with h5py.File(SLAB_FILE.format(s), "r") as f:
        return f["M"][()].astype(np.float32)


def load_log(s):
    M = load(s)
    np.clip(M, 1e-5, None, out=M)
    np.log10(M, out=M)
    return M


def load40(s):
    """mu_s on the working grid: the in-plane mean over DOWN x DOWN voxels, float32, cached."""
    path = f"{WORK}/slab040_{s:03d}.npy"
    if not os.path.exists(path):
        M = load(s)
        ny, nx = M.shape[1] // DOWN * DOWN, M.shape[2] // DOWN * DOWN
        np.save(path, M[:, :ny, :nx].reshape(M.shape[0], ny // DOWN, DOWN, nx // DOWN, DOWN).mean((2, 4), dtype=np.float32))
    return np.load(path)


def surface(M):
    """Cut surface (plane index) per pixel (docs/TILESTACK.md, Cut surface) and the coverage mask: where the middle
    plane of the window holds data."""
    valid = M[M.shape[0] // 2] > 0
    L = np.log10(np.clip(M, 1e-6, None))
    J = np.minimum(np.minimum(L[:-2], L[1:-1]), L[2:])                  # the three planes from z down
    J[0] = -np.inf
    J[1] -= L[0]
    J[2] -= np.maximum(L[0], L[1])
    J[3:] -= np.maximum(np.maximum(L[:-5], L[1:-4]), L[2:-3])          # the three planes over z
    del L
    S = J.argmax(0).astype(np.float32)
    del J
    iy, ix = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    med = median_filter(S[iy, ix], size=3, mode="nearest")             # single columns are not pits
    return np.where(valid & (np.abs(S - med) > SURFACE_TOL), med, S).astype(np.float32), valid


def geometry(s):
    """Cached cut surface and coverage mask of slab s on the raw grid."""
    path = f"{WORK}/geom_{s:03d}.npz"
    if not os.path.exists(path):
        surf, valid = surface(load(s))
        np.savez_compressed(path, surf=surf, valid=valid)
    z = np.load(path)
    return z["surf"], z["valid"]


def geometry40(s):
    """The cut surface and coverage of slab s on the working grid, as load40 reduces mu_s: the deepest cut of the raw
    pixels of a working pixel, so that the margin under the cut holds for all of them, and coverage by all of them."""
    surf, valid = geometry(s)
    my, mx = surf.shape[0] // DOWN, surf.shape[1] // DOWN
    S = surf[:my * DOWN, :mx * DOWN].reshape(my, DOWN, mx, DOWN).max((1, 3)).astype(np.float32)
    V = valid[:my * DOWN, :mx * DOWN].reshape(my, DOWN, mx, DOWN).all((1, 3))
    return S, V


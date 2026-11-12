"""The block, its slabs and their cut surfaces.

The block is described by a JSON file, named by the environment variable TILESTACK_BLOCK (python -m tilestack sets it):
    slab_file   path of a slab with {} for its number, e.g. "/data/block/Mus_slice{:03d}.mat"
    work        folder for the caches of the run
    slabs       slab numbers in cut order, all with the same imaging window
    voxel_mm    the isotropic voxel of the maps
    neighbours  slabs ahead that still share tissue: the window's depth against the cut step
    margin_mm   skipped under the cut surface: the surface reflection
    deep_mm     end of the reliable planes: below it the estimate rises toward the window's bottom
    deep_a_mm   deepest raw plane matched in the older slab of a pair and taken by the volume
A slab is a MATLAB v7.3 file with one array M (z, y, x) of mu_s per um, read as h5py returns it: z grows with depth
and plane 0 lies above the cut surface. Lengths of the method are given in mm and converted to voxels here, once.
"""
import json
import os

import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt, median_filter

if not os.environ.get("TILESTACK_BLOCK"):
    raise RuntimeError("TILESTACK_BLOCK is not set: python -m tilestack BLOCK_JSON OUT_DIR sets it to the block's JSON")
BLOCK = json.load(open(os.environ["TILESTACK_BLOCK"]))
SLAB_FILE, WORK, SLABS = BLOCK["slab_file"], BLOCK["work"], [int(s) for s in BLOCK["slabs"]]
VOX, NEIGHBOURS = float(BLOCK["voxel_mm"]), int(BLOCK["neighbours"])
MARGIN, DEEP, DEEP_A = (int(round(float(BLOCK[k]) / VOX)) for k in ("margin_mm", "deep_mm", "deep_a_mm"))
MATCH_MM = (0.08, 0.16, 0.24, 0.32, 0.40)     # depths under the newer slab's cut that the dense matcher reads
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
XC, YC = NX / 2.0, (NY + 1) / 2.0              # centre of the field, the origin of the linear terms


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


def surface(M, tol=SURFACE_TOL):
    """Cut surface (plane index) per pixel and the coverage mask. The cut is the plane where log10 mu_s rises most
    sharply along depth: the three planes from it down all lie above the three planes over it, by as much as possible.
    A rise, not a level: water reads brighter the less signal lies under it in the window, so no level separates it
    from the material in every column. A column more than tol planes from the median of its 3 x 3 neighbourhood takes
    that median: single columns are not pits. The coverage mask is where the middle plane of the window holds data."""
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
    med = median_filter(S[iy, ix], size=3, mode="nearest")
    return np.where(valid & (np.abs(S - med) > tol), med, S).astype(np.float32), valid


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


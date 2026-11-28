"""The block, its slabs and their cut surfaces (docs/TILESTACK.md, Grid and Cut surface).

The block is the JSON named by the environment variable TILESTACK_BLOCK, which python -m tilestack sets
(docs/TILESTACK.md, Usage). Its lengths are in mm, converted here to working voxels.
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
NEIGHBOURS = int(BLOCK["neighbours"])
GRID_MM = 0.02                                 # spacing of the working grid the method is set for
os.makedirs(WORK, exist_ok=True)


def _slabs():
    """Shape (z, y, x) and voxel in mm (z, y, x) of the slab files, the same for every slab."""
    found = {}
    for s in SLABS:
        with h5py.File(SLAB_FILE.format(s), "r") as f:
            um = np.ravel(f["res_um"]).astype(float) if "res_um" in f else np.zeros(0)
            if um.size != 3:
                raise ValueError(f"slab {s}: res_um must hold the voxel (x, y, depth) in um, three values")
            found[s] = (f["M"].shape, tuple(float(v) / 1000 for v in um[::-1]))
    if len(set(found.values())) > 1:
        raise ValueError(f"the slabs differ in shape or voxel: {found}")
    return found[SLABS[0]]


SHAPE, RES_MM = _slabs()                       # the slab files: shape and voxel in mm, (z, y, x)
_f = max(1, int(round(GRID_MM / RES_MM[2])))
VOX = RES_MM[2] * _f                           # the working voxel, mm
FACTOR = (int(round(VOX / RES_MM[0])), _f, _f)
if not np.isclose(RES_MM[1], RES_MM[2], rtol=1e-6) or not np.isclose(FACTOR[0] * RES_MM[0], VOX, rtol=1e-6):
    raise ValueError(f"the voxel {[v * 1000 for v in RES_MM[::-1]]} um (x, y, depth) has no working grid: x and y must be "
                     f"equal and the depth voxel must divide {VOX * 1000:g} um")
NP, NY, NX = (n // f for n, f in zip(SHAPE, FACTOR))   # the working grid: planes of the window, in-plane size
MARGIN, DEEP, DEEP_A = (int(round(float(BLOCK[k]) / VOX)) for k in ("margin_mm", "deep_mm", "deep_a_mm"))
if DEEP_A + 2 >= NP:
    raise ValueError(f"deep_a_mm ({BLOCK['deep_a_mm']}) reaches the end of the imaging window ({NP * VOX:g} mm)")
MATCH_MM = (0.08, 0.16, 0.24, 0.32, 0.40)     # depths under the newer slab's cut that the dense matcher reads
PAD = int(round(0.64 / VOX))                   # margin of the output grid around the field, working pixels
SURFACE_TOL = round(0.06 / VOX, 6)             # planes a column may lie off its 3 x 3 neighbourhood's median cut
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
    """mu_s of slab s at the voxel of its file, float32 (z, y, x)."""
    with h5py.File(SLAB_FILE.format(s), "r") as f:
        return f["M"][()].astype(np.float32)


def load_grid(s):
    """mu_s of slab s on the working grid: means over FACTOR blocks, read row by row (cached unless FACTOR is 1)."""
    if FACTOR == (1, 1, 1):
        return load(s)
    path = f"{WORK}/grid_{s:03d}.npy"
    if not os.path.exists(path):
        fz, fy, fx = FACTOR
        out = np.empty((NP, NY, NX), np.float32)
        with h5py.File(SLAB_FILE.format(s), "r") as f:
            for j in range(NY):
                b = f["M"][:NP * fz, j * fy:(j + 1) * fy, :NX * fx].astype(np.float32)
                out[:, j] = b.reshape(NP, fz, fy, NX, fx).mean((1, 2, 4))
        np.save(path, out)
    return np.load(path)


def load_log(s):
    M = load_grid(s)
    np.clip(M, 1e-5, None, out=M)
    np.log10(M, out=M)
    return M


def surface(M, tol=SURFACE_TOL):
    """Cut surface (plane index per pixel, float32) of M (z, y, x) on the working grid (docs/TILESTACK.md, Cut surface)
    and the coverage mask, where the middle plane of the window holds data."""
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
    """Cached cut surface and coverage mask of slab s on the working grid."""
    path = f"{WORK}/geom_{s:03d}.npz"
    if not os.path.exists(path):
        surf, valid = surface(load_grid(s))
        np.savez_compressed(path, surf=surf, valid=valid)
    z = np.load(path)
    return z["surf"], z["valid"]


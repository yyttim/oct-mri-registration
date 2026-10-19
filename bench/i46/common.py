"""Files, reference R and scores of the I46 test (bench/i46/README.md).

R is the authors' OCT -> MRI affine, from the test's OCT world to the world of the flip-1 MRI header.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from bench.paths import BENCH_RUNS, DANDI000026_DIR  # noqa: E402

SUB = DANDI000026_DIR / "sub-I46"
OCT_TIFF = SUB / "oct/sub-I46_ses-OCT_sample-BrocaAreaS01_OCT.ome.tiff"      # 627 x 1271 x 1230 (plane, y, x), uint8
VESSELS_TIFF = SUB / "oct/ves_seg.tif"                                         # OCT planes 0..560 only
MRI_DIR = SUB / "mri"
LABEL_DIR = SUB / "labels"
AUTHORS = SUB / "costantini_repo/results"
MASK = HERE / "octmask_24um.nii.gz"         # one voxel per 2 x 2 x 2 OCT voxels

FLIPS = {"flip1": ("sub-I46_ses-MRI_flip-1_VFA.nii.gz", 10), "flip2": ("sub-I46_ses-MRI_flip-2_VFA.nii.gz", 20),
         "flip3": ("sub-I46_ses-MRI_flip-3_VFA.nii.gz", 30)}
PRIMARY = "flip1"
T1MAP = MRI_DIR / "sub-I46_T1map.nii.gz"
LABELS = {"vessels": LABEL_DIR / "sub-I46_ses-MRI_space-EPIC_label-vessels_dseg.nii.gz",
          "layers": LABEL_DIR / "sub-I46_ses-MRI_space-EPIC_label-infrasupra_dseg.nii.gz"}   # 1 white matter, 2 infra, 3 supra
CONDITIONS = {"C3": (3.0, 2.0), "C6": (6.0, 5.0), "C12": (12.0, 10.0)}                       # margin, shift bound (mm)
SEEDS = 10
SEED0 = 4600                                # crop k: default_rng(4600 + k)
TIERS = (0.30, 1.0)                         # mm, mean distance from R
TIER_NAMES = ("<= 0.30 mm", "<= 1 mm", "beyond")

SLAB = 33                                   # OCT planes per 500 um slab
SPACING = np.array([0.012, 0.012, 500 / SLAB / 1000])   # mm
OCT_AFFINE = np.diag([*SPACING, 1.0])
VESSEL_PLANES = 561
TIFF_SHAPE = (627, 1271, 1230)
CHOSEN = {"depth_sign": 1, "slab08": 0, "swap": True, "sign_y": -1, "sign_x": -1}   # the authors' OCT header

WORK = BENCH_RUNS / "I46"
INPUTS = WORK / "inputs"
RUNS = WORK / "runs"
RESULTS = REPO / "bench/results/I46"
OCT = INPUTS / "oct.nii.gz"


def mri_file(image):
    return MRI_DIR / FLIPS[image][0]


def crop_name(image, cond, k):
    return f"{image}_{cond}_k{k}"


def crop_file(name):
    return INPUTS / "crops" / f"{name}.nii.gz"


def lta_matrix(path):
    lines = Path(path).read_text().splitlines()
    i = next(n for n, line in enumerate(lines) if line.strip().startswith("1 4 4"))
    return np.array([[float(x) for x in lines[i + 1 + r].split()] for r in range(4)])


@lru_cache(None)
def slab_lattices():
    """The 16 lattice centres [16, 3] of the authors' OCT -> LSFM fields svf.08 to svf.23 and the unit axes of the lattice
    of svf.08."""
    centres, ax = [], None
    for i in range(8, 24):
        img = nib.load(str(AUTHORS / f"oct2lsfm/svf.{i:02d}.nii.gz"))
        centres.append((img.affine @ np.r_[(np.array(img.shape[:3]) - 1) / 2, 1])[:3])
        if ax is None:
            ax = img.affine[:3, :3] / np.linalg.norm(img.affine[:3, :3], axis=0)
    return np.array(centres), ax


def oct_header(depth_sign, slab08, swap, sign_y, sign_x):
    """An OCT header M, TIFF index (plane, y, x) -> the authors' OCT world (mm), rebuilt from the svf lattices.
    LSFM slab i lies at OCT slab slab08 + depth_sign (i - 8). swap puts lattice axis 1 along TIFF y and axis 0 along TIFF x."""
    centres, ax = slab_lattices()
    ey, ex = (ax[:, 1], ax[:, 0]) if swap else (ax[:, 0], ax[:, 1])
    M = np.eye(4)
    # the planes 12 um apart as in the authors' frame, their affine A sets the depth scale
    M[:3, 0], M[:3, 1], M[:3, 2] = -depth_sign * ax[:, 2] * 0.012, sign_y * ey * 0.012, sign_x * ex * 0.012
    plane = np.mean([SLAB * (slab08 + depth_sign * (i - 8)) + (SLAB - 1) / 2 for i in range(8, 24)])
    M[:3, 3] = centres.mean(0) - M[:3, :3] @ np.array([plane, (TIFF_SHAPE[1] - 1) / 2, (TIFF_SHAPE[2] - 1) / 2])
    return M


def authors_oct_header():
    """The authors' OCT header: oct_header at CHOSEN, the axis order, signs and slab placement that reference_check.py frame
    finds best. Slab i holds OCT planes 33 (i - 8) to 33 (i - 8) + 32."""
    return oct_header(**CHOSEN)


def reference_matrix():
    """The reference R = recon2raw A M Pz OCT_AFFINE^-1, the test's OCT world -> MRI world of the flip-1 header.
    A (mri2oct/affine.lta) maps the authors' OCT world to the MRI recon world, the convention of their nitorch code.
    Pz: the test's index (x, y, plane) -> TIFF index (plane, y, x)."""
    Pz = np.zeros((4, 4))
    Pz[0, 2] = Pz[1, 1] = Pz[2, 0] = Pz[3, 3] = 1
    return (lta_matrix(AUTHORS / "mri/recon2raw.lta") @ lta_matrix(AUTHORS / "mri2oct/affine.lta") @ authors_oct_header()
            @ Pz @ np.linalg.inv(OCT_AFFINE))


def tissue_points():
    """The OCT tissue points (OCT world, mm): the specimen mask on a grid of every eighth OCT voxel.
    Mask voxel i covers OCT voxels 2i and 2i + 1, so mask voxel 4i sits at OCT voxel 8i + 0.5."""
    m = np.asarray(nib.load(str(MASK)).dataobj) > 0
    idx = np.argwhere(m[::4, ::4, ::4]) * 8 + 0.5
    return (OCT_AFFINE @ np.c_[idx, np.ones(len(idx))].T).T[:, :3]


@lru_cache(None)
def reference():
    return np.loadtxt(INPUTS / "ref_T_oct2mri.txt")


@lru_cache(None)
def tissue():
    return np.load(INPUTS / "tissue_oct.npy")


@lru_cache(None)
def label_fit():
    """The label-only fit L (A12 of reference_check.json), or None before reference_check.py has run."""
    import json
    p = RESULTS / "reference_check.json"
    T = json.loads(p.read_text(encoding="utf-8")).get("labels", {}).get("A12", {}).get("T_oct2mri") if p.is_file() else None
    return None if T is None else np.array(T, float)


def rotation(T, R):
    """Angle (deg) and unit axis (MRI world) of the polar rotation of T R^-1."""
    U, _, Vt = np.linalg.svd((T @ np.linalg.inv(R))[:3, :3])
    Q = U @ Vt
    ax = np.array([Q[2, 1] - Q[1, 2], Q[0, 2] - Q[2, 0], Q[1, 0] - Q[0, 1]])
    n = np.linalg.norm(ax)
    return float(np.degrees(np.arccos(np.clip((np.trace(Q) - 1) / 2, -1, 1)))), (ax / n if n > 1e-12 else ax)


def tier(mean):
    return TIER_NAMES[0 if mean <= TIERS[0] else 1 if mean <= TIERS[1] else 2]


def error(T, R=None):
    """Distances (mm) and rotation (deg) of T from R over the tissue points, and from L once it exists."""
    R = reference() if R is None else R
    T = np.asarray(T, float)
    X = np.c_[tissue(), np.ones(len(tissue()))].T
    e = np.linalg.norm((T @ X - R @ X)[:3], axis=0)
    ang, ax = rotation(T, R)
    mean = float(e.mean())
    c = np.r_[tissue().mean(0), 1]
    out = {"mean_mm": round(mean, 4), "median_mm": round(float(np.median(e)), 4),
           "p95_mm": round(float(np.percentile(e, 95)), 4), "max_mm": round(float(e.max()), 4), "rotation_deg": round(ang, 3),
           "rotation_axis": ax.round(3).tolist(), "centroid_mm": round(float(np.linalg.norm((T @ c - R @ c)[:3])), 4),
           "scales": np.linalg.norm(T[:3, :3], axis=0).round(4).tolist(),
           "singular_values": np.linalg.svd(T[:3, :3] @ np.linalg.inv(R[:3, :3]))[1].round(4).tolist(), "tier": tier(mean)}
    L = label_fit()
    if L is not None:
        out["mean_L_mm"] = round(float(np.linalg.norm((T @ X - L @ X)[:3], axis=0).mean()), 4)
        out["rotation_L_deg"] = round(rotation(T, L)[0], 3)
    return out


@lru_cache(None)
def _vessels():
    """The MRI vessel label points (flip-1 world) and, on the mask grid, the distance to the OCT vessels and the inner tissue.
    The inner tissue lies at least 0.15 mm inside the specimen mask."""
    import tifffile
    from scipy import ndimage
    f1 = nib.load(str(mri_file("flip1")))
    la = np.asarray(nib.load(str(LABELS["vessels"])).dataobj)
    if la.shape[:3] != f1.shape[:3]:
        raise ValueError(f"{LABELS['vessels']}: shape {la.shape}, the MRI {f1.shape}")
    v = np.argwhere(la == 255)
    del la
    mri_pts = (f1.affine @ np.c_[v, np.ones(len(v))].T).T[:, :3]
    ves = tifffile.imread(str(VESSELS_TIFF))
    if ves.shape != (VESSEL_PLANES,) + TIFF_SHAPE[1:]:
        raise ValueError(f"{VESSELS_TIFF}: shape {ves.shape}, expected {(VESSEL_PLANES,) + TIFF_SHAPE[1:]}")
    n = [x - x % 2 for x in ves.shape]
    vb = (ves[:n[0], :n[1], :n[2]] > 0).reshape(n[0] // 2, 2, n[1] // 2, 2, n[2] // 2, 2).any(axis=(1, 3, 5)).transpose(2, 1, 0)
    del ves
    sp = 2 * SPACING
    dist = ndimage.distance_transform_edt(~vb, sampling=sp).astype(np.float32)
    m = np.asarray(nib.load(str(MASK)).dataobj) > 0
    m = m[:vb.shape[0], :vb.shape[1], :vb.shape[2]]
    inner = ndimage.distance_transform_edt(m, sampling=sp) > 0.15
    return mri_pts, dist, sp, vb.shape, inner


def vessel_score(T):
    """Mean and median distance (mm) of the MRI vessel labels that T^-1 places inside the tissue to the nearest OCT vessel,
    None below 50 labels."""
    from scipy import ndimage
    x, dist, sp, shp, inner = _vessels()
    q = (np.linalg.inv(T) @ np.c_[x, np.ones(len(x))].T).T[:, :3] / sp - 0.25    # mask voxel b covers OCT voxels 2b, 2b + 1
    inside = np.all((q >= 2) & (q <= np.array(shp) - 3), 1)
    qi = np.round(q[inside]).astype(int)
    inside[np.flatnonzero(inside)[~inner[qi[:, 0], qi[:, 1], qi[:, 2]]]] = False
    d = ndimage.map_coordinates(dist, q[inside].T, order=1).astype(float)
    if len(d) < 50:
        return {"vessel_n": len(d), "vessel_mean_mm": None, "vessel_median_mm": None}
    return {"vessel_n": len(d), "vessel_mean_mm": round(float(d.mean()), 4), "vessel_median_mm": round(float(np.median(d)), 4)}

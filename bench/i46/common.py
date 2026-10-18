"""Shared definitions of the I46 test (bench/i46/README.md): the files, the reference, the inputs and the two scores.

The reference is the authors' OCT -> MRI affine of Costantini et al. 2023, from github.com/neuroscales/biccn-broca-I46 at commit
b36e398c, expressed from the world of the test's OCT input to the world of the DANDI flip-1 MRI header:

    R = R2W A M Pz S^-1

    A    results/mri2oct/affine.lta: the authors' OCT world -> MRI recon world (the convention of their nitorch code)
    R2W  results/mri/recon2raw.lta: MRI recon world -> the world of the DANDI flip-1 header
    M    the authors' OCT header, TIFF index (plane, y, x) -> their OCT world. It is not published, and is rebuilt here from
         their OCT -> LSFM fields results/oct2lsfm/svf.08 ... svf.23, one per 500 um slab of 33 OCT planes. With a8 the affine
         of svf.08 and ax its 3 x 3 part with unit columns, TIFF plane, y and x run along -ax[:, 2], -ax[:, 1] and -ax[:, 0] at
         12 um, and the mean of the 16 lattice centres (affine applied to (shape - 1) / 2) sits at TIFF index (263.5, 635,
         614.5), the mean over the slabs of 33 (i - 8) + 16 and the centre of the plane. The three signs are those under which
         the OCT placed by the reference correlates best with the MRI, and slab 08 (OCT planes 0..32) is the slab that puts
         the most OCT vessels on the MRI vessel labels (bench/i46/reference_check.py frame).
    Pz   the test's OCT index (x, y, plane) -> TIFF index (plane, y, x)
    S    the test's OCT header, diag(0.012, 0.012, 0.01515) mm: 12 um in the plane and, in depth, 500 um sections (Costantini
         et al. 2023) of 33 planes each. The dataset's own metadata give 12 um on every axis. The authors' frame places the
         planes 12 um apart and leaves the depth scale to their affine.

Scores of an affine T (OCT world -> MRI world):
    error         the distance from the reference, |T x - R x| over the OCT tissue points (the specimen mask on a grid of every
                  eighth OCT voxel, 96 um in the plane and 121 um in depth): mean, median, p95 and maximum, the rotation
                  angle and axis of T R^-1 (of its polar rotation), the shift of the tissue centroid, the scale along each OCT
                  axis (1 = size kept), the mean distance and rotation angle from the label-only fit L of reference_check.py
                  (the vessels alone, A12), and the tier of the mean distance from R: "<= 0.30 mm" (the reference band),
                  "<= 1 mm" or "beyond"
    vessel_score  the MRI vessel labels (value 255) that T places inside the OCT tissue, at least 0.15 mm from its edge and
                  within the annotated planes 0..560, and their distance to the nearest OCT vessel (DANDI ves_seg), mean and
                  median
    vessel_field  the vessel check of T with and without the deformation of a run, on the labels both place inside: the MRI
                  point x goes to the OCT world at T^-1 (x + u(x)), u the run's pull-back field
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
VESSELS_TIFF = SUB / "oct/ves_seg.tif"                                         # the OCT vessels of planes 0..560
MRI_DIR = SUB / "mri"
LABEL_DIR = SUB / "labels"
AUTHORS = SUB / "costantini_repo/results"
MASK = HERE / "octmask_24um.nii.gz"         # specimen mask of the test, one voxel per 2 x 2 x 2 OCT voxels (README.md, Inputs)

FLIPS = {"flip1": ("sub-I46_ses-MRI_flip-1_VFA.nii.gz", 10), "flip2": ("sub-I46_ses-MRI_flip-2_VFA.nii.gz", 20),
         "flip3": ("sub-I46_ses-MRI_flip-3_VFA.nii.gz", 30)}
PRIMARY = "flip1"                           # the 10 deg FLASH (README.md, Inputs)
T1MAP = MRI_DIR / "sub-I46_T1map.nii.gz"
LABELS = {"vessels": LABEL_DIR / "sub-I46_ses-MRI_space-EPIC_label-vessels_dseg.nii.gz",
          "layers": LABEL_DIR / "sub-I46_ses-MRI_space-EPIC_label-infrasupra_dseg.nii.gz"}   # 1 white matter, 2 infra, 3 supra
CONDITIONS = {"C3": (3.0, 2.0), "C6": (6.0, 5.0), "C12": (12.0, 10.0)}                       # margin, shift bound (mm)
SEEDS = 10
SEED0 = 4600                                # crop k of every condition: numpy default_rng(4600 + k)
TIERS = (0.30, 1.0)                         # mean distance from R (mm) of the two tiers
TIER_NAMES = ("<= 0.30 mm", "<= 1 mm", "beyond")

SPACING = np.array([0.012, 0.012, 500 / 33 / 1000])     # mm
OCT_AFFINE = np.diag([*SPACING, 1.0])
VESSEL_PLANES = 561
TIFF_SHAPE = (627, 1271, 1230)

WORK = BENCH_RUNS / "I46"
INPUTS = WORK / "inputs"                    # OCT, reference, tissue points, crops (prepare.py)
RUNS = WORK / "runs"                        # octreg runs
RESULTS = REPO / "bench/results/I46"
OCT = INPUTS / "oct.nii.gz"                 # the released OCT with the test's header


def mri_file(image):
    return MRI_DIR / FLIPS[image][0]


def crop_name(image, cond, k):
    return f"{image}_{cond}_k{k}"


def crop_file(name):
    return INPUTS / "crops" / f"{name}.nii.gz"


def lta_matrix(path):
    """The 4 x 4 matrix of an LTA file (the four lines after '1 4 4')."""
    lines = Path(path).read_text().splitlines()
    i = next(n for n, line in enumerate(lines) if line.strip().startswith("1 4 4"))
    return np.array([[float(x) for x in lines[i + 1 + r].split()] for r in range(4)])


def authors_oct_header():
    """M of the module docstring: TIFF index (plane, y, x) -> the authors' OCT world (mm)."""
    centres, a8 = [], None
    for i in range(8, 24):
        img = nib.load(str(AUTHORS / f"oct2lsfm/svf.{i:02d}.nii.gz"))
        centres.append((img.affine @ np.r_[(np.array(img.shape[:3]) - 1) / 2, 1])[:3])
        a8 = img.affine if a8 is None else a8
    ax = a8[:3, :3] / np.linalg.norm(a8[:3, :3], axis=0)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2] = -ax[:, 2] * 0.012, -ax[:, 1] * 0.012, -ax[:, 0] * 0.012
    plane = np.mean([33 * (i - 8) + 16 for i in range(8, 24)])
    M[:3, 3] = np.mean(centres, 0) - M[:3, :3] @ np.array([plane, (TIFF_SHAPE[1] - 1) / 2, (TIFF_SHAPE[2] - 1) / 2])
    return M


def reference_matrix():
    """R of the module docstring: the test's OCT world -> MRI world (flip-1 header)."""
    Pz = np.zeros((4, 4))
    Pz[0, 2] = Pz[1, 1] = Pz[2, 0] = Pz[3, 3] = 1
    return (lta_matrix(AUTHORS / "mri/recon2raw.lta") @ lta_matrix(AUTHORS / "mri2oct/affine.lta") @ authors_oct_header()
            @ Pz @ np.linalg.inv(OCT_AFFINE))


def tissue_points():
    """The OCT tissue points (OCT world mm): every fourth block (2 x 2 x 2 OCT voxels) of the specimen mask, at the centre of
    its first 2 x 2 x 2 OCT voxels (block 4i covers OCT voxels 8i and 8i + 1), a grid of every eighth OCT voxel (96 x 96 x 121
    um)."""
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
    """L: the label-only fit A12 of bench/results/I46/reference_check.json, or None before reference_check.py has run."""
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
    """T against the reference R (both OCT world -> MRI world), as in the module docstring."""
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
    """The MRI vessel label voxels (flip-1 header world), and on blocks of 2 x 2 x 2 OCT voxels (the lattice of the
    specimen mask) the distance to the nearest OCT vessel block and the tissue at least 0.15 mm from the edge of the mask.
    Planes 0..559 of the 561 annotated planes form whole blocks."""
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


def _vessel_distances(T, moved=None):
    """Distance (mm) of every MRI vessel label (moved to `moved` first, if given) placed in the OCT world by T^-1 to the nearest
    OCT vessel block; NaN outside the inner tissue. Block b covers OCT voxels 2b and 2b + 1, so a point x of the OCT world is
    looked up at block index x / (2 S) - 0.25."""
    from scipy import ndimage
    mri_pts, dist, sp, shp, inner = _vessels()
    x = mri_pts if moved is None else moved
    q = (np.linalg.inv(T) @ np.c_[x, np.ones(len(x))].T).T[:, :3] / sp - 0.25    # OCT world -> mask block index
    inside = np.all((q >= 2) & (q <= np.array(shp) - 3), 1)
    qi = np.round(q[inside]).astype(int)
    inside[np.flatnonzero(inside)[~inner[qi[:, 0], qi[:, 1], qi[:, 2]]]] = False
    d = np.full(len(q), np.nan)
    d[inside] = ndimage.map_coordinates(dist, q[inside].T, order=1)
    return d


def vessel_score(T):
    """The vessel check of T (module docstring). Fewer than 50 labels inside the tissue give no distance."""
    d = _vessel_distances(T)
    d = d[~np.isnan(d)]
    if len(d) < 50:
        return {"vessel_n": len(d), "vessel_mean_mm": None, "vessel_median_mm": None}
    return {"vessel_n": len(d), "vessel_mean_mm": round(float(d.mean()), 4), "vessel_median_mm": round(float(np.median(d)), 4)}


def vessel_field(T, field, affine):
    """vessel_field of the module docstring: field [3, D, H, W] (mm, MRI world axes) on the grid of `affine`, trilinear and 0
    outside it. -> {n, affine_median_mm, field_median_mm} over the labels both place inside."""
    from scipy import ndimage
    mri_pts = _vessels()[0]
    idx = (np.linalg.inv(affine) @ np.c_[mri_pts, np.ones(len(mri_pts))].T)[:3]
    u = np.stack([ndimage.map_coordinates(field[c], idx, order=1, mode="constant", cval=0.0) for c in range(3)], 1)
    da, df = _vessel_distances(T), _vessel_distances(T, mri_pts + u)
    both = ~np.isnan(da) & ~np.isnan(df)
    if both.sum() < 50:
        return {"n": int(both.sum()), "affine_median_mm": None, "field_median_mm": None}
    return {"n": int(both.sum()), "affine_median_mm": round(float(np.median(da[both])), 4),
            "field_median_mm": round(float(np.median(df[both])), 4)}

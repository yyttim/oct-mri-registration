"""Shared definitions of the DANDI:000874 evaluation (bench/dandi874/README.md): the blocks, the files, the reference affines,
the OCT tissue points and the error of an affine against the reference."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from bench.paths import BENCH_RUNS, DANDI874_DIR  # noqa: E402

BLOCKS = ("01", "02", "03", "04")
NAMES = {"01": "caudal medulla", "02": "rostral medulla and caudal pons", "03": "rostral pons", "04": "midbrain"}
CONTRASTS = [f"flip{i}" for i in range(1, 8)] + ["T1map", "PDmap"]
FLIP_DEG = {"flip1": 3, "flip2": 5, "flip3": 10, "flip4": 15, "flip5": 20, "flip6": 30, "flip7": 40}
CONDITIONS = {"C3": (3.0, 2.0), "C6": (6.0, 5.0), "C12": (12.0, 10.0)}     # margin, shift bound (mm)
SEEDS = 10
TIERS = (0.30, 1.0)                                                          # accurate, correct pose (mean error, mm)

OCT_DIR = DANDI874_DIR / "derivatives/OCT-pipeline/sub-SP002/micr"
MRI_DIR = DANDI874_DIR / "derivatives/MRI-pipeline/sub-SP002/anat"
WORK = BENCH_RUNS / "SP002"
INPUTS = WORK / "inputs"                  # converted OCT, reference affines, crops
RUNS = WORK / "runs"                      # octreg runs
RESULTS = REPO / "bench/results/SP002"


def oct_zarr(b):
    return OCT_DIR / f"sub-SP002_ses-OCT_sample-{b}_res-20um_OCT.ome.zarr"


def oct_lta(b):
    return OCT_DIR / f"sub-SP002_ses-OCT_sample-{b}_to-SP002_xfm.lta"


def mri_file(contrast):
    if contrast.startswith("flip"):
        return MRI_DIR / f"sub-SP002_ses-MRI_flip-{contrast[4:]}_chunk-2_VFA.nii.gz"
    return MRI_DIR / f"sub-SP002_ses-MRI_chunk-2_{contrast}_VFA.nii.gz"


def oct_nifti(b):
    return INPUTS / f"sample-{b}_oct_20um.nii.gz"


def reference(b):
    return np.loadtxt(INPUTS / f"sample-{b}_T_ref.txt")


def crop_name(b, contrast, cond, k):
    return f"s{b}_{contrast}_{cond}_k{k}"


def crop_file(name):
    return INPUTS / "crops" / f"{name}.nii.gz"


def lta_matrix(path):
    txt = Path(path).read_text()
    if "LINEAR_RAS_TO_RAS" not in txt:
        raise ValueError(f"{path}: not a RAS-to-RAS LTA")
    v = [float(x) for x in txt.split("1 4 4")[1].split("src volume info")[0].split()[:16]]
    T = np.array(v).reshape(4, 4)
    T[3] = [0, 0, 0, 1]
    return T


def lta_geometry(path, which):
    """FreeSurfer vox2ras of the 'src' or 'dst' volume of an LTA: P0 = cras - M (N / 2), M = directions x voxel size."""
    part = Path(path).read_text().split(f"{which} volume info")[1]
    num = lambda key: np.array([float(x) for x in re.search(rf"^{key}\s*=\s*(.+)$", part, re.M).group(1).split("#")[0].split()])
    dims, vs = num("volume").astype(int), num("voxelsize")
    M = np.c_[num("xras") * vs[0], num("yras") * vs[1], num("zras") * vs[2]]
    A = np.eye(4)
    A[:3, :3], A[:3, 3] = M, num("cras") - M @ (dims / 2)
    return dims, A


def tissue_points(oct_path, step=4):
    """World coordinates (mm) of the OCT tissue on every `step`-th voxel: the Otsu threshold of the values above the background
    floor that some blocks are clipped to, a closing and the largest connected component.
    -> (points [N, 3], tissue fraction)."""
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from octreg import io
    vol = io.load_volume(oct_path)                       # streamed plane by plane, so that a large file fits in memory
    v = np.stack([p[::step, ::step] for k, p in io.iter_planes(vol) if k % step == 0], axis=-1)
    img = nib.load(str(oct_path))
    m = v > threshold_otsu(v[v > v.min()])
    m = ndimage.binary_closing(m, iterations=2)
    lab, n = ndimage.label(m)
    if n > 1:
        m = lab == (1 + int(np.argmax(ndimage.sum(m, lab, range(1, n + 1)))))
    idx = np.argwhere(m) * step
    return (img.affine @ np.c_[idx, np.ones(len(idx))].T).T[:, :3], float(m.mean())


def tissue(b):
    f = INPUTS / f"s{b}_tissue_oct.npy"
    if not f.is_file():
        np.save(f, tissue_points(oct_nifti(b))[0].astype(np.float64))
    return np.load(f)


def error(b, T, R=None):
    """T against the reference R (both OCT world -> MRI world): |T x - R x| over the OCT tissue (mean, median, p95, max, mm),
    the maximum over the 8 corners of the tissue box (OCT voxel axes), and of D = T R^-1 the rotation angle of its polar
    rotation, the displacement at the tissue centroid and its singular values."""
    R = reference(b) if R is None else R
    pts = tissue(b)
    X = np.c_[pts, np.ones(len(pts))].T
    e = np.linalg.norm((T @ X - R @ X)[:3], axis=0)
    A = nib.load(str(oct_nifti(b))).affine
    v = (np.linalg.inv(A) @ X).T[:, :3]
    lo, hi = v.min(0), v.max(0)
    C = A @ np.array([[x, y, z, 1] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]).T
    ec = np.linalg.norm((T @ C - R @ C)[:3], axis=0)
    D = T @ np.linalg.inv(R)
    U, s, Vt = np.linalg.svd(D[:3, :3])
    ang = float(np.degrees(np.arccos(np.clip((np.trace(U @ Vt) - 1) / 2, -1, 1))))
    c = np.r_[pts.mean(0), 1]
    mean = float(e.mean())
    return {"mean_mm": round(mean, 4), "median_mm": round(float(np.median(e)), 4), "p95_mm": round(float(np.percentile(e, 95)), 4),
            "max_mm": round(float(e.max()), 4), "corners_max_mm": round(float(ec.max()), 4), "rotation_deg": round(ang, 3),
            "centroid_mm": round(float(np.linalg.norm((T @ c - R @ c)[:3])), 4), "singular_values": s.round(4).tolist(),
            "tier": "accurate" if mean <= TIERS[0] else "correct" if mean <= TIERS[1] else "failed"}

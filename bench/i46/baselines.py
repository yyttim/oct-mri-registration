#!/usr/bin/env python3
"""The standard registration tools on the I46 crops, each at one documented setting with no masks
(bench/baselines/standard_tools.py gives the settings, the commands, the conversions and the check). Every tool gets the
masked OCT that octreg gets, box-averaged to the MRI voxel size with its own header, and the MRI crop.

Runs: every tool on the C3 crops k = 0..9 of the 10 deg FLASH and on the C3 k = 0 crops of the 20 and 30 deg FLASH.

Three variants each change one thing in those settings, on the C3 crops k = 0..9 of the 10 deg FLASH, and read the input
files of the main runs (<bench_runs>/I46/baselines/<crop>/inputs):

--search   the tools' own global searches (standard_tools.SEARCH: FLIRT over -180 to 180 deg about every axis, antsAI before
           ANTs, greedy -search), in <bench_runs>/I46/baselines_search.
--mask     the tools that take a mask of the moving image, with the OCT tissue as that mask (standard_tools.MASK: the voxels > 0
           of the moving image, oct_mask.nii.gz next to the inputs of the crop), in <bench_runs>/I46/baselines_mask.
           mri_synthmorph register has no mask option.
--near     every tool from a start near the reference R, given as the header of the moving image, with the tool's option
           that starts from the header alignment (standard_tools.NEAR), in <bench_runs>/I46/baselines_near/<crop>.

The near start of crop k is S_k(x) = R(c + Q(x - c)) + d: Q turns by 10 deg about a unit axis, d is a shift of 1 mm in the
MRI world, both directions drawn as normalised standard normal vectors from numpy default_rng(4700 + k) (the axis first), and
c is the centre of the OCT tissue points (common.tissue, OCT world). R has a shear (its columns are up to 2.5 deg from
orthogonal), which a NIfTI qform cannot hold. Given a sform with a shear, ANTs, elastix and greedy (ITK) read the qform and
NiftyReg, FSL and FreeSurfer the sform. So every tool, and octreg's A14 (analyses.py near), starts from H_k, the start
closest to S_k that a qform holds: a rotation with a scale per axis, the least squares fit over the tissue points with
H_k(c) = S_k(c). The moving image is that of the main runs with the header H_k A (A its own header) in both sform and
qform, and the crop and the x10 copies of both lie next to it (baselines_near/<crop>/inputs, with start.json: the draws,
S_k, H_k and the distance between them over the tissue points). The transform T of a tool maps the world of that header,
and T H_k the OCT world. Each row also holds the start (start_pose, common.error of H_k) and its distance from S_k
(start_from_S).

    python bench/i46/baselines.py prepare [--search | --mask | --near]   the inputs and run.sh of every run, and queue.sh
    bash <bench_runs>/I46/baselines/queue.sh           runs them one at a time (baselines_<search|mask|near>/queue.sh)
    python bench/i46/baselines.py collect [--search | --mask | --near]   converts each transform (-> <run>/T_oct2mri.txt),
                                                       scores it against the reference (common.error and common.vessel_score),
                                                       and on the k = 0 crops checks the conversion against the tool's own
                                                       resampling of the OCT it ran on
-> bench/results/I46/baselines.json (baselines_search.json, baselines_mask.json, baselines_near.json), one entry per run. A run
   without a readable transform has the tier "tool error", the last lines of its log and the log line that names the error.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (FLIPS, OCT, PRIMARY, REPO, RESULTS, SEEDS, WORK, crop_file, crop_name, error, reference,  # noqa: E402
                    tissue, vessel_score)

sys.path.insert(0, str(REPO / "bench/baselines"))
import standard_tools as H  # noqa: E402

BL = WORK / "baselines"
BASE = {"search": WORK / "baselines_search", "mask": WORK / "baselines_mask", "near": WORK / "baselines_near"}
VARIANTS = {"search": H.SEARCH, "mask": H.MASK, "near": H.NEAR}
NEAR_SEED, NEAR_DEG, NEAR_MM = 4700, 10.0, 1.0


def runs(kind=None):
    if kind:
        return [(crop_name(PRIMARY, "C3", k), t) for k in range(SEEDS) for t in VARIANTS[kind]]
    crops = [crop_name(PRIMARY, "C3", k) for k in range(SEEDS)] + [crop_name(im, "C3", 0) for im in FLIPS if im != PRIMARY]
    return [(c, t) for c in crops for t in H.TOOLS]


def inputs(crop):
    """The input directory of a crop, shared by the main, search and mask runs (made once, kept if already there)."""
    return H.prepare_inputs(BL / crop / "inputs", OCT, crop_file(crop))


def near_start(k, A):
    """The near start of crop k (module docstring) for a moving image with the header A. -> dict: axis, shift_mm (MRI world),
    centre_mm (OCT world), S (S_k) and H (H_k as the header stores it, sform and qform in float32), each 4x4 OCT world ->
    MRI world."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    g = np.random.default_rng(NEAR_SEED + k)
    axis = g.standard_normal(3)
    axis /= np.linalg.norm(axis)
    shift = g.standard_normal(3)
    shift *= NEAR_MM / np.linalg.norm(shift)
    X = tissue()
    c = X.mean(0)
    Q, C, D = np.eye(4), np.eye(4), np.eye(4)
    Q[:3, :3] = Rotation.from_rotvec(np.radians(NEAR_DEG) * axis).as_matrix()
    C[:3, 3], D[:3, 3] = c, shift
    S = D @ reference() @ C @ Q @ np.linalg.inv(C)
    M = S[:3, :3]
    W = np.linalg.cholesky(np.cov((X - c).T, bias=True))          # mean |(L - M)(x - c)|^2 = |(L - M) W|^2
    lin = lambda p: Rotation.from_rotvec(p[:3]).as_matrix() * np.exp(p[3:])
    U, _, Vt = np.linalg.svd(M)
    p0 = np.r_[Rotation.from_matrix(U @ Vt).as_rotvec(), np.log(np.diag((U @ Vt).T @ M))]
    fit = least_squares(lambda p: ((lin(p) - M) @ W).ravel(), p0, xtol=1e-15, ftol=1e-15, gtol=1e-15)
    Hk = np.eye(4)
    Hk[:3, :3] = lin(fit.x)
    Hk[:3, 3] = (S @ np.r_[c, 1])[:3] - Hk[:3, :3] @ c
    Hk = (Hk @ A).astype(np.float32).astype(float) @ np.linalg.inv(A)
    return {"k": k, "seed": NEAR_SEED + k, "degrees": NEAR_DEG, "axis": axis.tolist(), "shift_mm": shift.tolist(),
            "centre_mm": c.tolist(), "S": S.tolist(), "H": Hk.tolist()}


def start_from_S(st):
    """The distance between H_k and S_k over the OCT tissue points (mm): mean and max."""
    X = np.c_[tissue(), np.ones(len(tissue()))].T
    e = np.linalg.norm(((np.array(st["H"]) - np.array(st["S"])) @ X)[:3], axis=0)
    return {"mean_mm": round(float(e.mean()), 4), "max_mm": round(float(e.max()), 4)}


def near_inputs(crop):
    """The inputs of the near runs of a crop in baselines_near/<crop>/inputs: oct.nii.gz (the moving image of the main runs
    with the header H_k A in sform and qform), mri.nii.gz, their x10 copies and start.json. Kept if already there."""
    d, src = BASE["near"] / crop / "inputs", inputs(crop)
    if (d / "start.json").is_file():
        return d
    from convert_affines import scale_header
    d.mkdir(parents=True, exist_ok=True)
    o = nib.load(str(src / "oct.nii.gz"))
    st = near_start(int(crop.rsplit("_k", 1)[1]), o.affine)
    Hk = np.array(st["H"])
    hdr = Hk @ o.affine
    img = nib.Nifti1Image(np.asarray(o.dataobj, np.float32), hdr)
    img.header.set_xyzt_units("mm")
    img.header.set_qform(hdr, code=1)
    img.header.set_sform(hdr, code=1)
    nib.save(img, str(d / "oct.nii.gz"))
    back = nib.load(str(d / "oct.nii.gz"))
    q, s = back.header.get_qform(), back.header.get_sform()
    if np.abs(s - hdr).max() > 1e-9 or np.abs(q - s).max() > 1e-5:
        raise ValueError(f"{d / 'oct.nii.gz'}: the header does not hold H_k A (sform {np.abs(s - hdr).max():.2g}, "
                         f"qform {np.abs(q - s).max():.2g})")
    shutil.copy(src / "mri.nii.gz", d / "mri.nii.gz")
    for f in ("oct", "mri"):
        nib.save(scale_header(nib.load(str(d / f"{f}.nii.gz")), 10.0), str(d / f"{f}_x10.nii.gz"))
    st["start_from_S"] = start_from_S(st)
    st["header_sform_minus_qform"] = float(np.abs(q - s).max())
    (d / "start.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
    return d


def start_of(crop):
    return json.loads((BASE["near"] / crop / "inputs" / "start.json").read_text(encoding="utf-8"))


def prepare(kind=None):
    if kind == "mask":
        for crop, _ in runs(kind):
            H.prepare_mask(inputs(crop))
    H.prepare(BASE[kind] if kind else BL, runs(kind), near_inputs if kind == "near" else inputs)


def collect(base=BL, out=RESULTS / "baselines.json", kind=None):
    def score(crop, T):
        row = {**error(T), **vessel_score(T)}
        if kind == "near":
            st = start_of(crop)
            row.update(start_pose=error(np.array(st["H"])), start_from_S=st["start_from_S"])
        return row
    start = (lambda crop: np.array(start_of(crop)["H"])) if kind == "near" else None
    rows = H.collect(base, score, lambda crop: crop.endswith("C3_k0"), tools=list(VARIANTS[kind]) if kind else H.TOOLS,
                     inputs=(lambda crop: BL / crop / "inputs") if kind in ("search", "mask") else None, start=start)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(len(rows), "runs ->", out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prepare", "collect"])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--search", dest="kind", action="store_const", const="search", help="the tools' own global searches")
    g.add_argument("--mask", dest="kind", action="store_const", const="mask", help="the OCT tissue as the moving mask")
    g.add_argument("--near", dest="kind", action="store_const", const="near", help="the start near R in the moving header")
    ap.add_argument("--runs", type=Path, help="collect: the folder of runs (default <bench_runs>/I46/baselines, or "
                                              "baselines_<search|mask|near>)")
    ap.add_argument("-o", "--out", type=Path, help="collect: the JSON file to write (default bench/results/I46/baselines.json, "
                                                   "or baselines_<search|mask|near>.json)")
    a = ap.parse_args()
    if a.step == "prepare":
        prepare(a.kind)
    else:
        collect(a.runs or BASE.get(a.kind, BL),
                a.out or RESULTS / (f"baselines_{a.kind}.json" if a.kind else "baselines.json"), a.kind)

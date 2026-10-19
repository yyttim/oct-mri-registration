#!/usr/bin/env python3
"""Standard registration tools on the I46 crops (bench/i46/README.md), one documented setting each and no mask.

The settings and commands are in bench/baselines/standard_tools.py. The moving image is the OCT box-averaged to 0.15 mm, the
fixed image the C3 crops k = 0..9 of the 10 deg FLASH and k = 0 of the 20 and 30 deg FLASH. Each option changes one thing on
the 10 deg C3 crops, into <bench_runs>/I46/baselines_<kind>:

--search   the tools' own global searches
--mask     the test's specimen mask as the moving mask of the tools that take one
--near     a start near R as the header of the moving image, each tool started from the header

    python bench/i46/baselines.py prepare [--search | --mask | --near]   the inputs, the run.sh of every run and queue.sh
    bash <bench_runs>/I46/baselines/queue.sh                             runs them one at a time (baselines_<kind>/queue.sh)
    python bench/i46/baselines.py collect [--search | --mask | --near] [--runs DIR] [-o FILE]

collect writes <run>/T_oct2mri.txt, scores it against R and checks the conversion on the k = 0 crops. The scores go to
bench/results/I46/baselines[_<kind>].json. A run without a readable transform is a tool error.
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
from common import (FLIPS, MASK, OCT, PRIMARY, REPO, RESULTS, SEEDS, WORK, crop_file, crop_name, error,  # noqa: E402
                    reference, tissue, vessel_score)

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
    """The input directory of a crop, shared by the main, search and mask runs."""
    return H.prepare_inputs(BL / crop / "inputs", OCT, crop_file(crop))


def near_start(k, A):
    """Start S_k of crop k, R turned by NEAR_DEG about the tissue centre and shifted by NEAR_MM, and its header pose H_k.
    A: the moving header. H_k is the rotation with a scale per axis closest to S_k, as the qform that ANTs, elastix and
    greedy read cannot hold a shear. S and H map OCT world -> MRI world."""
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
    """Mean and max distance (mm) of H_k from S_k over the tissue points."""
    X = np.c_[tissue(), np.ones(len(tissue()))].T
    e = np.linalg.norm(((np.array(st["H"]) - np.array(st["S"])) @ X)[:3], axis=0)
    return {"mean_mm": round(float(e.mean()), 4), "max_mm": round(float(e.max()), 4)}


def near_inputs(crop):
    """Inputs of the near runs of a crop: the main inputs with the OCT header H_k A, their x10 copies and start.json."""
    d, src = BASE["near"] / crop / "inputs", inputs(crop)
    record = json.loads((src / "inputs.json").read_text(encoding="utf-8"))
    if H.made_from(d / "inputs.json", record):
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
    (d / "inputs.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return d


def start_of(crop):
    return json.loads((BASE["near"] / crop / "inputs" / "start.json").read_text(encoding="utf-8"))


def prepare(kind=None):
    if kind == "mask":
        for crop in dict.fromkeys(c for c, _ in runs(kind)):
            H.prepare_mask(inputs(crop), MASK)
    H.prepare(BASE[kind] if kind else BL, runs(kind), near_inputs if kind == "near" else inputs)


def collect(base=BL, out=RESULTS / "baselines.json", kind=None):
    """Score every finished run under base and write the rows to out. A near run also scores its start H_k."""
    def score(crop, T):
        row = {**error(T), **vessel_score(T)}
        if kind == "near":
            st = start_of(crop)
            row.update(start_pose=error(np.array(st["H"])), start_from_S=st["start_from_S"])
        return row
    start = (lambda crop: np.array(start_of(crop)["H"])) if kind == "near" else None
    rows = H.collect(base, score, lambda crop: crop.endswith("C3_k0"), tools=list(VARIANTS[kind]) if kind else H.TOOLS,
                     inputs=(lambda crop: BL / crop / "inputs") if kind in ("search", "mask") else None, start=start)
    missing = sorted(set(runs(kind)) - {(r["crop"], r["tool"]) for r in rows})
    if missing:
        sys.exit(f"{len(missing)} runs never ran in {base}: " + ", ".join(f"{c}/{t}" for c, t in missing))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(len(rows), "runs ->", out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prepare", "collect"])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--search", dest="kind", action="store_const", const="search", help="the tools' own global searches")
    g.add_argument("--mask", dest="kind", action="store_const", const="mask", help="the test's specimen mask as moving mask")
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

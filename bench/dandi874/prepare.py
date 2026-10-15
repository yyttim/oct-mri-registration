#!/usr/bin/env python3
"""Inputs of the DANDI:000874 evaluation, from the download of bench/dandi874/fetch.sh, into <bench_runs>/SP002/inputs.

OCT      each block's 20 um scattering-coefficient volume (OME-Zarr level 0) as NIfTI in the authors' own frame: the zarr holds
         their NIfTI header (group 'nifti'), whose affine equals the source geometry of their OCT -> MRI LTA. Both are checked.
         The array is the zarr's (z, y, x) transposed to (x, y, z).
Reference each block's LTA (LINEAR_RAS_TO_RAS) as a 4x4 text file sample-<block>_T_ref.txt. The destination geometry of every
         LTA must equal the header of every chunk-2 MRI image, which is checked.
Tissue   the OCT tissue points of each block (common.tissue_points), the extent the crops are built around and the points the
         errors are measured on.
Crops    the MRI cut like the I58 input: an axis-aligned box on the MRI voxel grid around the 0.5 and 99.5 percentiles of the
         tissue placed by the reference, a margin m on every face, then shifted by d, each d_i uniform in [-s, s]
         (numpy default_rng(100 block + k)): conditions C3 (m 3, s 2 mm), C6 (6, 5) and C12 (12, 10), k = 0..9, on the 5 deg
         FLASH (flip2), and the C3 k = 0 box on the eight other chunk-2 images. -> crops/<name>.nii.gz and crops.json
Context  the same tissue rule applied to I58 (its OCT placed by octreg's I58 result, bench/results/I58/result.json) gives the
         face margins of the I58 MRI crop, and octreg's MRI foreground (octreg.preprocess.foreground) gives the share of each C3
         k = 0 crop and of the I58 crop that holds MRI tissue. -> bench/results/SP002/crops.json and context.json

    python bench/dandi874/prepare.py [--context]      (--context: only the context, from the crops already made)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BLOCKS, CONDITIONS, CONTRASTS, INPUTS, REPO, RESULTS, SEEDS, crop_file, crop_name, lta_geometry,  # noqa: E402
                    lta_matrix, mri_file, oct_lta, oct_nifti, oct_zarr, tissue, tissue_points)


def convert(b):
    import zarr
    out = oct_nifti(b)
    if out.is_file():
        return
    g = zarr.open_group(str(oct_zarr(b)), mode="r")
    hdr = dict(g["nifti"].attrs)
    A = np.eye(4)
    A[:3] = np.array(hdr["Affine"], float)
    dims, A_lta = lta_geometry(oct_lta(b), "src")
    a0 = g["0"]
    if tuple(hdr["Dim"][:3]) != tuple(a0.shape[::-1]) or tuple(dims) != tuple(a0.shape[::-1]):
        sys.exit(f"block {b}: shapes differ: zarr {a0.shape}, NIfTI header {hdr['Dim']}, LTA source {dims}")
    if not np.allclose(A, A_lta, atol=1e-4):
        sys.exit(f"block {b}: the zarr NIfTI header and the LTA source geometry differ")
    img = nib.Nifti1Image(np.ascontiguousarray(np.asarray(a0[:], np.float32).transpose(2, 1, 0)), A)
    img.header.set_xyzt_units("mm")
    img.set_sform(A, 1); img.set_qform(A, 1)
    nib.save(img, str(out))
    print(f"block {b}: OCT {img.shape} at 20 um, header = the LTA source geometry", flush=True)


def main():
    (INPUTS / "crops").mkdir(parents=True, exist_ok=True)
    mri = {c: nib.load(str(mri_file(c))) for c in CONTRASTS}
    for b in BLOCKS:
        convert(b)
        _, A_dst = lta_geometry(oct_lta(b), "dst")
        bad = [c for c, img in mri.items() if not np.allclose(img.affine, A_dst, atol=1e-3)]
        if bad:
            sys.exit(f"block {b}: the LTA destination geometry differs from {bad}")
        np.savetxt(INPUTS / f"sample-{b}_T_ref.txt", lta_matrix(oct_lta(b)), fmt="%.10f")
    A = mri["flip2"].affine.astype(float)
    h = np.linalg.norm(A[:3, :3], axis=0)
    rows = []
    for b in BLOCKS:
        T = np.loadtxt(INPUTS / f"sample-{b}_T_ref.txt")
        pts = tissue(b)
        vox = (np.linalg.inv(A) @ T @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
        lo, hi = np.percentile(vox, 0.5, 0), np.percentile(vox, 99.5, 0)
        for cond, (m, s) in CONDITIONS.items():
            for k in range(SEEDS):
                seed = 100 * int(b) + k
                d = np.random.default_rng(seed).uniform(-s, s, 3) / h
                c_lo = np.maximum(np.floor(lo - m / h + d).astype(int), 0)
                c_hi = np.minimum(np.ceil(hi + m / h + d).astype(int) + 1, np.array(mri["flip2"].shape[:3]))
                row = {"block": b, "cond": cond, "k": k, "seed": seed, "margin_mm": m, "shift_bound_mm": s,
                       "shift_mm": (d * h).round(3).tolist(), "crop_voxels": [c_lo.tolist(), c_hi.tolist()],
                       "crop_extent_mm": ((c_hi - c_lo) * h).round(2).tolist(),
                       "face_margins_low_mm": ((lo - c_lo + 0.5) * h).round(2).tolist(),
                       "face_margins_high_mm": ((c_hi - 0.5 - hi) * h).round(2).tolist(),
                       "tissue_inside_crop": round(float(np.all((vox >= c_lo - 0.5) & (vox <= c_hi - 0.5), 1).mean()), 4)}
                for c in (CONTRASTS if (cond, k) == ("C3", 0) else ["flip2"]):
                    name = crop_name(b, c, cond, k)
                    if not crop_file(name).is_file():
                        nib.save(mri[c].slicer[c_lo[0]:c_hi[0], c_lo[1]:c_hi[1], c_lo[2]:c_hi[2]], str(crop_file(name)))
                    rows.append({"name": name, "contrast": c, **row})
        print(f"block {b}: {sum(r['block'] == b for r in rows)} crops", flush=True)
    (INPUTS / "crops.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "crops.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    context(rows)


def foreground_share(path):
    from octreg import preprocess as pp
    from octreg.params import Params
    m = nib.load(str(path))
    return round(float(pp.foreground(np.asarray(m.dataobj, np.float32), float(np.linalg.norm(m.affine[:3, 0])), Params())[0].mean()), 4)


def context(rows):
    from bench.paths import MRI_I58, OCT_I58
    m = nib.load(str(MRI_I58))
    A, h = m.affine.astype(float), np.linalg.norm(m.affine[:3, :3], axis=0)
    T = np.asarray(json.loads((REPO / "bench/results/I58/result.json").read_text(encoding="utf-8"))["T_oct2mri"], float)
    pts, _ = tissue_points(OCT_I58)
    vox = (np.linalg.inv(A) @ T @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
    lo, hi = np.percentile(vox, 0.5, 0), np.percentile(vox, 99.5, 0)
    margins = {}
    for cond in CONDITIONS:
        r = [x for x in rows if x["cond"] == cond and x["contrast"] == "flip2"]
        margins[cond] = [min(min(x["face_margins_low_mm"] + x["face_margins_high_mm"]) for x in r),
                         max(max(x["face_margins_low_mm"] + x["face_margins_high_mm"]) for x in r)]
    ctx = {"i58_face_margins_low_mm": ((lo + 0.5) * h).round(2).tolist(),
           "i58_face_margins_high_mm": ((np.array(m.shape[:3]) - 0.5 - hi) * h).round(2).tolist(),
           "crop_face_margins_mm": margins,
           "mri_foreground_share": {**{b: foreground_share(crop_file(crop_name(b, "flip2", "C3", 0))) for b in BLOCKS},
                                    "I58": foreground_share(MRI_I58)}}
    (RESULTS / "context.json").write_text(json.dumps(ctx, indent=1), encoding="utf-8")
    print(json.dumps(ctx), flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context", action="store_true", help="only the context, from the crops already made")
    if ap.parse_args().context:
        context(json.loads((INPUTS / "crops.json").read_text(encoding="utf-8")))
    else:
        main()

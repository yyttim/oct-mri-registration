#!/usr/bin/env python3
"""Inputs of the I46 test (bench/i46/README.md), from the download of bench/i46/fetch.sh, into <bench_runs>/I46/inputs.

OCT      the DANDI OME-TIFF (627 x 1271 x 1230, plane y x, uint8), its voxels unchanged, as the array (x, y, plane) with the
         header diag(0.012, 0.012, 0.01515) mm -> oct.nii.gz
Ref      the authors' affine in the frame of this OCT (common.reference_matrix) -> ref_T_oct2mri.txt
Tissue   the OCT tissue points of the scores (common.tissue_points), OCT world mm -> tissue_oct.npy
Crops    the MRI cut around the block, as octreg expects it: an axis-aligned box on the MRI voxel grid around the 0.5 and 99.5
         percentiles of the tissue placed by the reference, a margin m on every face, then shifted by d, each d_i uniform in
         [-s, s] (numpy default_rng(4600 + k)): C3 (m 3, s 2 mm), C6 (6, 5) and C12 (12, 10), k = 0..9. C3 on the 10, 20 and
         30 deg FLASH, C6 and C12 on the 10 deg FLASH. flip-2 carries a header that is a 120 deg rotation (a cyclic permutation
         of the axes) of the same voxel grid as flip-1, so every crop is written with the flip-1 header (the arrays are voxel
         aligned). -> crops/<name>.nii.gz and crops.json, also as bench/results/I46/crops.json
Context  -> bench/results/I46/context.json:
         reference         determinant and singular values of its 3 x 3 part, and its own vessel check (common.vessel_score)
         tissue_points     the number of OCT tissue points
         grey_white_cnr    the grey/white contrast of each MRI image, which chose the primary image: (mean grey - mean white) /
                           sqrt((var grey + var white) / 2), white matter = layer label 1, grey matter = labels 2 and 3 (infra-
                           and supragranular cortex), each eroded by one voxel (6-connected), inside the box of MRI voxels that
                           holds the OCT block placed by the reference, 54 voxels (8.1 mm) wider on every face
         oct_noise         the 2 x 2 x 2 voxel blocks of the OCT more than 0.3 mm outside the specimen mask: their number and
                           the share whose mean is not zero

    python bench/i46/prepare.py [--context]      (--context: only the context, from the inputs already made)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CONDITIONS, FLIPS, INPUTS, LABELS, MASK, OCT, OCT_AFFINE, OCT_TIFF, PRIMARY,  # noqa: E402
                    RESULTS, SEED0, SEEDS, SPACING, T1MAP, TIFF_SHAPE, crop_file, crop_name, mri_file, reference_matrix,
                    tissue_points, vessel_score)

MARGIN_BOX = 54                                 # MRI voxels around the block for the grey/white contrast
MASK_AFFINE = OCT_AFFINE @ np.array([[2, 0, 0, 0.5], [0, 2, 0, 0.5], [0, 0, 2, 0.5], [0, 0, 0, 1]])   # voxel i: OCT 2i, 2i + 1


def read_oct():
    import tifffile
    o = tifffile.imread(str(OCT_TIFF))
    if o.shape != TIFF_SHAPE or o.dtype != np.uint8:
        sys.exit(f"{OCT_TIFF}: {o.shape} {o.dtype}, expected {TIFF_SHAPE} uint8")
    return o.transpose(2, 1, 0)                 # (x, y, plane)


def read_mask(shape):
    img = nib.load(str(MASK))
    m = np.asarray(img.dataobj) > 0
    if m.shape != tuple(n // 2 for n in shape) or not np.allclose(img.affine, MASK_AFFINE, atol=1e-6):
        sys.exit(f"{MASK}: {m.shape}, expected half of the OCT array {shape} with the header {MASK_AFFINE.tolist()}")
    return m


def save(arr, affine, path):
    img = nib.Nifti1Image(arr, affine)
    img.header.set_xyzt_units("mm")
    img.set_sform(affine, 1)
    img.set_qform(affine, 1)
    nib.save(img, str(path))


def oct_input():
    """oct.nii.gz of the module docstring, kept if already there."""
    read_mask(TIFF_SHAPE[::-1])
    if not OCT.is_file():
        save(read_oct(), OCT_AFFINE, OCT)
        print(f"OCT {TIFF_SHAPE[::-1]} -> {OCT}", flush=True)


def cnr(a, white, grey):
    g, w = a[grey].astype(np.float64), a[white].astype(np.float64)
    return round(float((g.mean() - w.mean()) / np.sqrt((g.var() + w.var()) / 2)), 3)


def contrast_box(R, f1):
    """The box of MRI voxels (flip-1 grid, slices) that holds the corners of the OCT array placed by R, MARGIN_BOX wider."""
    n = np.array(TIFF_SHAPE[::-1]) - 1
    corners = np.array([[x, y, z, 1] for x in (0, n[0]) for y in (0, n[1]) for z in (0, n[2])], float).T
    v = (np.linalg.inv(f1.affine) @ R @ OCT_AFFINE @ corners)[:3]
    lo = np.maximum(np.floor(v.min(1)).astype(int) - MARGIN_BOX, 0)
    hi = np.minimum(np.ceil(v.max(1)).astype(int) + MARGIN_BOX, np.array(f1.shape[:3]) - 1)
    return tuple(slice(a, b + 1) for a, b in zip(lo, hi))


def layer_masks(box):
    from scipy import ndimage
    lab = np.asarray(nib.load(str(LABELS["layers"])).dataobj[box])
    return ndimage.binary_erosion(lab == 1), ndimage.binary_erosion((lab == 2) | (lab == 3))


def main():
    INPUTS.mkdir(parents=True, exist_ok=True)
    (INPUTS / "crops").mkdir(exist_ok=True)
    oct_input()
    R = reference_matrix()
    np.savetxt(INPUTS / "ref_T_oct2mri.txt", R, fmt="%.10f")
    pts = tissue_points()
    np.save(INPUTS / "tissue_oct.npy", pts)
    print(f"reference det {np.linalg.det(R[:3, :3]):.4f}, singular values {np.round(np.linalg.svd(R[:3, :3])[1], 4)}; "
          f"{len(pts)} tissue points", flush=True)

    f1 = nib.load(str(mri_file("flip1")))
    A = f1.affine.astype(float)
    h = np.linalg.norm(A[:3, :3], axis=0)
    shape = np.array(f1.shape[:3])
    vox = (np.linalg.inv(A) @ R @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
    lo, hi = np.percentile(vox, 0.5, 0), np.percentile(vox, 99.5, 0)
    tissue_centre = R @ np.r_[pts.mean(0), 1]
    rows, boxes = [], []
    for cond, (mg, s) in CONDITIONS.items():
        for k in range(SEEDS):
            d = np.random.default_rng(SEED0 + k).uniform(-s, s, 3) / h
            c_lo = np.maximum(np.floor(lo - mg / h + d).astype(int), 0)
            c_hi = np.minimum(np.ceil(hi + mg / h + d).astype(int) + 1, shape)
            centre = A @ np.r_[(c_lo + c_hi - 1) / 2, 1]
            row = {"cond": cond, "k": k, "seed": SEED0 + k, "margin_mm": mg, "shift_bound_mm": s,
                   "shift_mm": (d * h).round(3).tolist(), "crop_voxels": [c_lo.tolist(), c_hi.tolist()],
                   "crop_extent_mm": ((c_hi - c_lo) * h).round(2).tolist(),
                   "crop_centre_to_tissue_centre_mm": round(float(np.linalg.norm(centre[:3] - tissue_centre[:3])), 3),
                   "tissue_inside_crop": round(float(np.all((vox >= c_lo - 0.5) & (vox <= c_hi - 0.5), 1).mean()), 4)}
            images = list(FLIPS) if cond == "C3" else [PRIMARY]
            rows += [{"name": crop_name(im, cond, k), "image": im, "flip_deg": FLIPS[im][1], **row} for im in images]
            boxes.append((cond, k, c_lo, c_hi, images))
    box = contrast_box(R, f1)
    white, grey = layer_masks(box)
    contrast = {"box_voxels": [[int(b.start) for b in box], [int(b.stop) for b in box]]}
    for im, (fname, _) in FLIPS.items():
        arr = np.asarray(nib.load(str(mri_file(im))).dataobj)
        if arr.shape[:3] != tuple(shape):
            sys.exit(f"{fname}: {arr.shape}, the flip-1 image {tuple(shape)}")
        contrast[im] = cnr(arr[box], white, grey)
        img = nib.Nifti1Image(arr, A, f1.header)
        for cond, k, c_lo, c_hi, images in boxes:
            if im in images and not crop_file(crop_name(im, cond, k)).is_file():
                nib.save(img.slicer[c_lo[0]:c_hi[0], c_lo[1]:c_hi[1], c_lo[2]:c_hi[2]], str(crop_file(crop_name(im, cond, k))))
        print(f"{im}: crops written, grey/white contrast {contrast[im]}", flush=True)
        del arr, img
    contrast["T1map"] = cnr(np.asarray(nib.load(str(T1MAP)).dataobj[box]), white, grey)
    for f in (INPUTS / "crops.json", RESULTS / "crops.json"):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"{len(rows)} crops", flush=True)
    context(R, contrast)


def context(R=None, contrast=None):
    from scipy import ndimage
    R = np.loadtxt(INPUTS / "ref_T_oct2mri.txt") if R is None else R
    if contrast is None:
        f1 = nib.load(str(mri_file("flip1")))
        box = contrast_box(R, f1)
        white, grey = layer_masks(box)
        contrast = {"box_voxels": [[int(b.start) for b in box], [int(b.stop) for b in box]],
                    **{im: cnr(np.asarray(nib.load(str(mri_file(im))).dataobj[box]), white, grey) for im in FLIPS},
                    "T1map": cnr(np.asarray(nib.load(str(T1MAP)).dataobj[box]), white, grey)}
    oa = read_oct()
    m = read_mask(oa.shape)
    n = [2 * x for x in m.shape]
    b = oa[:n[0], :n[1], :n[2]].reshape(m.shape[0], 2, m.shape[1], 2, m.shape[2], 2).mean((1, 3, 5), dtype=np.float32)
    del oa
    far = ndimage.distance_transform_edt(~m, sampling=2 * SPACING) > 0.3
    ctx = {"reference": {"det": round(float(np.linalg.det(R[:3, :3])), 4),
                         "singular_values": np.linalg.svd(R[:3, :3])[1].round(4).tolist(), **vessel_score(R)},
           "tissue_points": int(len(np.load(INPUTS / "tissue_oct.npy"))), "grey_white_cnr": contrast,
           "oct_noise": {"blocks_beyond_0.3mm": int(far.sum()), "nonzero_share": round(float((b[far] > 0).mean()), 4)}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "context.json").write_text(json.dumps(ctx, indent=1), encoding="utf-8")
    print(json.dumps(ctx), flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context", action="store_true", help="only the context, from the inputs already made")
    if ap.parse_args().context:
        context()
    else:
        main()

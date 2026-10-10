#!/usr/bin/env python3
"""The shared inputs of the baseline comparison (bench only, not part of the package).

    python bench/baselines/prepare_inputs.py --oct OCT.nii.gz --mri MRI.nii.gz --run OCTREG_RUN --masks PREP_DIR -o DIR

Writes into DIR: oct_0.08mm.nii.gz and oct_0.15mm.nii.gz (the OCT box-averaged to isotropic grids, original world header),
mri_0.08mm.nii.gz (the crop as given) and mri_0.15mm.nii.gz, the field-of-view (FOV) masks oct_fov_*.nii.gz (voxels > 0),
the moving images of the deformable stage oct_affine_0.08mm.nii.gz and oct_affine_0.15mm.nii.gz (the 20 um OCT through the
affine of OCTREG_RUN onto the MRI grids) with oct_affine_fov_0.08mm.nii.gz, and, for the evaluation, copies of octreg's
masks from PREP_DIR (oct_mask, oct_valid and mri_mask of the bench/ablate.py preprocessing).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from octreg import geometry as G, io  # noqa: E402


def fov(src: Path, dst: Path):
    im = nib.load(src)
    nib.save(nib.Nifti1Image((np.asarray(im.dataobj, np.float32) > 0).astype(np.uint8), im.affine), dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oct", required=True, type=Path)
    ap.add_argument("--mri", required=True, type=Path)
    ap.add_argument("--run", required=True, type=Path, help="octreg run directory (T_oct2mri.txt, result.json)")
    ap.add_argument("--masks", required=True, type=Path, help="prep directory with oct_mask / oct_valid / mri_mask .nii.gz")
    ap.add_argument("-o", "--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    oct_ = io.load_volume(a.oct)
    for mm in (0.08, 0.15):
        arr, A = G.resample_iso(oct_, mm)
        io.save_nifti(arr.astype(np.float32), A, a.out / f"oct_{mm:.2f}mm.nii.gz")
        fov(a.out / f"oct_{mm:.2f}mm.nii.gz", a.out / f"oct_fov_{mm:.2f}mm.nii.gz")
    shutil.copy(a.mri, a.out / "mri_0.08mm.nii.gz")
    arr, A = G.resample_iso(io.load_volume(a.mri), 0.15)
    io.save_nifti(arr.astype(np.float32), A, a.out / "mri_0.15mm.nii.gz")
    for mm in ("0.08", "0.15"):
        subprocess.run([sys.executable, "-m", "octreg", "apply", "--run", str(a.run), "--moving", str(a.oct), "--reference",
                        str(a.out / f"mri_{mm}mm.nii.gz"), "-o", str(a.out / f"oct_affine_{mm}mm.nii.gz"), "--affine-only"],
                       check=True, env={**__import__("os").environ, "PYTHONPATH": str(REPO)})
    fov(a.out / "oct_affine_0.08mm.nii.gz", a.out / "oct_affine_fov_0.08mm.nii.gz")
    for f in ("oct_mask.nii.gz", "oct_valid.nii.gz", "mri_mask.nii.gz"):
        shutil.copy(a.masks / f, a.out / f)
    print("inputs in", a.out)


if __name__ == "__main__":
    main()

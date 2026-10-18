#!/usr/bin/env python3
"""The shared inputs of the I58 baseline comparison.

    python bench/baselines/prepare_inputs.py --oct OCT.nii.gz --mri MRI.nii.gz --run OCTREG_RUN -o DIR

Writes into DIR: oct_0.08mm.nii.gz (the OCT box-averaged to 0.08 mm isotropic, its original world header) with its
field-of-view (FOV) mask oct_fov_0.08mm.nii.gz (voxels > 0) and mri_0.08mm.nii.gz (the crop as given), the inputs of the
affine stage; and oct_affine_0.08mm.nii.gz, the 20 um OCT through the affine of OCTREG_RUN onto the MRI grid (octreg apply
--affine-only), the moving image of the deformable stage (deform_tools.py), whose fixed image is mri_0.08mm.nii.gz.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from octreg import geometry as G, io  # noqa: E402
from octreg.register import apply  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oct", required=True, type=Path)
    ap.add_argument("--mri", required=True, type=Path)
    ap.add_argument("--run", required=True, type=Path, help="octreg run directory (T_oct2mri.txt, result.json)")
    ap.add_argument("-o", "--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    arr, A = G.resample_iso(io.load_volume(a.oct), 0.08)
    io.save_nifti(arr, A, a.out / "oct_0.08mm.nii.gz")
    io.save_nifti((arr > 0).astype(np.uint8), A, a.out / "oct_fov_0.08mm.nii.gz")
    shutil.copy(a.mri, a.out / "mri_0.08mm.nii.gz")
    apply(a.run, a.oct, a.out / "mri_0.08mm.nii.gz", a.out / "oct_affine_0.08mm.nii.gz", affine_only=True)
    print("inputs in", a.out)


if __name__ == "__main__":
    main()

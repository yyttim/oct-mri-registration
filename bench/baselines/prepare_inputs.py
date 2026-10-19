#!/usr/bin/env python3
"""The shared inputs of the I58 baseline comparison.

    python bench/baselines/prepare_inputs.py --oct OCT.nii.gz --mri MRI.nii.gz -o DIR

Writes into DIR the inputs, oct_0.08mm.nii.gz (the OCT box-averaged to 0.08 mm in its own world), its field-of-view mask
oct_fov_0.08mm.nii.gz (voxels > 0), mri_0.08mm.nii.gz (the crop as given) and oct_0.15mm.nii.gz for the antsAI search, too
slow at 0.08 mm.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from octreg import geometry as G, io  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oct", required=True, type=Path)
    ap.add_argument("--mri", required=True, type=Path)
    ap.add_argument("-o", "--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    arr, A = G.resample_iso(io.load_volume(a.oct), 0.08)
    io.save_nifti(arr, A, a.out / "oct_0.08mm.nii.gz")
    io.save_nifti((arr > 0).astype(np.uint8), A, a.out / "oct_fov_0.08mm.nii.gz")
    io.save_nifti(*G.resample_iso(io.load_volume(a.oct), 0.15), a.out / "oct_0.15mm.nii.gz")
    shutil.copy(a.mri, a.out / "mri_0.08mm.nii.gz")
    print("inputs in", a.out)


if __name__ == "__main__":
    main()

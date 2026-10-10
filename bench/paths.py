"""Roots of the bench scripts, relative to the repository by default. The package (octreg/) never reads this.

    OCTREG_PROJECT_ROOT   where the runs live (bench_runs/...), by default the repository root
    OCTREG_DATA_ROOT      where the data live, by default <repository>/data
    OCTREG_I58_DIR        the folder that holds the two I58 input files, by default <data root>/I58

bench_runs/ and data/ of the repository are in .gitignore. Those three are the only overrides, and nothing else in the
benchmark hard-codes a path.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("OCTREG_PROJECT_ROOT") or REPO)
DATA_ROOT = Path(os.environ.get("OCTREG_DATA_ROOT") or REPO / "data")
BENCH_RUNS = PROJECT_ROOT / "bench_runs"
I58_DIR = Path(os.environ.get("OCTREG_I58_DIR") or DATA_ROOT / "I58")
OCT_I58 = I58_DIR / "I58_Brainstem_mus_Slice_full_20um_corr.nii.gz"
MRI_I58 = I58_DIR / "I58_brainstem_MRI_cropped_to_OCT.nii.gz"

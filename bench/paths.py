"""Roots of the bench scripts, relative to the repository by default. Four environment variables move them:

    OCTREG_PROJECT_ROOT     where the runs live (bench_runs/... and, for the baseline comparison, baselines/...), by default
                            the repository root
    OCTREG_DATA_ROOT        where the data live, by default <repository>/data
    OCTREG_I58_DIR          the folder that holds the two I58 input files, by default <data root>/I58
    OCTREG_DANDI000026_DIR  the download of DANDI:000026 sub-I46 and the authors' registration files (bench/i46/fetch.sh), by
                            default <data root>/dandi-000026

bench_runs/, baselines/ and data/ of the repository are in .gitignore.
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
DANDI000026_DIR = Path(os.environ.get("OCTREG_DANDI000026_DIR") or DATA_ROOT / "dandi-000026")

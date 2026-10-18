"""Roots of the bench scripts, inside the repository unless the OCTREG_* environment variables below move them.
The runs go to <project root>/bench_runs and the baseline comparison to <project root>/baselines.
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

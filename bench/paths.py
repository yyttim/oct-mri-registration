"""Roots of the bench scripts, inside the repository unless the OCTREG_* environment variables below move them.
The runs go to <project root>/bench_runs and the baseline comparison to <project root>/baselines. On Windows the Linux
tools run in WSL.
"""
from __future__ import annotations

import os
import platform
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("OCTREG_PROJECT_ROOT") or REPO)
DATA_ROOT = Path(os.environ.get("OCTREG_DATA_ROOT") or REPO / "data")
BENCH_RUNS = PROJECT_ROOT / "bench_runs"
I58_DIR = Path(os.environ.get("OCTREG_I58_DIR") or DATA_ROOT / "I58")
OCT_I58 = I58_DIR / "I58_Brainstem_mus_Slice_full_20um_corr.nii.gz"
MRI_I58 = I58_DIR / "I58_brainstem_MRI_cropped_to_OCT.nii.gz"
DANDI000026_DIR = Path(os.environ.get("OCTREG_DANDI000026_DIR") or DATA_ROOT / "dandi-000026")
WSL = platform.system() == "Windows"


def host_path(p: Path) -> str:
    """Path as the shell of the Linux tools sees it, /mnt/<drive>/... in WSL."""
    s = str(Path(p).resolve()).replace("\\", "/")
    return "/mnt/" + s[0].lower() + s[2:] if WSL and len(s) > 1 and s[1] == ":" else s

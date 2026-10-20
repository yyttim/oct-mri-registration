#!/usr/bin/env python3
"""Run octreg once more on a pair and compare its affine with a stored run of the same octreg version.

    python bench/repeat.py OCT MRI --run STORED -o OUT [--device cuda]

OUT gets the outputs of octreg register at its defaults. bench/results/repeat.json records, under the path of STORED inside
<project root>/bench_runs (else its name), the octreg version and code, the largest absolute difference between the two
T_oct2mri.txt, whether the two files are identical, and the Python, PyTorch, CUDA, GPU and OS of the run. A stored run of
another octreg version, whose code.txt names another octreg code, or that was not made at the defaults on the two files
given, without masks or a voxel size, is refused.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.ablate import code_id  # noqa: E402
from bench.paths import BENCH_RUNS  # noqa: E402
from octreg import __version__  # noqa: E402
from octreg.params import Params  # noqa: E402
from octreg.register import register  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results" / "repeat.json"


def key(run):
    """The path of run inside BENCH_RUNS with forward slashes, else its name."""
    try:
        return run.resolve().relative_to(BENCH_RUNS.resolve()).as_posix()
    except ValueError:
        return run.name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("oct")
    ap.add_argument("mri")
    ap.add_argument("--run", type=Path, required=True, help="the stored run")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    code = code_id()
    stored = (a.run / "code.txt").read_text(encoding="utf-8").strip() if (a.run / "code.txt").is_file() else None
    res = json.loads((a.run / "result.json").read_text(encoding="utf-8"))
    version, inp = res["octreg_version"], res["inputs"]
    if version != __version__ or stored not in (None, code):
        sys.exit(f"{a.run}: octreg {version}, code {stored}, not {__version__} with code {code}")
    same_files = [Path(inp[k]["path"]).name for k in ("oct", "mri")] == [Path(a.oct).name, Path(a.mri).name]
    plain = all(inp[k] is None for k in ("oct_mask", "mri_mask", "oct_spacing_um"))
    if not (same_files and plain and Params.from_dict(res["params"]) == Params()):
        sys.exit(f"{a.run}: not made at the defaults on {Path(a.oct).name} and {Path(a.mri).name}")
    register(a.oct, a.mri, a.out, device=a.device)
    t0, t1 = (np.loadtxt(p / "T_oct2mri.txt") for p in (a.run, a.out))
    same = (a.run / "T_oct2mri.txt").read_bytes() == (a.out / "T_oct2mri.txt").read_bytes()
    rows = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.is_file() else {}
    k = key(a.run)
    rows[k] = {"version": version, "code": code, "stored_code": stored,
               "max_abs_difference": float(np.abs(t1 - t0).max()), "identical": same,
               "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(a.device) if a.device.startswith("cuda") else None, "os": platform.platform()}
    RESULTS.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"{k}: {'identical' if same else 'differs by ' + str(rows[k]['max_abs_difference'])} -> {RESULTS}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run octreg once more on a pair and compare its affine with a stored run of the same octreg code.

    python bench/repeat.py OCT MRI --run STORED -o OUT [--device cuda]

OUT gets the outputs of octreg register at its defaults. bench/results/repeat.json records, under the path of STORED inside
<project root>/bench_runs (else its name), the octreg version and code, the largest absolute difference between the two
T_oct2mri.txt and whether the two files are identical. A stored run of another octreg version, or whose code.txt names
another octreg code, is refused.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.ablate import code_id  # noqa: E402
from bench.paths import BENCH_RUNS  # noqa: E402
from octreg import __version__  # noqa: E402
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
    version = json.loads((a.run / "result.json").read_text(encoding="utf-8"))["octreg_version"]
    if version != __version__ or stored not in (None, code):
        sys.exit(f"{a.run}: octreg {version}, code {stored}, not {__version__} with code {code}")
    register(a.oct, a.mri, a.out, device=a.device)
    t0, t1 = (np.loadtxt(p / "T_oct2mri.txt") for p in (a.run, a.out))
    same = (a.run / "T_oct2mri.txt").read_bytes() == (a.out / "T_oct2mri.txt").read_bytes()
    rows = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.is_file() else {}
    k = key(a.run)
    rows[k] = {"version": version, "code": code, "stored_code": stored,
               "max_abs_difference": float(np.abs(t1 - t0).max()), "identical": same}
    RESULTS.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"{k}: {'identical' if same else 'differs by ' + str(rows[k]['max_abs_difference'])} -> {RESULTS}")


if __name__ == "__main__":
    main()

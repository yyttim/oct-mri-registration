#!/usr/bin/env python3
"""octreg on the DANDI:000874 crops (bench/dandi874/README.md), one run at a time, each scored against the authors' reference.

Runs, in this order: the C3 crops on the 5 deg FLASH (the main condition), the C3 k = 0 crop on the eight other chunk-2 images,
the C6 and C12 crops, and the whole chunk-2 MRI (outside octreg's input assumption). Every run is
`octreg register OCT MRI -o RUN` with the default parameters and no masks. A run whose result.json is complete (it holds the run
time) is not run again. If any run fails, the script names it and stops before writing the stored results.
-> <bench_runs>/SP002/runs/<name>/ and bench/results/SP002/octreg.json (one entry per run)

    python bench/dandi874/run.py [--only NAME ...]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BLOCKS, CONDITIONS, CONTRASTS, REPO, RESULTS, RUNS, SEEDS, crop_file, crop_name, error, mri_file,  # noqa: E402
                    oct_nifti)


def jobs():
    j = [(crop_name(b, "flip2", "C3", k), b, crop_file(crop_name(b, "flip2", "C3", k))) for k in range(SEEDS) for b in BLOCKS]
    j += [(crop_name(b, c, "C3", 0), b, crop_file(crop_name(b, c, "C3", 0))) for c in CONTRASTS if c != "flip2" for b in BLOCKS]
    j += [(crop_name(b, "flip2", cond, k), b, crop_file(crop_name(b, "flip2", cond, k)))
          for cond in CONDITIONS if cond != "C3" for k in range(SEEDS) for b in BLOCKS]
    j += [(f"s{b}_flip2_WHOLE", b, mri_file("flip2")) for b in BLOCKS]
    return j


def done(run):
    try:
        return "seconds" in json.loads((run / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False


def entry(name, b, run):
    res = json.loads((run / "result.json").read_text(encoding="utf-8"))
    parts = name.split("_")
    return {"name": name, "block": b, "contrast": parts[1], "cond": parts[2], "k": int(parts[3][1:]) if len(parts) > 3 else None,
            **error(b, np.loadtxt(run / "T_oct2mri.txt")),
            **{k: res["pose"][k] for k in ("S", "S_class", "S_outline", "polarity")},
            "flags": res["flags"], "deformation": res["deform"]["status"], "seconds": round(res["seconds"]["total"], 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", help="run only these names")
    a = ap.parse_args()
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    out, failed = [], []
    for name, b, mri in jobs():
        if a.only and name not in a.only:
            continue
        run = RUNS / name
        if not done(run):
            run.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            with open(RUNS / f"{name}.log", "w") as log:
                rc = subprocess.run([sys.executable, "-m", "octreg", "register", str(oct_nifti(b)), str(mri), "-o", str(run)],
                                    stdout=log, stderr=subprocess.STDOUT, env=env).returncode
            print(f"{name}: rc {rc}, {time.time() - t0:.0f} s", flush=True)
            if rc != 0 or not done(run):
                failed.append(name)
                continue
        out.append(entry(name, b, run))
        print(f"  {name}: mean {out[-1]['mean_mm']:.3f} mm, rotation {out[-1]['rotation_deg']:.1f} deg, {out[-1]['tier']}", flush=True)
    if failed:
        sys.exit(f"{len(failed)} runs failed (logs in {RUNS}): " + ", ".join(failed))
    if not a.only:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "octreg.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""octreg on the I46 crops (bench/i46/README.md), and the scores of the runs.

run      octreg register at the default parameters on the released OCT and the C3 crops of the 10, 20 and 30 deg FLASH,
         the C6 and C12 crops of the 10 deg FLASH, and the C3 crops of the 10 deg FLASH with the test's specimen mask as
         --oct-mask (<crop>_mask). A run already made by the checkout's octreg code is skipped. Then collect.
collect  scores the final affine and the search pose of every run against the reference and with the vessel check. It
         stops before writing if a run is missing, incomplete or made by other code.
-> <project root>/bench_runs/I46/runs/<name>/ and bench/results/I46/octreg.json (one entry per run)

    python bench/i46/run.py [run] [--only NAME ...]
    python bench/i46/run.py collect [--runs DIR] [-o FILE]
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
from common import (CONDITIONS, FLIPS, MASK, OCT, PRIMARY, REPO, RESULTS, RUNS, SEEDS, crop_file, crop_name,  # noqa: E402
                    error, vessel_score)


def jobs():
    """(name, crop, with --oct-mask) of every run, in the order they run."""
    j = [(crop_name(im, "C3", k), False) for im in FLIPS for k in range(SEEDS)]
    j += [(crop_name(PRIMARY, c, k), False) for c in CONDITIONS if c != "C3" for k in range(SEEDS)]
    j += [(crop_name(PRIMARY, "C3", k), True) for k in range(SEEDS)]
    return [(f"{crop}_mask" if masked else crop, crop, masked) for crop, masked in j]


def done(run):
    """Whether the run was finished by the checkout's octreg code."""
    from bench.ablate import code_id, made_by
    return made_by(run) == code_id()


def entry(name, crop, masked, run):
    res = json.loads((run / "result.json").read_text(encoding="utf-8"))
    pose = res["pose"]
    image, cond, k = crop.split("_")
    T, T4 = np.loadtxt(run / "T_oct2mri.txt"), np.array(pose["T_search"], float)
    s4 = {**error(T4), **vessel_score(T4)}
    return {"name": name, "crop": crop, "image": image, "cond": cond, "k": int(k[1:]), "oct_mask": masked,
            "version": res["octreg_version"], "code": (run / "code.txt").read_text(encoding="utf-8").strip(), **error(T),
            **vessel_score(T),
            "search": {x: s4[x] for x in ("mean_mm", "rotation_deg", "tier", "vessel_median_mm")},
            **{x: pose[x] for x in ("search_rank", "search_F", "search_O", "L", "outline_used")},
            "outside_fraction": {x: res["foreground"][x]["outside_fraction"] for x in ("oct", "mri")},
            "seconds": round(res["seconds"]["total"], 1)}


def collect(runs, out):
    missing = [name for name, _, _ in jobs() if not done(runs / name)]
    if missing:
        sys.exit(f"{len(missing)} runs missing, incomplete or made by other code in {runs}: " + ", ".join(missing))
    rows = []
    for name, crop, masked in jobs():
        rows.append(entry(name, crop, masked, runs / name))
        r = rows[-1]
        print(f"{name}: mean {r['mean_mm']:.3f} mm, rotation {r['rotation_deg']:.1f} deg, {r['tier']} "
              f"(search pose {r['search']['mean_mm']:.3f} mm), vessels {r['vessel_median_mm']}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(len(rows), "runs ->", out)


def run(only):
    from bench.ablate import code_id
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    RUNS.mkdir(parents=True, exist_ok=True)
    failed = []
    for name, crop, masked in jobs():
        if only and name not in only:
            continue
        out = RUNS / name
        if done(out):
            continue
        (out / "code.txt").unlink(missing_ok=True)
        cmd = [sys.executable, "-m", "octreg", "register", str(OCT), str(crop_file(crop))]
        cmd += ["--oct-mask", str(MASK)] if masked else []
        t0 = time.time()
        with open(RUNS / f"{name}.log", "w") as log:
            rc = subprocess.run(cmd + ["-o", str(out)], stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(REPO)).returncode
        print(f"{name}: rc {rc}, {time.time() - t0:.0f} s", flush=True)
        try:
            ok = rc == 0 and "seconds" in json.loads((out / "result.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            ok = False
        if ok:
            (out / "code.txt").write_text(code_id() + "\n", encoding="utf-8")
        else:
            failed.append(name)
    if failed:
        sys.exit(f"{len(failed)} runs failed (logs in {RUNS}): " + ", ".join(failed))
    if not only:
        collect(RUNS, RESULTS / "octreg.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", nargs="?", choices=["run", "collect"], default="run")
    ap.add_argument("--only", nargs="+", help="run: only these names (and no collect)")
    ap.add_argument("--runs", type=Path, default=RUNS, help="collect: the folder of runs "
                                                               "(default <project root>/bench_runs/I46/runs)")
    ap.add_argument("-o", "--out", type=Path, default=RESULTS / "octreg.json", help="collect: the JSON file to write")
    a = ap.parse_args()
    if a.step == "collect":
        collect(a.runs, a.out)
    else:
        run(a.only)


if __name__ == "__main__":
    main()

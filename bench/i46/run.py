#!/usr/bin/env python3
"""octreg on the I46 crops (bench/i46/README.md), one run at a time, and the stored scores of a folder of runs.

run      Runs, in this order, with the default parameters: the C3 crops of the 10, 20 and 30 deg FLASH, then the C6 and C12
         crops of the 10 deg FLASH, each as `octreg register OCT CROP -o RUN` on the released OCT (prepare.py, oct.nii.gz).
         Then the C3 crops of the 10 deg FLASH with the test's specimen mask (common.MASK) as --oct-mask, named
         <crop>_mask. A run whose result.json is complete (it holds the run time) and comes from this octreg version is not
         run again. Then collect.
collect  Scores every run of that list in the runs folder against the reference (common.error and common.vessel_score): the
         final affine T_oct2mri.txt, and in "search" the search pose it was refined from (pose.T_search of result.json).
         "deform" holds the vessel check of the affine with and without the run's deformation (common.vessel_field) and the
         median displacement of the last pass of §5. Each entry also holds, from its result.json, the version of the run, the
         search rank and the search scores F and O of the chosen candidate, its L after §4, whether the outline was used, the
         outside fraction of each image and the run time. If a run is missing or incomplete, the script names it and stops
         before writing.
-> <bench_runs>/I46/runs/<name>/ and bench/results/I46/octreg.json (one entry per run)

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
                    error, vessel_field, vessel_score)


def jobs():
    """(name, crop, with --oct-mask) of every run, in the order they run."""
    j = [(crop_name(im, "C3", k), False) for im in FLIPS for k in range(SEEDS)]
    j += [(crop_name(PRIMARY, c, k), False) for c in CONDITIONS if c != "C3" for k in range(SEEDS)]
    j += [(crop_name(PRIMARY, "C3", k), True) for k in range(SEEDS)]
    return [(f"{crop}_mask" if masked else crop, crop, masked) for crop, masked in j]


def done(run, version=None):
    """Whether run/result.json is complete (it holds the run time) and, if version is given, from that octreg version."""
    try:
        res = json.loads((run / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return "seconds" in res and version in (None, res.get("octreg_version"))


def entry(name, crop, masked, run):
    from octreg import io
    res = json.loads((run / "result.json").read_text(encoding="utf-8"))
    pose = res["pose"]
    image, cond, k = crop.split("_")
    T, T4 = np.loadtxt(run / "T_oct2mri.txt"), np.array(pose["T_search"], float)
    s4 = {**error(T4), **vessel_score(T4)}
    d = res["deform"]
    deform = None if d is None else {**vessel_field(T, *io.load_field(run / "oct2mri_warp.nii.gz")),
                                     "displacement_median_mm": d["passes"][-1]["displacement_median_mm"]}
    return {"name": name, "crop": crop, "image": image, "cond": cond, "k": int(k[1:]), "oct_mask": masked,
            "version": res["octreg_version"], **error(T), **vessel_score(T),
            "search": {x: s4[x] for x in ("mean_mm", "rotation_deg", "tier", "vessel_median_mm")},
            **{x: pose[x] for x in ("search_rank", "search_F", "search_O", "L", "outline_used")}, "deform": deform,
            "outside_fraction": {x: res["foreground"][x]["outside_fraction"] for x in ("oct", "mri")},
            "seconds": round(res["seconds"]["total"], 1)}


def collect(runs, out):
    missing = [name for name, _, _ in jobs() if not done(runs / name)]
    if missing:
        sys.exit(f"{len(missing)} runs missing or incomplete in {runs}: " + ", ".join(missing))
    rows = []
    for name, crop, masked in jobs():
        rows.append(entry(name, crop, masked, runs / name))
        r = rows[-1]
        print(f"{name}: mean {r['mean_mm']:.3f} mm, rotation {r['rotation_deg']:.1f} deg, {r['tier']} "
              f"(search pose {r['search']['mean_mm']:.3f} mm), vessels {r['vessel_median_mm']}"
              + (f", with §5 {r['deform']['field_median_mm']}" if r["deform"] else ""), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(len(rows), "runs ->", out)


def run(only):
    from octreg import __version__
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    RUNS.mkdir(parents=True, exist_ok=True)
    failed = []
    for name, crop, masked in jobs():
        if only and name not in only:
            continue
        out = RUNS / name
        if done(out, __version__):
            continue
        cmd = [sys.executable, "-m", "octreg", "register", str(OCT), str(crop_file(crop))]
        cmd += ["--oct-mask", str(MASK)] if masked else []
        t0 = time.time()
        with open(RUNS / f"{name}.log", "w") as log:
            rc = subprocess.run(cmd + ["-o", str(out)], stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(REPO)).returncode
        print(f"{name}: rc {rc}, {time.time() - t0:.0f} s", flush=True)
        if rc != 0 or not done(out, __version__):
            failed.append(name)
    if failed:
        sys.exit(f"{len(failed)} runs failed (logs in {RUNS}): " + ", ".join(failed))
    if not only:
        collect(RUNS, RESULTS / "octreg.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", nargs="?", choices=["run", "collect"], default="run")
    ap.add_argument("--only", nargs="+", help="run: only these names (and no collect)")
    ap.add_argument("--runs", type=Path, default=RUNS, help="collect: the folder of runs (default <bench_runs>/I46/runs)")
    ap.add_argument("-o", "--out", type=Path, default=RESULTS / "octreg.json", help="collect: the JSON file to write")
    a = ap.parse_args()
    if a.step == "collect":
        collect(a.runs, a.out)
    else:
        run(a.only)


if __name__ == "__main__":
    main()

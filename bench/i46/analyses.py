#!/usr/bin/env python3
"""The I46 analyses beyond the main runs (bench/i46/README.md): the ablation and the start test. Every run uses the checkout's
octreg and bench/ablate.py on the released OCT with octreg's own masks, as the main runs of run.py.

ablate   the variants of bench/ablate.py in VARIANTS on the C3 crops k = 0..9 of the 10 deg FLASH, into
         <bench_runs>/I46/ablation/<crop>/<variant>/. The outline and deformation variants are shown on I58 only.
starts   the method on flip1_C3_k0 from the OCT turned by 90 and 180 deg about each of its array axes and by 10 rotations drawn
         uniformly (scipy Rotation.random, seed 4646), about the centre of the OCT image box
         (bench/ablate.py --only base --turn), into <bench_runs>/I46/starts/<start>/base/
collect  scores every result against the reference: common.error and common.vessel_score of the affine, and for the ablation
         common.vessel_field of the deformation -> bench/results/I46/ablation.json and starts.json. The start of a turned
         run is bench/ablate.py header_start of the turned OCT and the crop, composed with the turn.

    python bench/i46/analyses.py ablate | starts | collect
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
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (OCT, PRIMARY, REPO, RESULTS, SEEDS, WORK, crop_file, crop_name, error, vessel_field,  # noqa: E402
                    vessel_score)

ABL, STARTS = WORK / "ablation", WORK / "starts"
VARIANTS = ["no_search", "no_refine", "first_choice", "no_prior", "signed", "signed_neg", "tracefree", "search_whole_mask"]
START_CROP = crop_name(PRIMARY, "C3", 0)
SEED, N_RANDOM = 4646, 10


def starts():
    """{name: rotation vector in degrees} of the start test."""
    s = {f"{a}{d}": [d * (i == k) for i in range(3)] for k, a in enumerate("ijk") for d in (90, 180)}
    for n, r in enumerate(Rotation.random(N_RANDOM, random_state=SEED).as_rotvec(degrees=True)):
        s[f"rand{n:02d}"] = r.round(6).tolist()
    return s


def ablate_one(crop, out, only, extra=()):
    cmd = [sys.executable, str(REPO / "bench" / "ablate.py"), str(OCT), str(crop_file(crop)), "-o", str(out), "--only", only,
           *extra]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out / f"{only}.log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONPATH": str(REPO)},
                            cwd=str(REPO)).returncode
    print(f"{out.name} {only}: rc {rc}, {time.time() - t0:.0f} s", flush=True)


def scored(run, deform=False):
    """Scores of the affine of a run directory and, with deform, of its deformation."""
    from octreg import io
    T = np.loadtxt(run / "T_oct2mri.txt")
    row = {**error(T), **vessel_score(T)}
    if deform:
        f = run / "oct2mri_warp.nii.gz"
        row["deform"] = vessel_field(T, *io.load_field(f)) if f.is_file() else None
    return row


def turned_start(turn):
    """The start of a turned run as a pose of the original OCT world (module docstring)."""
    from bench.ablate import header_start
    from octreg import io
    vo, vm = io.load_volume(OCT), io.load_volume(crop_file(START_CROP))
    return header_start(vm.affine, vm.shape, turn @ vo.affine, vo.shape) @ turn


def collect():
    rows = []
    for k in range(SEEDS):
        crop = crop_name(PRIMARY, "C3", k)
        for v in VARIANTS:
            run = ABL / crop / v
            if not (run / "variant.json").is_file():
                print("missing", run)
                continue
            rows.append({"crop": crop, "variant": v, **scored(run, deform=True)})
    (RESULTS / "ablation.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(len(rows), "ablation rows")
    st = []
    for name, rv in starts().items():
        run = STARTS / name / "base"
        if not (run / "variant.json").is_file():
            print("missing", run)
            continue
        turn = np.array(json.loads((run / "variant.json").read_text(encoding="utf-8"))["turn"])
        st.append({"start": name, "turn_deg": rv, "start_pose": error(turned_start(turn)), **scored(run)})
    (RESULTS / "starts.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
    print(len(st), "start rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["ablate", "starts", "collect"])
    a = ap.parse_args()
    if a.step == "ablate":
        for v in VARIANTS:
            for k in range(SEEDS):
                crop = crop_name(PRIMARY, "C3", k)
                if not (ABL / crop / v / "variant.json").is_file():
                    ablate_one(crop, ABL / crop, v)
    elif a.step == "starts":
        for name, rv in starts().items():
            if not (STARTS / name / "base" / "variant.json").is_file():
                ablate_one(START_CROP, STARTS / name, "base", ["--turn=" + ",".join(map(str, rv))])   # a value may start with -
    else:
        collect()


if __name__ == "__main__":
    main()

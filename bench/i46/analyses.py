#!/usr/bin/env python3
"""The I46 analyses beyond the main runs (bench/i46/README.md): the ablation and the start test. Every run uses the checkout's
octreg and bench/ablate.py, the masked OCT given as its own specimen mask (--oct-mask).

ablate   the variants of bench/ablate.py in VARIANTS on the C3 crops k = 0..9 of the 10 deg FLASH, into
         <bench_runs>/I46/ablation/<crop>/<variant>/. The outline variants are left out: the OCT is given as tissue only
         and the MRI of these crops is all tissue, so the outline is never used. The deformation variants are shown on I58.
starts   the method on flip1_C3_k0 from the OCT turned by 90 and 180 deg about each of its array axes and by 10 rotations drawn
         uniformly (scipy Rotation.random, seed 4646), about the centre of the OCT image box
         (bench/ablate.py --only base --turn), into <bench_runs>/I46/starts/<start>/base/
collect  scores every result against the reference: common.error and common.vessel_score of the affine, and for the ablation
         common.vessel_field of the deformation -> bench/results/I46/ablation.json and starts.json

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
VARIANTS = ["no_search", "no_refine", "first_choice", "no_prior", "signed", "signed_neg", "full_tensor"]
START_CROP = crop_name(PRIMARY, "C3", 0)
SEED, N_RANDOM = 4646, 10


def starts():
    """{name: rotation vector in degrees} of the start test."""
    s = {f"{a}{d}": [d * (i == k) for i in range(3)] for k, a in enumerate("ijk") for d in (90, 180)}
    for n, r in enumerate(Rotation.random(N_RANDOM, random_state=SEED).as_rotvec(degrees=True)):
        s[f"rand{n:02d}"] = r.round(6).tolist()
    return s


def ablate_one(crop, out, only, extra=()):
    cmd = [sys.executable, str(REPO / "bench" / "ablate.py"), str(OCT), str(crop_file(crop)), "-o", str(out), "--oct-mask",
           str(OCT), "--only", only, *extra]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out / f"{only}.log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONPATH": str(REPO)},
                            cwd=str(REPO)).returncode
    print(f"{out.name} {only}: rc {rc}, {time.time() - t0:.0f} s", flush=True)


def scored(run, T=None):
    """Scores of a run directory (its affine, or T) and of its deformation."""
    from octreg import io
    T = np.loadtxt(run / "T_oct2mri.txt") if T is None else np.asarray(T, float)
    f = run / "oct2mri_warp.nii.gz"
    return {**error(T), **vessel_score(T), "deform": vessel_field(T, *io.load_field(f)) if f.is_file() else None}


def collect():
    rows = []
    for k in range(SEEDS):
        crop = crop_name(PRIMARY, "C3", k)
        for v in VARIANTS:
            run = ABL / crop / v
            if not (run / "variant.json").is_file():
                print("missing", run)
                continue
            rows.append({"crop": crop, "variant": v, **scored(run)})
    (RESULTS / "ablation.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(len(rows), "ablation rows")
    st = []
    for name, rv in starts().items():
        run = STARTS / name / "base"
        if not (run / "variant.json").is_file():
            print("missing", run)
            continue
        turn = np.array(json.loads((run / "variant.json").read_text(encoding="utf-8"))["turn"])
        st.append({"start": name, "turn_deg": rv, "start_pose": error(_centres(turn)),
                   **{x: v for x, v in scored(run).items() if x != "deform"}})
    (RESULTS / "starts.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
    print(len(st), "start rows")


def _centres(turn):
    """The start of a turned run as a pose of the original OCT world: the header orientation of the turned OCT with the centres
    of the two image boxes aligned, composed with the turn."""
    from octreg import geometry as G, io
    vo, vm = io.load_volume(str(OCT)), io.load_volume(str(crop_file(START_CROP)))
    T = np.eye(4)
    T[:3, 3] = G.box(vm.affine, vm.shape)[0] - G.box(turn @ vo.affine, vo.shape)[0]
    return T @ turn


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

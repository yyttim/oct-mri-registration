#!/usr/bin/env python3
"""The I46 ablation and start test (bench/i46/README.md), through bench/ablate.py on the released OCT with octreg's own masks.

ablate   the variants of bench/ablate.py except the outline and deformation ones (shown on I58 only), on the C3 crops
         k = 0..9 of the 10 deg FLASH, into <bench_runs>/I46/ablation/<crop>/<variant>/
starts   the method on flip1_C3_k0 from the OCT turned about the centre of its image box by 90 and 180 deg about each array
         axis and by 10 uniform random rotations (seed 4646), into <bench_runs>/I46/starts/<start>/base/
collect  scores the affine of every run against the reference -> bench/results/I46/ablation.json and starts.json. It stops
         and writes nothing if a run was not made by the checkout's octreg code.

ablate and starts skip the runs already made by the checkout's octreg code and exit with an error listing the runs that
failed.

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
from common import OCT, PRIMARY, REPO, RESULTS, SEEDS, WORK, crop_file, crop_name, error, vessel_score  # noqa: E402

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
    """bench/ablate.py --only `only` on a crop into out/<only>. -> whether it finished with the checkout's octreg code."""
    from bench.ablate import code_id, made_by
    cmd = [sys.executable, str(REPO / "bench" / "ablate.py"), str(OCT), str(crop_file(crop)), "-o", str(out), "--only", only,
           *extra]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out / f"{only}.log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONPATH": str(REPO)},
                            cwd=str(REPO)).returncode
    print(f"{out.name} {only}: rc {rc}, {time.time() - t0:.0f} s", flush=True)
    return rc == 0 and made_by(out / only) == code_id()


def scored(run):
    T = np.loadtxt(run / "T_oct2mri.txt")
    return {**error(T), **vessel_score(T)}


def turned_start(turn):
    """Start of a turned run in the original OCT world: header_start of the turned OCT and the crop, composed with the turn."""
    from bench.ablate import header_start
    from octreg import io
    vo, vm = io.load_volume(OCT), io.load_volume(crop_file(START_CROP))
    return header_start(vm.affine, vm.shape, turn @ vo.affine, vo.shape) @ turn


def collect():
    from bench.ablate import code_id, made_by
    from octreg import __version__
    code = code_id()
    abl = [(crop_name(PRIMARY, "C3", k), v) for k in range(SEEDS) for v in VARIANTS]
    st = list(starts().items())
    runs = [ABL / crop / v for crop, v in abl] + [STARTS / name / "base" for name, _ in st]
    other = [str(run) for run in runs if made_by(run) != code]
    if other:
        sys.exit(f"{len(other)} runs missing or not made by the octreg code {code}: " + ", ".join(other))
    rows = [{"crop": crop, "variant": v, "version": __version__, "code": code, **scored(ABL / crop / v)} for crop, v in abl]
    (RESULTS / "ablation.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(len(rows), "ablation rows")
    rows = []
    for name, rv in st:
        run = STARTS / name / "base"
        turn = np.array(json.loads((run / "variant.json").read_text(encoding="utf-8"))["turn"])
        rows.append({"start": name, "turn_deg": rv, "version": __version__, "code": code,
                     "start_pose": error(turned_start(turn)), **scored(run)})
    (RESULTS / "starts.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(len(rows), "start rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["ablate", "starts", "collect"])
    a = ap.parse_args()
    from bench.ablate import code_id, made_by
    code, failed = code_id(), []
    if a.step == "ablate":
        for v in VARIANTS:
            for k in range(SEEDS):
                crop = crop_name(PRIMARY, "C3", k)
                if made_by(ABL / crop / v) != code and not ablate_one(crop, ABL / crop, v):
                    failed.append(f"{crop}/{v}")
    elif a.step == "starts":
        for name, rv in starts().items():
            if made_by(STARTS / name / "base") != code and not ablate_one(
                    START_CROP, STARTS / name, "base", ["--turn=" + ",".join(map(str, rv))]):   # a value may start with -
                failed.append(name)
    else:
        collect()
    if failed:
        sys.exit(f"{len(failed)} runs failed (see their logs): " + ", ".join(failed))


if __name__ == "__main__":
    main()

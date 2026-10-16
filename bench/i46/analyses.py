#!/usr/bin/env python3
"""The analyses of the I46 test beyond the main runs (bench/i46/README.md): the ablation, the start tests and the design
choices of the method. Every run uses the checkout's octreg with its default parameters unless a change is named.

ablate     bench/ablate.py on the crops of the test, the masked OCT given as its own specimen mask (--oct-mask), into
           <bench_runs>/I46/ablation/<crop>/:
             the C3 crops k = 0..9 of the 10 deg FLASH   A4, A6p, A9, A10, A13 and A14 (A12 is the stored §4 pose of base)
             the C3 crops of the 20 and 30 deg FLASH and the C12 crops of the 10 deg FLASH   A4
             flip1_C3_k0, the crop with the median distance   --starts and --random-starts 20 --seed 4646: the method and
                                                               A14 from 6 turned and 20 random starts
choices    the method with one design choice replaced at a time, into <bench_runs>/I46/attr/<choice>/<run>/:
             outline   `octreg register --params` with outline_min_outside 0: the outline used whatever the outside, on all 60
                       runs
             valley    medium_fraction 1.0: every deep MRI valley taken as the tissue edge, on the runs where the method passes
                       one over (elsewhere the masks are the same)
             nms       nms_deg 10: candidates within 10 deg count as one, on all 60 runs
             score     §5 scored by the weighted mean of (n_O . n_M)^2 (Haber and Modersitzki), from the
                       stored §4 pose of the run (§1 to §4 do not depend on it), on all 60 runs
             topk48    48 poses refined instead of 24, on the runs whose result came from the 18th to 24th of the 24 search
                       candidates (search_rank 17 to 23, counted from 0)
collect    scores every result against the reference (common.error) -> bench/results/I46/{ablation,starts,choices}.json
near       A14 (§4 and §5 without the orientation search) from the near start H_k of baselines.py, 10 deg and 1 mm from R, on
           the C3 crops k = 0..9 of the 10 deg FLASH: the base grids of the crop (bench/ablate.py prepare, the masked OCT as
           --oct-mask), then ablate.solve(..., search=False, T0=H_k), §4 from H_k at both polarities and §5 on the best pose,
           into <bench_runs>/I46/near/<crop>/A14/. The start is the one the tools get in their moving header. The package
           and bench/ablate.py are those of --checkout (its bench/ablate.py puts that package first on the path), by default
           this repository. Finished runs are kept. Every run is scored against the reference with the distance of its
           start -> bench/results/I46/near.json

    python bench/i46/analyses.py ablate | choices
    python bench/i46/analyses.py collect [--ablation DIR] [--attr DIR]
    python bench/i46/analyses.py near [--checkout DIR] [--device cuda]
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
from common import (CONDITIONS, FLIPS, OCT, PRIMARY, REPO, RESULTS, SEEDS, WORK, crop_file, crop_name, error,  # noqa: E402
                    vessel_score)

ABL, ATTR, NEAR = WORK / "ablation", WORK / "attr", WORK / "near"
SEED, N_RANDOM = 4646, 20
MEDIAN_CROP = crop_name(PRIMARY, "C3", 0)
PRIMARY_VARIANTS = ["A4", "A6p", "A9", "A10", "A13", "A14"]
PARAMS = {"outline": {"outline_min_outside": 0.0}, "valley": {"medium_fraction": 1.0}, "nms": {"nms_deg": 10.0},
          "topk48": {"topk": 48}}


def ablation_jobs():
    j = [(crop_name(PRIMARY, "C3", k), PRIMARY_VARIANTS) for k in range(SEEDS)]
    j += [(crop_name(im, "C3", k), ["A4"]) for im in FLIPS if im != PRIMARY for k in range(SEEDS)]
    j += [(crop_name(PRIMARY, c, k), ["A4"]) for c in CONDITIONS if c not in ("C3", "C6") for k in range(SEEDS)]
    return j


def ablate():
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    for crop, variants in ablation_jobs():
        out = ABL / crop
        extra = ["--starts", "--random-starts", str(N_RANDOM), "--seed", str(SEED)] if crop == MEDIAN_CROP else []
        cmd = [sys.executable, str(REPO / "bench/ablate.py"), "--out", str(out), "--oct", str(OCT),
               "--mri", str(crop_file(crop)), "--oct-mask", str(OCT), "--pair", f"DANDI:000026 I46 {crop}",
               "--only", ",".join(variants), *extra]
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(ABL / f"{crop}.log", "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(REPO)).returncode
        print(f"{crop}: rc {rc}, {time.time() - t0:.0f} s", flush=True)


def octreg_rows(path):
    return {r["name"]: r for r in json.loads(Path(path).read_text(encoding="utf-8"))}


def choice_jobs():
    rows = octreg_rows(RESULTS / "octreg.json")
    runs = sorted(rows)
    valley = [n for n in runs if "mri_tissue_valley_passed" in rows[n]["flags"]]
    late = [n for n in runs if rows[n].get("search_rank") is not None and 17 <= rows[n]["search_rank"] <= 23]
    return ([("score", n) for n in runs] + [("outline", n) for n in runs] + [("nms", n) for n in runs]
            + [("valley", n) for n in valley] + [("topk48", n) for n in late])


def score_hm(self, T):
    """§5 scored as Haber and Modersitzki do: the w-weighted mean of (n_O . n_M)^2, in place of NGFGrid.F."""
    import torch
    from octreg import geometry as G
    Ti = torch.linalg.inv(T)
    x = G.apply_affine(Ti, self.pts)
    g = Ti[:3, :3].T @ G.sample_world(self.gO, self.A_O, x)
    w = G.sample_world(self.w, self.A_O, x)[0]
    n = g / torch.sqrt((g * g).sum(0) + self.eps_O ** 2)
    return (w * ((n * self.nM).sum(0)) ** 2).sum() / w.sum().clamp(min=1.0)


def score_choice(run, runs):
    """§5 scored as Haber and Modersitzki do, from the stored §4 pose of runs/<run>, on the base grids of bench/ablate.py."""
    sys.path.insert(0, str(REPO / "bench"))
    sys.path.insert(0, str(REPO))
    import ablate as A
    from octreg import io, ngf
    from octreg.params import Params
    nomask = run.endswith("_nomask")
    A.E.OCT, A.E.MRI, A.MASK_FILE = OCT, crop_file(run.removesuffix("_nomask")), None if nomask else OCT
    d, src = ATTR / "score" / run, "texture" if nomask else "file"
    A.prepare(d, [src])
    with np.load(d / "prep" / src / "oct_h.npz") as z, np.load(d / "prep" / "mri" / "mri_h.npz") as y:
        o, m = {k: z[k] for k in z.files}, {k: y[k] for k in y.files}
    main = json.loads((runs / run / "result.json").read_text(encoding="utf-8"))
    T4 = np.array(main["pose"]["T_before_ngf"], float)
    t0 = time.time()
    F21, ngf.NGFGrid.F = ngf.NGFGrid.F, score_hm
    try:
        T, info = ngf.refine_ngf((m["arr"], m["mask"], m["affine"]), (o["arr"], o["mask"], o["affine"]), T4, Params(), "cuda")
    finally:
        ngf.NGFGrid.F = F21
    io.write_transform_txt(T, d / "T_oct2mri.txt")
    io.write_json({"run": run, "change": "§5 scored as Haber and Modersitzki do, from the run's §4 pose", "T": T, "T_before_ngf": T4,
                   "ngf": info, "seconds": time.time() - t0}, d / "result.json")


def finished(d):
    """A run is done when its result.json, written after T_oct2mri.txt, parses and holds the run time."""
    try:
        return "seconds" in json.loads((d / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False


def choices(runs):
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    for change, run in choice_jobs():
        d = ATTR / change / run
        if finished(d):
            continue
        t0 = time.time()
        if change == "score":
            score_choice(run, runs)
        else:
            d.mkdir(parents=True, exist_ok=True)
            pf = ATTR / change / "params.json"
            pf.write_text(json.dumps(PARAMS[change]), encoding="utf-8")
            extra = [] if run.endswith("_nomask") else ["--oct-mask", str(OCT)]
            cmd = [sys.executable, "-m", "octreg", "register", str(OCT), str(crop_file(run.removesuffix("_nomask"))), *extra,
                   "--params", str(pf), "-o", str(d)]
            with open(ATTR / change / f"{run}.log", "w") as log:
                code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(REPO)).returncode
            if code:
                print(f"{change} {run}: octreg register exited with {code}, see {ATTR / change / run}.log", flush=True)
        print(f"{change} {run}: {time.time() - t0:.0f} s", flush=True)


def scored(T):
    T = np.asarray(T, float)
    return {**error(T), **vessel_score(T)}


def collect(abl, attr):
    rows = []
    for crop, variants in ablation_jobs():
        for v in variants + (["A12"] if crop.startswith(f"{PRIMARY}_C3_") else []):
            f = abl / crop / "variants" / ("base" if v == "A12" else v) / "result.json"
            if not f.is_file():
                print("missing", f)
                continue
            r = json.loads(f.read_text(encoding="utf-8"))
            if "T" not in r:
                rows.append({"crop": crop, "variant": v, "error": r.get("error")})
                continue
            T = r["T_before_ngf"] if v == "A12" else r["T"]
            rows.append({"crop": crop, "variant": v, **scored(T), "s4": error(r["T_before_ngf"]),
                         **{k: r["best"][k] for k in ("S", "S_class", "S_outline", "polarity")}})
    (RESULTS / "ablation.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(len(rows), "ablation rows")
    st = []
    for d in sorted((abl / MEDIAN_CROP / "starts").iterdir()):
        if d.is_dir() and (d / "T_start.txt").is_file():
            for kind in ("method", "A14"):
                r = json.loads((d / kind / "result.json").read_text(encoding="utf-8"))
                st.append({"start": d.name, "kind": kind, "start_pose": error(np.loadtxt(d / "T_start.txt")), **scored(r["T"]),
                           "s4": error(r["T_before_ngf"])})
    (RESULTS / "starts.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
    print(len(st), "start rows")
    at = []
    for change, run in choice_jobs():
        f = attr / change / run / "T_oct2mri.txt"
        if not finished(f.parent):
            print("missing", f.parent)
            continue
        res = json.loads((attr / change / run / "result.json").read_text(encoding="utf-8"))
        T4 = res["T_before_ngf"] if "T_before_ngf" in res else res["pose"]["T_before_ngf"]
        at.append({"choice": change, "run": run, **scored(np.loadtxt(f)), "s4": error(T4)})
    (RESULTS / "choices.json").write_text(json.dumps(at, indent=1), encoding="utf-8")
    print(len(at), "design-choice rows")


def near(checkout, device):
    """The near step of the module docstring: the runs not done yet, then near.json."""
    sys.path.insert(0, str(Path(checkout).resolve() / "bench"))
    import ablate as A                          # bench/ablate.py of the checkout, which puts its package first on the path
    import octreg
    import torch
    from octreg import io
    from octreg.params import Params
    import baselines as B
    print("octreg", octreg.__version__, "from", Path(octreg.__file__).parent, flush=True)
    P, rows = Params(), []
    for k in range(SEEDS):
        crop = crop_name(PRIMARY, "C3", k)
        B.near_inputs(crop)
        st = B.start_of(crop)
        Hk, d = np.array(st["H"]), NEAR / crop
        res = d / "A14" / "result.json"
        old = json.loads(res.read_text(encoding="utf-8")) if res.is_file() else {}
        if old.get("params_hash") != P.hash() or old.get("start_T") != st["H"]:
            A.E.OCT, A.E.MRI, A.MASK_FILE = OCT, crop_file(crop), OCT
            A.prepare(d, ["file"])
            with np.load(d / "prep" / "file" / "oct_h.npz") as z, np.load(d / "prep" / "mri" / "mri_h.npz") as y:
                o, m = {n: z[n] for n in z.files}, {n: y[n] for n in y.files}
            t0 = time.time()
            poses, info = A.solve(o, m, P, device, search=False, T0=Hk)
            best = poses[0]
            (d / "A14").mkdir(parents=True, exist_ok=True)
            io.write_transform_txt(best["T"], d / "A14" / "T_oct2mri.txt")
            io.write_json({"crop": crop, "kind": "A14", "version": octreg.__version__, "params_hash": P.hash(),
                           "start_T": st["H"],
                           "T": best["T"], "T_before_ngf": best.get("T_before_ngf", best["T"]), "S": best["S"], "L": best["L"],
                           "polarity": best["polarity"], "ngf": info.get("ngf"), "seconds": time.time() - t0}, res)
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
            print(f"{crop}: A14 in {time.time() - t0:.0f} s", flush=True)
            old = json.loads(res.read_text(encoding="utf-8"))
        rows.append({"crop": crop, "kind": "A14", "version": old["version"], "params_hash": old["params_hash"],
                     "start_pose": error(Hk), "start_from_S": st["start_from_S"], **scored(old["T"]),
                     "s4": error(old["T_before_ngf"]), "polarity": old["polarity"], "seconds": round(old["seconds"], 1)})
    (RESULTS / "near.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(len(rows), "near rows ->", RESULTS / "near.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["ablate", "choices", "collect", "near"])
    ap.add_argument("--ablation", type=Path, default=ABL, help="collect: the folder of the ablation runs")
    ap.add_argument("--attr", type=Path, default=ATTR, help="collect: the folder of the design-choice runs")
    ap.add_argument("--runs", type=Path, default=WORK / "runs", help="the folder of the octreg runs (choices: their §4 poses)")
    ap.add_argument("--checkout", type=Path, default=REPO, help="near: the checkout whose package and bench/ablate.py run")
    ap.add_argument("--device", default="cuda", help="near: the device of §4 and §5")
    a = ap.parse_args()
    if a.step == "ablate":
        ablate()
    elif a.step == "choices":
        choices(a.runs)
    elif a.step == "near":
        near(a.checkout, a.device)
    else:
        collect(a.ablation, a.attr)


if __name__ == "__main__":
    main()

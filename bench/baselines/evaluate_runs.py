#!/usr/bin/env python3
"""Evaluate affine baseline run directories the same way as an octreg run (bench only, not part of the package).

    python bench/baselines/evaluate_runs.py RUN [RUN ...] [--octreg-run DIR] [--shots] [--summary OUT.json] [--force]

For every RUN (a directory made by rundir.py init, with a check.json that is ok):
  1. octreg apply --affine-only: the 20 um OCT through RUN/T_oct2mri.txt onto the MRI grid -> RUN/oct_in_mri_affine.nii.gz.
     Every method goes through the same resampler, so the visual comparison shows the registration and not the
     interpolation.
  2. bench/evaluate.py with octreg's masks (--masks) and the octreg run (--octreg-run, default
     OCTREG_PROJECT_ROOT/bench_runs/I58/octreg) as --previous -> RUN/eval.json: rim outline agreement forward and reverse
     (median and p90 per face class), pose distance to octreg's affine, raw-data frame check.
  3. Dice of the OCT specimen mask through T against the MRI foreground on the 0.15 mm grid -> RUN/dice.json. The MRI holds
     tissue beyond the block, so Dice stays below 1 for every method and only compares methods.
  4. With --shots: freeview screenshots at the shared sections (baselines/shots/shots.sh, run in WSL), named
     <tool>_<variant>.
A summary row per run goes into --summary (default OCTREG_PROJECT_ROOT/baselines/summary_affine.json).
Environment: OCTREG_PROJECT_ROOT, the root of the inputs, runs and screenshots (default: the repository), and
OCTREG_I58_DIR, the folder of the two original files (default OCTREG_PROJECT_ROOT/data/I58).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from octreg import geometry as G  # noqa: E402

ROOT = Path(os.environ.get("OCTREG_PROJECT_ROOT") or REPO)
I58 = Path(os.environ.get("OCTREG_I58_DIR") or ROOT / "data" / "I58")
OCT = I58 / "I58_Brainstem_mus_Slice_full_20um_corr.nii.gz"
MRI = I58 / "I58_brainstem_MRI_cropped_to_OCT.nii.gz"
INPUTS = ROOT / "baselines" / "inputs"
OCTREG_RUN = ROOT / "bench_runs" / "I58" / "octreg"
SHOTS = ROOT / "baselines" / "shots"
PY = sys.executable
ENV = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONIOENCODING": "utf-8", "CUDA_VISIBLE_DEVICES": ""}


def wsl_path(p: Path) -> str:
    s = str(Path(p).resolve()).replace("\\", "/")
    return "/mnt/" + s[0].lower() + s[2:] if len(s) > 1 and s[1] == ":" else s


def run(cmd, log=None):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=ENV, encoding="utf-8", errors="replace")
    if log:
        Path(log).write_text(r.stdout + r.stderr, encoding="utf-8")
    if r.returncode:
        raise RuntimeError(f"{cmd[0]} failed ({r.returncode}): {(r.stdout + r.stderr)[-2000:]}")
    return time.time() - t0


def dice(run_dir: Path):
    T = np.loadtxt(run_dir / "T_oct2mri.txt")
    om, mm = nib.load(INPUTS / "oct_mask.nii.gz"), nib.load(INPUTS / "mri_mask.nii.gz")
    o = np.asarray(om.dataobj) > 0
    moved = G.resample_to(o, om.affine, mm.shape, mm.affine, np.linalg.inv(T), order=0) > 0
    m = np.asarray(mm.dataobj) > 0
    d = {"dice_oct_mask_vs_mri_foreground": float(2 * (moved & m).sum() / max(moved.sum() + m.sum(), 1)),
         "oct_mask_inside_mri_foreground": float((moved & m).sum() / max(moved.sum(), 1)),
         "grid_mm": float(np.linalg.norm(mm.affine[:3, 0]))}
    (run_dir / "dice.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
    return d


def scales(run_dir: Path):
    """Scale of the transform along each OCT array axis (octreg's pose.scale_per_oct_axis): the length of T applied to a unit
    step along each axis of the OCT header, so 1 means the voxel size is kept."""
    T = np.loadtxt(run_dir / "T_oct2mri.txt")
    A = np.array(nib.load(INPUTS / "oct_0.08mm.nii.gz").affine)
    sp = np.linalg.norm(A[:3, :3], axis=0)
    return np.linalg.norm(T[:3, :3] @ (A[:3, :3] / sp), axis=0).round(4).tolist()


def evaluate(run_dir: Path, shots: bool, force: bool, octreg_run: Path = OCTREG_RUN):
    chk = run_dir / "check.json"
    if not chk.is_file() or not json.loads(chk.read_text())["ok"]:
        print(f"{run_dir}: no check.json ok, skipped", flush=True)
        return None
    name = f"{run_dir.parent.name}_{run_dir.name}"
    t_apply = t_eval = None
    if force or not (run_dir / "oct_in_mri_affine.nii.gz").is_file():
        t_apply = run([PY, "-m", "octreg", "apply", "--run", str(run_dir), "--moving", str(OCT), "--reference", str(MRI),
                       "-o", str(run_dir / "oct_in_mri_affine.nii.gz"), "--affine-only"], run_dir / "apply.log")
    if force or not (run_dir / "eval.json").is_file():
        t_eval = run([PY, str(REPO / "bench" / "evaluate.py"), str(run_dir), "--masks", str(INPUTS), "--previous",
                      str(octreg_run), "-o", str(run_dir / "eval.json")], run_dir / "evaluate.log")
    d = dice(run_dir)
    if shots and (force or not (SHOTS / "shots" / f"ax_26_{name}.png").is_file()):
        run(["wsl", "bash", wsl_path(SHOTS / "shots.sh"), name, wsl_path(run_dir / "oct_in_mri_affine.nii.gz")],
            run_dir / "shots.log")
    e = json.loads((run_dir / "eval.json").read_text(encoding="utf-8"))
    b, pv, fc = e.get("boundary", {}), e.get("pose_to_previous", {}), e.get("frame_check", {})
    tm = run_dir / "time.txt"
    row = {"run": name, "tool": run_dir.parent.name, "variant": run_dir.name,
           "wall_s": float(tm.read_text().split()[0]) if tm.is_file() else None,
           "rim_fwd_mm": b.get("rim_median_mm", [None, None])[0], "rim_rev_mm": b.get("rim_median_mm", [None, None])[1],
           "fwd_all_median_mm": b.get("forward_oct_to_mri", {}).get("all", {}).get("median_mm"),
           "fwd_all_p90_mm": b.get("forward_oct_to_mri", {}).get("all", {}).get("p90_mm"),
           "rev_all_median_mm": b.get("reverse_mri_to_oct", {}).get("all", {}).get("median_mm"),
           "rev_all_p90_mm": b.get("reverse_mri_to_oct", {}).get("all", {}).get("p90_mm"),
           "dice": d["dice_oct_mask_vs_mri_foreground"], "oct_inside_mri": d["oct_mask_inside_mri_foreground"],
           "to_octreg_mean_mm": pv.get("mean_mm"), "to_octreg_corners_max_mm": pv.get("corners_max_mm"),
           "to_octreg_rotation_deg": pv.get("rotation_deg"),
           "frame_ok": fc.get("ok"), "frame_spearman": fc.get("spearman_export"),
           "det": float(np.linalg.det(np.loadtxt(run_dir / "T_oct2mri.txt")[:3, :3])),
           "scale_per_oct_axis": scales(run_dir),
           "apply_s": t_apply, "evaluate_s": t_eval}
    f = lambda v, nd=2: "n/a" if v is None else f"{v:.{nd}f}"
    print(f"{name}: rim {f(row['rim_fwd_mm'])} / {f(row['rim_rev_mm'])} mm, dice {f(row['dice'], 3)}, "
          f"to octreg {f(row['to_octreg_mean_mm'])} mm, frame {row['frame_ok']}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--octreg-run", type=Path, default=OCTREG_RUN)
    ap.add_argument("--shots", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--summary", type=Path, default=ROOT / "baselines" / "summary_affine.json")
    a = ap.parse_args()
    rows = json.loads(a.summary.read_text(encoding="utf-8")) if a.summary.is_file() else {}
    for r in a.runs:
        row = evaluate(r, a.shots, a.force, a.octreg_run)
        if row:
            rows = json.loads(a.summary.read_text(encoding="utf-8")) if a.summary.is_file() else {}   # merge with other runs
            rows[row["run"]] = row
            a.summary.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"{len(rows)} rows in {a.summary}")


if __name__ == "__main__":
    main()

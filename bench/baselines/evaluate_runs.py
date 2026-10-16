#!/usr/bin/env python3
"""Method-independent properties of affine baseline runs (bench only, not part of the package).

    python bench/baselines/evaluate_runs.py [RUN ...] [--octreg-run DIR] [--oct-grid OCT] [--shots] [--summary OUT.json] [--force]

For every RUN (a directory made by rundir.py init, with a check.json that is ok):
  1. octreg apply --affine-only: the 20 um OCT through RUN/T_oct2mri.txt onto the MRI grid -> RUN/oct_in_mri_affine.nii.gz.
     Every method goes through the same resampler, so the panels of bench/figures.py show the registration and not the
     interpolation.
  2. The properties of the transform that need no reference: the scale along each OCT array axis (the length of T applied to
     a unit step along that axis of the OCT header, 1 = voxel size kept) and the determinant of its 3x3 (1 = volume kept).
  3. The wall time of the registration, from RUN/time.txt (seconds, written by the tool's own command), and the hardware,
     from RUN/threads.txt.
  4. With --shots: freeview screenshots at the shared sections, named <tool>_<variant>, by
     OCTREG_PROJECT_ROOT/baselines/shots/shots.sh, run in WSL. That script and its list of sections are local files and not
     in the repository.
With --octreg-run, a row "octreg" is added for that octreg run: its T_oct2mri.txt, the sum of its §1 to §5 step times in
result.json and its device. A summary row per run goes into --summary (default
OCTREG_PROJECT_ROOT/baselines/summary_affine.json). No row holds a distance to octreg's affine or a score computed with
octreg's masks or read-outs. The roots come from
bench/paths.py (OCTREG_PROJECT_ROOT, OCTREG_DATA_ROOT, OCTREG_I58_DIR). The inputs, summaries and screenshots of the
comparison are under OCTREG_PROJECT_ROOT/baselines.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "bench")]
from paths import MRI_I58 as MRI, OCT_I58 as OCT, PROJECT_ROOT as ROOT  # noqa: E402  (bench/paths.py)

INPUTS = ROOT / "baselines" / "inputs"
SHOTS = ROOT / "baselines" / "shots"
PY = sys.executable
ENV = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONIOENCODING": "utf-8", "CUDA_VISIBLE_DEVICES": ""}
OCTREG_AFFINE_STEPS = ("mri", "oct_fine_grid", "oct_mask", "two_class", "search_refine_ngf")   # result.json seconds, §1 to §5


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


def scales(T, oct_affine):
    """Scale of the transform along each OCT array axis (octreg's pose.scale_per_oct_axis): the length of T applied to a unit
    step along each axis of the OCT header, so 1 means the voxel size is kept."""
    A = np.asarray(oct_affine, float)
    sp = np.linalg.norm(A[:3, :3], axis=0)
    return np.linalg.norm(T[:3, :3] @ (A[:3, :3] / sp), axis=0).round(4).tolist()


def hardware(run_dir: Path, tool: str) -> str:
    """The device and thread count a run recorded in threads.txt ('' when there is none, as for the start)."""
    f = run_dir / "threads.txt"
    t = f.read_text(encoding="utf-8", errors="replace") if f.is_file() else ""
    if not t.strip():
        return ""
    if "GPU" in t:
        return "GPU"
    if tool == "flirt":
        return "CPU (FLIRT runs single-threaded)"
    n = re.findall(r"\d+", t.split("machine")[0])
    return f"CPU, {n[0]} threads" if n else "CPU"


def row_of(name, tool, variant, T, wall, hw, oct_affine):
    return {"run": name, "tool": tool, "variant": variant, "scale_per_oct_axis": scales(T, oct_affine),
            "det": float(np.linalg.det(T[:3, :3])), "wall_s": wall, "hardware": hw}


def evaluate(run_dir: Path, shots: bool, force: bool, oct_affine):
    chk = run_dir / "check.json"
    if not chk.is_file() or not json.loads(chk.read_text())["ok"]:
        print(f"{run_dir}: no check.json ok, skipped", flush=True)
        return None
    name = f"{run_dir.parent.name}_{run_dir.name}"
    if force or not (run_dir / "oct_in_mri_affine.nii.gz").is_file():
        run([PY, "-m", "octreg", "apply", "--run", str(run_dir), "--moving", str(OCT), "--reference", str(MRI),
             "-o", str(run_dir / "oct_in_mri_affine.nii.gz"), "--affine-only"], run_dir / "apply.log")
    if shots and (force or not (SHOTS / "shots" / f"ax_26_{name}.png").is_file()):
        run(["wsl", "bash", wsl_path(SHOTS / "shots.sh"), name, wsl_path(run_dir / "oct_in_mri_affine.nii.gz")],
            run_dir / "shots.log")
    tm = run_dir / "time.txt"
    row = row_of(name, run_dir.parent.name, run_dir.name, np.loadtxt(run_dir / "T_oct2mri.txt"),
                 float(tm.read_text().split()[0]) if tm.is_file() else None, hardware(run_dir, run_dir.parent.name), oct_affine)
    print(f"{name}: scale {row['scale_per_oct_axis']}, det {row['det']:.4f}, wall {row['wall_s']} s, {row['hardware']}",
          flush=True)
    return row


def octreg_row(run_dir: Path, oct_affine):
    res = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    wall = sum(res["seconds"][k] for k in OCTREG_AFFINE_STEPS)
    return row_of("octreg", "octreg", "§1 to §5", np.loadtxt(run_dir / "T_oct2mri.txt"), wall,
                  "GPU" if res["inputs"].get("device") == "cuda" else "CPU", oct_affine)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--octreg-run", type=Path, default=None, help="add the row of this octreg run")
    ap.add_argument("--oct-grid", type=Path, default=INPUTS / "oct_0.08mm.nii.gz",
                    help="the moving image of the comparison, whose header gives the OCT array axes of the scales")
    ap.add_argument("--shots", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--summary", type=Path, default=ROOT / "baselines" / "summary_affine.json")
    a = ap.parse_args()
    oct_affine = nib.load(str(a.oct_grid)).affine
    rows = json.loads(a.summary.read_text(encoding="utf-8")) if a.summary.is_file() else {}
    new = [octreg_row(a.octreg_run, oct_affine)] if a.octreg_run else []
    for r in (r for r in a.runs if r.is_dir()):
        row = evaluate(r, a.shots, a.force, oct_affine)
        if row:
            new.append(row)
    for row in new:
        rows = json.loads(a.summary.read_text(encoding="utf-8")) if a.summary.is_file() else {}   # merge with other runs
        rows[row["run"]] = row
        a.summary.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"{len(rows)} rows in {a.summary}")


if __name__ == "__main__":
    main()

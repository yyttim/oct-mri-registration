#!/usr/bin/env python3
"""Properties of the affine baseline runs that need no reference.

    python bench/baselines/evaluate_runs.py [RUN ...] [--octreg-run DIR] [--oct OCT] [--mri MRI] [--oct-grid OCT_0.08MM]
        [--summary OUT.json] [--force]

For every RUN made by rundir.py init with a check.json that is ok:
  1. octreg apply: the 20 um OCT (--oct) through RUN/T_oct2mri.txt onto the grid of the MRI (--mri) -> RUN/oct_in_mri.nii.gz,
     with the same resampler for every method, kept when it exists unless --force is given.
  2. T itself, the scale along each OCT array axis (1 = voxel size kept), the determinant of the 3x3 of T (1 = volume kept),
     the rotation of T from the start below (deg), the shift of the OCT box centre under T from its place under the start
     (mm), and the correlation and support Dice of RUN/check.json.
  3. The wall time of the registration in seconds from RUN/time.txt and the hardware from RUN/threads.txt.
With --octreg-run, a row "octreg" is added for that run, timed by the sum of its §1 to §4 steps in result.json, and the start
"reference_centres" (the header orientation with the image box centres aligned, bench/ablate.py header_start of --oct and
--mri) is written by rundir.py init to <project root>/baselines/affine/reference/centres with its overlay and row. A row per
run goes into --summary (default <project root>/baselines/summary_affine.json). The roots and the defaults of --oct and --mri,
the two I58 files, come from bench/paths.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "bench"), str(REPO / "bench" / "baselines")]
import rundir  # noqa: E402
from ablate import header_start  # noqa: E402
from octreg import geometry as G, io  # noqa: E402
from paths import MRI_I58, OCT_I58, PROJECT_ROOT as ROOT  # noqa: E402

INPUTS = ROOT / "baselines" / "inputs"
START = ROOT / "baselines" / "affine" / "reference" / "centres"
PY = sys.executable
ENV = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONIOENCODING": "utf-8", "CUDA_VISIBLE_DEVICES": ""}


def run(cmd, log=None):
    r = subprocess.run(cmd, capture_output=True, text=True, env=ENV, encoding="utf-8", errors="replace")
    if log:
        Path(log).write_text(r.stdout + r.stderr, encoding="utf-8")
    if r.returncode:
        raise RuntimeError(f"{cmd[0]} failed ({r.returncode}): {(r.stdout + r.stderr)[-2000:]}")


def scales(T, oct_affine):
    """Scale of T along each OCT array axis as in octreg's pose.scale_per_oct_axis, 1 where the voxel size is kept."""
    A = np.asarray(oct_affine, float)
    sp = np.linalg.norm(A[:3, :3], axis=0)
    return np.linalg.norm(T[:3, :3] @ (A[:3, :3] / sp), axis=0).round(4).tolist()


def hardware(run_dir: Path, tool: str) -> str:
    """The thread count a run recorded in threads.txt."""
    f = run_dir / "threads.txt"
    n = re.findall(r"\d+", f.read_text(encoding="utf-8", errors="replace")) if f.is_file() else []
    return "" if not n else "CPU (FLIRT runs single-threaded)" if tool == "flirt" else f"CPU, {n[0]} threads"


def rotation_deg(T, S):
    """Angle (deg) of the polar rotation of T S^-1."""
    U, _, Vt = np.linalg.svd((T @ np.linalg.inv(S))[:3, :3])
    return float(np.degrees(np.arccos(np.clip((np.trace(U @ Vt) - 1) / 2, -1, 1))))


def start_of(oct_path: Path, mri_path: Path):
    """The start S, header_start of the two files, and the OCT box centre (mm)."""
    vo, vm = io.load_volume(oct_path), io.load_volume(mri_path)
    return header_start(vm.affine, vm.shape, vo.affine, vo.shape), G.box(vo.affine, vo.shape)[0]


def row_of(name, tool, variant, T, wall, hw, scale, start, check=None):
    """One row of the summary. start: (S, c) of start_of. check: RUN/check.json of a converted transform."""
    S, c = start
    check = check or {}
    return {"run": name, "tool": tool, "variant": variant, "T_oct2mri": np.round(T, 10).tolist(), "scale_per_oct_axis": scale,
            "det": float(np.linalg.det(T[:3, :3])), "rotation_from_header_deg": rotation_deg(T, S),
            "box_centre_shift_mm": float(np.linalg.norm(G.apply_affine(T, c) - G.apply_affine(S, c))),
            "correlation": check.get("correlation"), "support_dice": check.get("support_dice"), "wall_s": wall,
            "hardware": hw}


def overlay(run_dir: Path, force: bool, oct_path: Path, mri_path: Path):
    if force or not (run_dir / "oct_in_mri.nii.gz").is_file():
        run([PY, "-m", "octreg", "apply", "--run", str(run_dir), "--moving", str(oct_path), "--reference", str(mri_path),
             "-o", str(run_dir / "oct_in_mri.nii.gz")], run_dir / "apply.log")


def evaluate(run_dir: Path, force: bool, oct_affine, oct_path: Path, mri_path: Path, start):
    chk = run_dir / "check.json"
    check = json.loads(chk.read_text()) if chk.is_file() else {"ok": False}
    if not check["ok"]:
        print(f"{run_dir}: no check.json ok, skipped", flush=True)
        return None
    overlay(run_dir, force, oct_path, mri_path)
    tm, T = run_dir / "time.txt", np.loadtxt(run_dir / "T_oct2mri.txt")
    row = row_of(f"{run_dir.parent.name}_{run_dir.name}", run_dir.parent.name, run_dir.name, T,
                 float(tm.read_text().split()[0]) if tm.is_file() else None, hardware(run_dir, run_dir.parent.name),
                 scales(T, oct_affine), start, check)
    print(f"{row['run']}: scale {row['scale_per_oct_axis']}, det {row['det']:.4f}, rotation from the header "
          f"{row['rotation_from_header_deg']:.1f} deg, wall {row['wall_s']} s, {row['hardware']}", flush=True)
    return row


def octreg_row(run_dir: Path, start):
    res = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    return row_of("octreg", "octreg", "default", np.loadtxt(run_dir / "T_oct2mri.txt"),
                  res["seconds"]["total"] - res["seconds"]["outputs"], "GPU" if res["inputs"]["device"] == "cuda" else "CPU",
                  np.round(res["pose"]["scale_per_oct_axis"], 4).tolist(), start)


def start_row(octreg_run: Path, force: bool, oct_affine, oct_path: Path, mri_path: Path, start):
    """The start S as the run directory START with its overlay."""
    S = start[0]
    rundir.init(START, S, octreg_run / "result.json")
    overlay(START, force, oct_path, mri_path)
    return row_of(f"{START.parent.name}_{START.name}", START.parent.name, START.name, S, None, "", scales(S, oct_affine),
                  start)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--octreg-run", type=Path, default=None, help="add the row of this octreg run and the start")
    ap.add_argument("--oct", type=Path, default=OCT_I58, help="the 20 um OCT that octreg apply resamples")
    ap.add_argument("--mri", type=Path, default=MRI_I58, help="the MRI crop, the grid of the overlays")
    ap.add_argument("--oct-grid", type=Path, default=INPUTS / "oct_0.08mm.nii.gz",
                    help="the moving image of the comparison, whose header gives the OCT array axes of the scales")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--summary", type=Path, default=ROOT / "baselines" / "summary_affine.json")
    a = ap.parse_args()
    oct_affine, start = nib.load(str(a.oct_grid)).affine, start_of(a.oct, a.mri)
    new = ([octreg_row(a.octreg_run, start), start_row(a.octreg_run, a.force, oct_affine, a.oct, a.mri, start)]
           if a.octreg_run else [])
    runs = [r for r in a.runs if r.is_dir() and r.resolve() != START.resolve()]
    new += [row for row in (evaluate(r, a.force, oct_affine, a.oct, a.mri, start) for r in runs) if row]
    rows = json.loads(a.summary.read_text(encoding="utf-8")) if a.summary.is_file() else {}
    rows.update({row["run"]: row for row in new})
    a.summary.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"{len(rows)} rows in {a.summary}")


if __name__ == "__main__":
    main()

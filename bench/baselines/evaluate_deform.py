#!/usr/bin/env python3
"""Properties of the deformable baseline runs that need no reference.

    python bench/baselines/evaluate_deform.py RUN [RUN ...] [--octreg-run DIR] [--cache CACHE.npz] [--shots]
                                              [--summary OUT.json] [--force]

Every RUN holds octreg's affine (T_oct2mri.txt), the tool's field converted to octreg's convention (oct2mri_warp.nii.gz) and
a check_field.json that is ok (bench/baselines/rundir.py). For each RUN:
  1. octreg apply: the 20 um OCT through affine and field onto the MRI grid -> RUN/oct_in_mri.nii.gz (the overlays of
     bench/figures.py).
  2. The Jacobian determinant J of x + u(x), u the field, by finite differences on the field's own grid: its minimum, the
     share of voxels with J <= 0 (folded) and the standard deviation of log J (the smoothness measure of Learn2Reg, Hering et
     al., IEEE Trans. Med. Imaging 42:697, 2023), over the MRI foreground of --cache (octreg's MRI foreground on the 0.15 mm
     base grid) sampled onto the field's grid.
  3. The wall time of the registration from RUN/time.txt (seconds, written by the tool's own command), the hardware from
     RUN/threads.txt and the spacing of the field grid.
  4. With --shots: freeview screenshots at the shared sections, named deform_<tool>_<variant>, by
     OCTREG_PROJECT_ROOT/baselines/shots/shots.sh in WSL, a local script that is not in the repository.
With --octreg-run, the rows "none" (octreg's affine, no field) and "octreg" (its §6 field oct2mri_warp.nii.gz, wall time
the §6 step time of result.json) are added. A row per run goes into --summary (default
OCTREG_PROJECT_ROOT/baselines/summary_deform.json). The roots come from bench/paths.py.
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

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "bench"), str(REPO / "bench" / "baselines")]
from octreg import geometry as G, io  # noqa: E402
from evaluate_runs import hardware  # noqa: E402
from paths import BENCH_RUNS, MRI_I58 as MRI, OCT_I58 as OCT, PROJECT_ROOT as ROOT  # noqa: E402  (bench/paths.py)

SHOTS = ROOT / "baselines" / "shots"
PY = sys.executable
ENV = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONIOENCODING": "utf-8", "CUDA_VISIBLE_DEVICES": ""}


def wsl_path(p: Path) -> str:
    s = str(Path(p).resolve()).replace("\\", "/")
    return "/mnt/" + s[0].lower() + s[2:] if len(s) > 1 and s[1] == ":" else s


def run(cmd, log=None):
    r = subprocess.run(cmd, capture_output=True, text=True, env=ENV, encoding="utf-8", errors="replace")
    if log:
        Path(log).write_text(r.stdout + r.stderr, encoding="utf-8")
    if r.returncode:
        raise RuntimeError(f"{cmd[0]} failed ({r.returncode}): {(r.stdout + r.stderr)[-2000:]}")


def field_stats(field, A_f, fg_mask, A_m):
    """The Jacobian determinant of x + u(x) on the field grid, over the MRI foreground sampled onto that grid.
    field: [3, D, H, W] mm along the world axes, A_f its grid. -> {jac_min, jac_folding_fraction, sd_log_jac, n_foreground}."""
    shape = field.shape[1:]
    fg = G.resample_to(fg_mask.astype(np.uint8), A_m, shape, A_f, np.eye(4), order=0) > 0
    h = np.linalg.norm(A_f[:3, :3], axis=0)
    J = np.zeros(shape + (3, 3), np.float32)
    for i in range(3):                                   # d u_i / d s_j, s_j the distance in mm along grid axis j
        g = np.gradient(field[i], *h)
        for j in range(3):
            J[..., i, j] = g[j]
    # the field components follow the world axes, while the grid axes may be permuted or flipped, so the derivative is
    # expressed in world coordinates
    R = A_f[:3, :3] / h                                  # voxel step -> world direction (orthonormal for an axis-aligned grid)
    Jw = J @ np.linalg.inv(R)[None, None, None]           # d u / d x_world
    det = np.linalg.det(np.eye(3)[None, None, None] + Jw)
    sel = fg & np.isfinite(det)
    d = det[sel]
    return {"jac_min": float(d.min()), "jac_folding_fraction": float((d <= 0).mean()),
            "sd_log_jac": float(np.std(np.log(np.clip(d, 1e-6, None)))), "n_foreground": int(sel.sum())}


def stats_of(path, fg, A_m):
    field, A_f = io.load_field(path)
    return {**field_stats(field, A_f, fg, A_m), "field_grid_mm": round(float(np.linalg.norm(A_f[:3, 0])), 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--octreg-run", type=Path, default=None, help="add the rows 'none' and 'octreg' of this octreg run")
    ap.add_argument("--cache", type=Path, default=BENCH_RUNS / "I58" / "cache.npz",
                    help="base grids of bench/ngf_lam.py cache, for the MRI foreground (mri_mask, mri_affine)")
    ap.add_argument("--shots", action="store_true")
    ap.add_argument("--force", action="store_true", help="redo octreg apply when oct_in_mri.nii.gz exists")
    ap.add_argument("--summary", type=Path, default=ROOT / "baselines" / "summary_deform.json")
    a = ap.parse_args()
    t0 = time.time()
    z = np.load(a.cache)
    fg, A_m = np.asarray(z["mri_mask"], bool), np.asarray(z["mri_affine"], float)

    def merge(row):
        rows = json.loads(a.summary.read_text(encoding="utf-8")) if a.summary.is_file() else {}   # merge with other runs
        rows[row["run"]] = row
        a.summary.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")

    if a.octreg_run:
        res = json.loads((a.octreg_run / "result.json").read_text(encoding="utf-8"))
        dev = "GPU" if res["inputs"].get("device") == "cuda" else "CPU"
        merge({"run": "none", "tool": "none", "variant": "no deformation"})
        merge({"run": "octreg", "tool": "octreg", "variant": "§6", **stats_of(a.octreg_run / "oct2mri_warp.nii.gz", fg, A_m),
               "wall_s": res["seconds"].get("deform"), "hardware": dev})
        print(f"octreg §6 and no deformation added [{time.time() - t0:.0f} s]", flush=True)
    for r in (r for r in a.runs if r.is_dir()):
        chk = r / "check_field.json"
        if not chk.is_file() or not json.loads(chk.read_text())["ok"]:
            print(f"{r}: no check_field.json ok, skipped", flush=True)
            continue
        name = f"deform_{r.parent.name}_{r.name}"
        if a.force or not (r / "oct_in_mri.nii.gz").is_file():
            run([PY, "-m", "octreg", "apply", "--run", str(r), "--moving", str(OCT), "--reference", str(MRI),
                 "-o", str(r / "oct_in_mri.nii.gz")], r / "apply.log")
        if a.shots and (a.force or not (SHOTS / "shots" / f"ax_26_{name}.png").is_file()):
            run(["wsl", "bash", wsl_path(SHOTS / "shots.sh"), name, wsl_path(r / "oct_in_mri.nii.gz")], r / "shots.log")
        tm = r / "time.txt"
        row = {"run": name, "tool": r.parent.name, "variant": r.name, **stats_of(r / "oct2mri_warp.nii.gz", fg, A_m),
               "wall_s": float(tm.read_text().split()[0]) if tm.is_file() else None, "hardware": hardware(r, r.parent.name)}
        merge(row)
        print(f"{name}: Jacobian min {row['jac_min']:.3f}, folded {100 * row['jac_folding_fraction']:.2f} %, SD log J "
              f"{row['sd_log_jac']:.3f}, wall {row['wall_s']} s, {row['hardware']}", flush=True)
    print(f"{len(json.loads(a.summary.read_text(encoding='utf-8')))} rows in {a.summary} [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()

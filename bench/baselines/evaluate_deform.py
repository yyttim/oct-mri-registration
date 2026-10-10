#!/usr/bin/env python3
"""Evaluate deformable baseline runs against octreg's §6 on the same read-outs (bench only).

    python bench/baselines/evaluate_deform.py RUN [RUN ...] [--octreg-run DIR] [--cache CACHE.npz] [--shots] [--device cuda]
                                              [--summary OUT.json] [--force]

Every RUN holds octreg's affine (T_oct2mri.txt), the tool's field converted to octreg's convention (oct2mri_warp.nii.gz) and
check_field.json ok (bench/baselines/rundir.py). The octreg run itself and "no deformation" are scored the same way as rows.
  1. octreg apply: the 20 um OCT through affine + field onto the MRI grid -> RUN/oct_in_mri.nii.gz (one resampler for all).
  2. The §6 read-outs of bench/ablate_deform.py on the warped OCT on the 0.15 mm base grid: interior-match and surface-edge
     residuals, the share of edge offsets within 0.3 mm, F of §5 at sigma 0.3 mm, the two-class score over the core, and the
     boundary residual per 4 mm along the sectioning axis. These are octreg's own measurements, label-free but ours.
  3. Field statistics over the MRI foreground: |u| median / p95 / max, the Jacobian determinant of x + u(x) (min, fraction
     <= 0, standard deviation of log J).
  4. with --shots: freeview screenshots at the shared sections, named deform_<tool>_<variant>.
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
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bench"))
import ablate_deform as A  # noqa: E402
from octreg import geometry as G, io  # noqa: E402
from octreg.search import to_torch  # noqa: E402

ROOT = Path(os.environ.get("OCTREG_PROJECT_ROOT") or REPO)
I58 = Path(os.environ.get("OCTREG_I58_DIR") or ROOT / "data" / "I58")
OCT = I58 / "I58_Brainstem_mus_Slice_full_20um_corr.nii.gz"
MRI = I58 / "I58_brainstem_MRI_cropped_to_OCT.nii.gz"
INPUTS = ROOT / "baselines" / "inputs"
SHOTS = ROOT / "baselines" / "shots"
PY = sys.executable
ENV = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONIOENCODING": "utf-8"}


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
    """|u| and the Jacobian of x + u(x) on the field grid, over the MRI foreground sampled onto that grid."""
    mag = np.linalg.norm(field, axis=0)
    fg = G.resample_to(fg_mask.astype(np.uint8), A_m, mag.shape, A_f, np.eye(4), order=0) > 0
    h = np.linalg.norm(A_f[:3, :3], axis=0)
    J = np.zeros(mag.shape + (3, 3), np.float32)
    for i in range(3):                                   # d u_i / d x_j in world mm, the grid axes taken along the world axes
        g = np.gradient(field[i], *h)
        for j in range(3):
            J[..., i, j] = g[j]
    # the field components follow the world axes, the grid axes may be permuted or flipped: express the derivative in world
    R = A_f[:3, :3] / h                                  # voxel step -> world direction (orthonormal for an axis-aligned grid)
    Jw = J @ np.linalg.inv(R)[None, None, None]           # d u / d x_world
    det = np.linalg.det(np.eye(3)[None, None, None] + Jw)
    sel = fg & np.isfinite(det)
    d = det[sel]
    return {"field_median_mm": float(np.median(mag[fg])), "field_p95_mm": float(np.percentile(mag[fg], 95)),
            "field_max_mm": float(mag[fg].max()), "jac_min": float(d.min()), "jac_folding_fraction": float((d <= 0).mean()),
            "sd_log_jac": float(np.std(np.log(np.clip(d, 1e-6, None)))), "n_foreground": int(sel.sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--octreg-run", type=Path, default=ROOT / "bench_runs" / "I58" / "v2" / "main")
    ap.add_argument("--cache", type=Path, default=ROOT / "bench_runs" / "I58" / "pr6" / "cache.npz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shots", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--summary", type=Path, default=ROOT / "baselines" / "summary_deform.json")
    a = ap.parse_args()
    t0 = time.time()
    z = np.load(a.cache)
    mri = (z["mri_arr"], z["mri_mask"], z["mri_affine"])
    o_arr, o_mask, A_o = z["oct_arr"], z["oct_mask"], z["oct_affine"]
    res_run = json.loads((a.octreg_run / "result.json").read_text(encoding="utf-8"))
    T, polarity = np.asarray(res_run["T_oct2mri"], float), int(res_run["pose"]["polarity"])
    A_m, shape = np.asarray(mri[2], float), tuple(mri[0].shape)
    with torch.no_grad():
        src = A._on_grid(torch.stack([to_torch(o_arr, a.device), to_torch(o_mask, a.device)]), A_o, shape, A_m, np.linalg.inv(T))
    axis = A.stripe_axis(o_arr, o_mask)
    d = T[:3, :3] @ A_o[:3, axis]
    d /= np.linalg.norm(d)
    idx = np.argwhere(o_mask)[:: max(1, int(o_mask.sum() // 200000))]
    s = G.apply_affine(T @ A_o, idx.astype(float)) @ d
    s0, s1 = float(s.min()), float(s.max())
    mri_axis = int(np.argmax(np.abs(np.linalg.inv(A_m[:3, :3]) @ d)))
    if (np.linalg.inv(A_m[:3, :3]) @ d)[mri_axis] < 0:
        d, (s0, s1) = -d, (-s1, -s0)
    edges = np.arange(0.0, s1 - s0 + A.BIN_MM, A.BIN_MM)
    s_of = lambda X: np.asarray(X, float).reshape(-1, 3) @ d - s0
    scorer = A.Scorer(mri, polarity, s_of, edges, a.device)
    print(f"[{time.time() - t0:.0f} s] scorer ready on the {shape} base grid", flush=True)
    arr_aff, mask_aff = src[0].cpu().numpy(), src[1].cpu().numpy()

    def score_field(field, A_f):
        if field is None:
            vol, msk = arr_aff, mask_aff
        else:
            vol = G.resample_to(arr_aff, A_m, shape, A_m, np.eye(4), order=1, field=field, affine_field=A_f)
            msk = G.resample_to(mask_aff, A_m, shape, A_m, np.eye(4), order=1, field=field, affine_field=A_f)
        return scorer(to_torch(vol, a.device), to_torch(msk, a.device))

    rows = json.loads(a.summary.read_text(encoding="utf-8")) if a.summary.is_file() else {}
    fg = np.asarray(mri[1], bool)
    if "none" not in rows or a.force:
        rows["none"] = {"run": "none", "tool": "none", "variant": "no deformation", **score_field(None, None)}
        print("none:", {k: rows["none"][k] for k in ("interior_mm", "boundary_mm", "within", "F")}, flush=True)
    if "octreg" not in rows or a.force:
        f, A_f = io.load_field(a.octreg_run / "oct2mri_warp.nii.gz")
        rows["octreg"] = {"run": "octreg", "tool": "octreg", "variant": "§6", **score_field(f, A_f), **field_stats(f, A_f, fg, A_m),
                          "wall_s": res_run["seconds"].get("deform")}
        print("octreg:", {k: rows["octreg"][k] for k in ("interior_mm", "boundary_mm", "within", "F", "field_max_mm", "jac_min")}, flush=True)
    a.summary.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    for r in a.runs:
        chk = r / "check_field.json"
        if not chk.is_file() or not json.loads(chk.read_text())["ok"]:
            print(f"{r}: no check_field.json ok, skipped", flush=True)
            continue
        name = f"deform_{r.parent.name}_{r.name}"
        if a.force or not (r / "oct_in_mri.nii.gz").is_file():
            run([PY, "-m", "octreg", "apply", "--run", str(r), "--moving", str(OCT), "--reference", str(MRI),
                 "-o", str(r / "oct_in_mri.nii.gz")], r / "apply.log")
        f, A_f = io.load_field(r / "oct2mri_warp.nii.gz")
        tm = r / "time.txt"
        row = {"run": name, "tool": r.parent.name, "variant": r.name, **score_field(f, A_f), **field_stats(f, A_f, fg, A_m),
               "wall_s": float(tm.read_text().split()[0]) if tm.is_file() else None}
        if a.shots and (a.force or not (SHOTS / "shots" / f"ax_26_{name}.png").is_file()):
            run(["wsl", "bash", wsl_path(SHOTS / "shots.sh"), name, wsl_path(r / "oct_in_mri.nii.gz")], r / "shots.log")
        rows = json.loads(a.summary.read_text(encoding="utf-8")) if a.summary.is_file() else {}   # merge with other runs
        rows[name] = row
        (r / "eval_deform.json").write_text(json.dumps(row, indent=1), encoding="utf-8")
        a.summary.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"{name}: interior {row['interior_mm']:.3f} boundary {row['boundary_mm']:.3f} within {100 * row['within']:.0f} % "
              f"F {row['F']:.4f} field median/max {row['field_median_mm']:.2f}/{row['field_max_mm']:.2f} mm "
              f"jac min {row['jac_min']:.2f} folding {100 * row['jac_folding_fraction']:.2f} %", flush=True)
    print(f"{len(rows)} rows in {a.summary} [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run directories of the baseline comparison.

    python bench/baselines/rundir.py init RUN --T T.txt [--template RESULT.json]
    python bench/baselines/rundir.py check RUN --warped WARPED.nii.gz [--oct OCT.nii.gz] [--mri MRI.nii.gz]
    python bench/baselines/rundir.py check-field RUN --warped WARPED.nii.gz --moving MOVING.nii.gz [--mri MRI.nii.gz]

init         makes RUN a run directory in octreg's layout: T_oct2mri.txt (the given 4x4, moving world -> fixed world in
             NIfTI RAS mm, which is octreg's convention) and a result.json that holds only the inputs block of a template
             octreg run (default <project root>/bench_runs/I58/octreg/result.json), so that `octreg apply --run RUN` works on
             it. A mirrored transform (det <= 0) is refused.
check        tests the convention of a converted affine. The OCT resampled through RUN/T_oct2mri.txt onto the MRI grid by
             octreg's own resampler is compared with the image the tool itself wrote on that grid (WARPED), by the Pearson
             correlation over the voxels where either is non-zero and the Dice of the two non-zero supports. A correct
             conversion gives a correlation near 1, and the check passes when both exceed 0.95. Writes RUN/check.json.
check-field  the same test for a tool's displacement field after its conversion to octreg's convention
             (RUN/oct2mri_warp.nii.gz, deform_tools.py), with the affine-aligned OCT on the MRI grid (MOVING,
             oct_affine_0.08mm.nii.gz of prepare_inputs.py) sent through it. Writes RUN/check_field.json.
Defaults: the inputs of bench/baselines/README.md under <project root>/baselines/inputs (oct_0.08mm.nii.gz,
mri_0.08mm.nii.gz), the project root of bench/paths.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "bench")]
from octreg import geometry as G, io  # noqa: E402
from paths import PROJECT_ROOT as ROOT  # noqa: E402  (bench/paths.py)

INPUTS = ROOT / "baselines" / "inputs"


def init(run: Path, T_file: Path, template: Path | None = None):
    T = np.loadtxt(T_file, dtype=float)
    if T.shape != (4, 4) or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{T_file}: not a 4x4 affine")
    if np.linalg.det(T[:3, :3]) <= 0:
        raise ValueError(f"{T_file}: mirrored transform (det <= 0)")
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    np.savetxt(run / "T_oct2mri.txt", T, fmt="%.10f")
    src = json.loads(Path(template or ROOT / "bench_runs" / "I58" / "octreg" / "result.json").read_text(encoding="utf-8"))
    res = {"octreg_version": src.get("octreg_version"), "inputs": src["inputs"], "baseline": run.name,
           "note": "run directory of a baseline transform: T_oct2mri.txt is the converted baseline affine, "
                   "the inputs block is copied from the octreg run so that octreg apply works"}
    (run / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"{run}: T_oct2mri.txt written, det {np.linalg.det(T[:3, :3]):.4f}, translation {T[:3, 3].round(2).tolist()} mm")


def _compare(run, name, warped, mri, ours, extra=None):
    """octreg's resampling `ours` against the tool's own image `warped` on the MRI grid -> RUN/name, True when both measures
    exceed 0.95."""
    w = nib.load(str(warped))
    if w.shape != mri.shape or not np.allclose(w.affine, mri.affine, atol=1e-3):
        raise ValueError(f"{warped}: not on the MRI grid ({w.shape} vs {mri.shape})")
    theirs = np.asarray(w.dataobj, np.float32)
    a, b = ours > 0, theirs > 0
    both = a | b
    r = float(np.corrcoef(ours[both], theirs[both])[0, 1]) if both.any() else float("nan")
    dice = float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))
    out = {"warped": str(warped), "correlation": r, "support_dice": dice, "n_voxels": int(both.sum()), **(extra or {}),
           "ok": bool(r > 0.95 and dice > 0.95)}
    (Path(run) / name).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{Path(run).name}: correlation {r:.4f}, support Dice {dice:.4f} -> {'ok' if out['ok'] else 'MISMATCH'}")
    return out["ok"]


def check(run: Path, warped: Path, oct_path: Path, mri_path: Path):
    T = np.loadtxt(Path(run) / "T_oct2mri.txt")
    o, m = nib.load(str(oct_path)), nib.load(str(mri_path))
    ours = G.resample_to(np.asarray(o.dataobj, np.float32), o.affine, m.shape, m.affine, np.linalg.inv(T), order=1)
    return _compare(run, "check.json", warped, m, ours)


def check_field(run: Path, warped: Path, moving: Path, mri_path: Path):
    """The affine-aligned OCT on the MRI grid sent through RUN/oct2mri_warp.nii.gz by octreg's resampler (moving(x + u(x)))."""
    field, A_f = io.load_field(Path(run) / "oct2mri_warp.nii.gz")
    mv, m = nib.load(str(moving)), nib.load(str(mri_path))
    ours = G.resample_to(np.asarray(mv.dataobj, np.float32), mv.affine, m.shape, m.affine, np.eye(4), order=1,
                         field=field, affine_field=A_f)
    mag = np.linalg.norm(field, axis=0)
    return _compare(run, "check_field.json", warped, m, ours,
                    {"field_grid_mm": float(np.linalg.norm(A_f[:3, 0])), "field_median_mm": float(np.median(mag)),
                     "field_max_mm": float(mag.max())})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init")
    i.add_argument("run", type=Path)
    i.add_argument("--T", required=True, type=Path)
    i.add_argument("--template", type=Path, default=None)
    c = sub.add_parser("check")
    c.add_argument("run", type=Path)
    c.add_argument("--warped", required=True, type=Path)
    c.add_argument("--oct", type=Path, default=INPUTS / "oct_0.08mm.nii.gz")
    c.add_argument("--mri", type=Path, default=INPUTS / "mri_0.08mm.nii.gz")
    f = sub.add_parser("check-field")
    f.add_argument("run", type=Path)
    f.add_argument("--warped", required=True, type=Path)
    f.add_argument("--moving", type=Path, required=True)
    f.add_argument("--mri", type=Path, default=INPUTS / "mri_0.08mm.nii.gz")
    a = ap.parse_args()
    if a.cmd == "init":
        init(a.run, a.T, a.template)
    elif a.cmd == "check":
        sys.exit(0 if check(a.run, a.warped, a.oct, a.mri) else 1)
    else:
        sys.exit(0 if check_field(a.run, a.warped, a.moving, a.mri) else 1)


if __name__ == "__main__":
    main()

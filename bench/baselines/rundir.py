"""Run directories of the baseline comparison.

init         writes T_oct2mri.txt (moving world -> fixed world, NIfTI RAS mm) and a result.json with the inputs block of a
             template octreg run (default <project root>/bench_runs/I58/octreg/result.json), so that `octreg apply --run RUN`
             works on RUN. A mirrored transform is refused.
check        compares the OCT resampled by octreg through RUN/T_oct2mri.txt with the tool's own image `warped` on the MRI grid.
             It passes when the Pearson correlation over the voxels where either is non-zero and the Dice of the two non-zero
             supports both exceed 0.95. Writes RUN/check.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "bench")]
from octreg import geometry as G  # noqa: E402
from paths import BENCH_RUNS  # noqa: E402


def init(run: Path, T, template: Path | None = None):
    T = np.asarray(T, float)
    if T.shape != (4, 4) or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{run}: T is not a 4x4 affine")
    if np.linalg.det(T[:3, :3]) <= 0:
        raise ValueError(f"{run}: mirrored transform (det <= 0)")
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    np.savetxt(run / "T_oct2mri.txt", T, fmt="%.10f")
    src = json.loads(Path(template or BENCH_RUNS / "I58" / "octreg" / "result.json").read_text(encoding="utf-8"))
    res = {"octreg_version": src.get("octreg_version"), "inputs": src["inputs"], "baseline": run.name,
           "note": "run directory of a baseline transform: T_oct2mri.txt is the converted baseline affine, "
                   "the inputs block is copied from the octreg run so that octreg apply works"}
    (run / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"{run}: T_oct2mri.txt written, det {np.linalg.det(T[:3, :3]):.4f}, translation {T[:3, 3].round(2).tolist()} mm")


def check(run: Path, warped: Path, oct_path: Path, mri_path: Path):
    """octreg's resampling through RUN/T_oct2mri.txt against the tool's image `warped`, ok when both exceed 0.95."""
    T = np.loadtxt(Path(run) / "T_oct2mri.txt")
    o, m = nib.load(str(oct_path)), nib.load(str(mri_path))
    ours = G.resample_to(np.asarray(o.dataobj, np.float32), o.affine, m.shape, m.affine, np.linalg.inv(T))
    w = nib.load(str(warped))
    if w.shape != m.shape or not np.allclose(w.affine, m.affine, atol=1e-3):
        raise ValueError(f"{warped}: not on the MRI grid ({w.shape} vs {m.shape})")
    theirs = np.asarray(w.dataobj, np.float32)
    a, b = ours > 0, theirs > 0
    both = a | b
    r = float(np.corrcoef(ours[both], theirs[both])[0, 1]) if both.any() else float("nan")
    dice = float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))
    out = {"warped": str(warped), "correlation": r, "support_dice": dice, "n_voxels": int(both.sum()),
           "ok": bool(r > 0.95 and dice > 0.95)}
    (Path(run) / "check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{Path(run).name}: correlation {r:.4f}, support Dice {dice:.4f} -> {'ok' if out['ok'] else 'MISMATCH'}")

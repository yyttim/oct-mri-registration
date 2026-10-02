#!/usr/bin/env python3
"""Checkerboard figures of the base vs. boundary27 §6 field (ablate_deform.py) near the block ends, to see whether the
residual gain there is visible by eye, not just in the numbers. Slices along MRI array axis 1 (~ the sectioning axis),
affine-only / base-field / boundary27-field side by side.

    python bench/fig_deform_boundary.py --out OUT --main RUN [--slices-mm 2,5,8,11,14,20]

Result (bench/results/deform_ablation.md): the field (base or boundary27) visibly improves edge alignment over
affine-only, but base and boundary27 look indistinguishable from each other at every slice checked -- the residual gain
(a few hundredths of a mm) is well under one MRI voxel (~0.08 mm), below what a checkerboard at this zoom can show. Real
in the numbers, not visible in the picture.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate as E                                                    # noqa: E402  (bench/evaluate.py)
from ablate import prepare                                              # noqa: E402  (bench/ablate.py: the §1 prep step)
from deform_common import load_grids                                    # noqa: E402
from octreg import deform as D                                          # noqa: E402
from octreg import geometry as G                                        # noqa: E402
from octreg.blockmatch import _on_grid                                  # noqa: E402
from octreg.params import Params                                        # noqa: E402
from octreg.search import to_torch                                      # noqa: E402


def fit_variant(mri, src, inside0, shape, A_m, P, dev):
    evidence = D.Evidence(mri, P, dev)
    before = evidence.measure(src[0], inside0)
    corners = G.apply_affine(A_m, np.array([[i, j, k] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1)
                                            for k in (0, shape[2] - 1)], float))
    lattice = D.Lattice(corners.min(0), corners.max(0), P.df_grid_mm)
    none = [float(np.median(np.linalg.norm(before["D"], axis=1))), float(np.median(np.abs(before["delta"])))]
    best, _ = D.choose(lattice, before, sum(none), P)
    moved = src
    if best is not None:
        c = D.fit(lattice, before, best[0], P.df_huber_mm)
        field = lattice.on_grid(c, shape, A_m, dev)
        with torch.no_grad():
            moved = D.warp(src, A_m, field)
        print(f"  lam {best[0]:.3f} strain {lattice.strain(c):.3f}", flush=True)
    return moved


def checkerboard(mri_slice, oct_slice, n=8):
    h, w = mri_slice.shape
    cb = (np.add.outer(np.arange(h) // max(1, h // n), np.arange(w) // max(1, w // n)) % 2).astype(bool)
    mn, mx = np.percentile(mri_slice, 1), np.percentile(mri_slice, 99)
    on, ox = np.percentile(oct_slice, 1), np.percentile(oct_slice, 99)
    m = np.clip((mri_slice - mn) / max(mx - mn, 1e-6), 0, 1)
    o = np.clip((oct_slice - on) / max(ox - on, 1e-6), 0, 1)
    return np.where(cb, m, o)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--main", type=Path, required=True, help="CLI run dir (T_oct2mri.txt) whose pose the field is fitted on")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--slices-mm", default="2,5,8,11,14,20", help="MRI axis-1 positions (mm from the crop start) to render")
    args = ap.parse_args()
    out, dev = args.out, args.device
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    targets_mm = [float(x) for x in args.slices_mm.split(",") if x]

    P0 = Params()
    prepare(out, ["texture"])
    (m_arr, m_mask, A_m), (o_arr, o_mask, A_o) = load_grids(out)
    T = np.asarray(E.load_T(args.main / "T_oct2mri.txt"), float)
    mri, shape = (m_arr, m_mask, A_m), tuple(m_arr.shape)

    with torch.no_grad():
        src = _on_grid(torch.stack([to_torch(o_arr, dev), to_torch(o_mask, dev)]), A_o, shape, A_m, np.linalg.inv(T))
        inside0 = (src[1] > D.INSIDE).to(torch.float32)

    print("fitting base...", flush=True)
    moved_base = fit_variant(mri, src, inside0, shape, A_m, P0, dev)
    print("fitting boundary27...", flush=True)
    P27 = replace(P0, df_reach_mm=2.7, df_profile_mm=3.0)
    moved_27 = fit_variant(mri, src, inside0, shape, A_m, P27, dev)

    affine_src = src[0].cpu().numpy()
    base_src = moved_base[0].cpu().numpy()
    b27_src = moved_27[0].cpu().numpy()

    # MRI axis 1 ~ sectioning axis (shape[1] is axis 1); mm-per-voxel along it from A_m
    h1 = float(np.linalg.norm(A_m[:3, 1]))
    idxs = [min(shape[1] - 1, max(0, round(t / h1))) for t in targets_mm]

    for t_mm, j in zip(targets_mm, idxs):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        mri_sl = m_arr[:, j, :]
        for a, (title, oct_vol) in zip(ax, (("affine only (no field)", affine_src), ("base field", base_src), ("boundary27 field", b27_src))):
            cb = checkerboard(mri_sl, oct_vol[:, j, :])
            a.imshow(cb, cmap="gray", origin="lower")
            a.set_title(f"{title}\nMRI axis1 = {t_mm:g} mm")
            a.axis("off")
        fig.tight_layout()
        fig.savefig(fig_dir / f"deform_checker_{t_mm:02.0f}mm.png", dpi=110)
        plt.close(fig)
        print(f"saved figures/deform_checker_{t_mm:02.0f}mm.png", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""§6 (smooth deformation) at the block ends, on Xiangrui's I58 brainstem pair: where is the field held back, and does
relaxing octreg's own Params knobs help? The §1-5 counterpart is ablate.py; its own docstring notes §6 "leaves the pose
alone, so no variant runs it" there -- this is that missing ablation.

    python bench/ablate_deform.py --out OUT --main RUN [--variants base,boundary27,...] [--device cuda]

RUN is a CLI run directory (T_oct2mri.txt), e.g. the output of `octreg register` or of ablate.py's own `base` variant.
OUT holds OUT/prep (built once, reused by ablate.py and this script if pointed at the same directory) and
OUT/deform_variants.json.

Uses octreg's own Evidence / Lattice / choose / fit / warp (octreg.deform) directly, so a variant takes seconds-minutes, not
a whole registration. Nothing in octreg is modified; variants only change Params fields (df_max_strain, df_range_mm,
df_reach_mm, df_profile_mm, df_grid_mm). For the baseline (default Params) it also reports, along the sectioning axis of the
OCT (estimated from the section stripes): evidence coverage and how many blocks are rejected because their best match sits
on the border of the search range (truncation) or show no structure; the residuals before and after the field at the two
ends vs. the middle; where the strain limit binds. For every variant it reports the stage's own read-outs plus F
(fine-structure agreement at sigma 0.3 mm vs. a 10 mm-shift chance level), overall and excluding the first 10 mm of MRI
axis 1 (the torn cerebellar folia; left out of every evaluation number here and in BENCHMARK.md). The read-outs are
in-sample or semi-independent -- see docs/METHOD.md and bench/results/deform_ablation.md for the full reading.

Headline finding (bench/results/deform_ablation.md): of the variants below, `boundary27` (widening only the boundary-
evidence radius/profile length, df_reach_mm/df_profile_mm) is the one with a real, isolated improvement at the block ends;
every other variant either does nothing or trades a marginal global gain for no end-specific improvement.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate as E                                                    # noqa: E402  (bench/evaluate.py)
from ablate import prepare                                              # noqa: E402  (bench/ablate.py: the §1 prep step)
from deform_common import binned, contributions, f_vs_chance, load_grids, match_diag, stripe_axis  # noqa: E402
from octreg import deform as D                                          # noqa: E402
from octreg import geometry as G, io                                    # noqa: E402
from octreg.blockmatch import _erode, _on_grid, structure_feature       # noqa: E402
from octreg.params import Params                                        # noqa: E402
from octreg.search import to_torch                                      # noqa: E402

VARIANTS = {
    "base": {},
    "strain30": {"df_max_strain": 0.30},            # also stands in for 0.20: lambda floors at the schedule's smallest weight (0.3) either way
    "range20": {"df_range_mm": 2.0, "df_reach_mm": 2.0},
    "range27": {"df_range_mm": 2.7, "df_reach_mm": 2.7, "df_profile_mm": 3.0},      # Params needs 0 < df_reach_mm <= df_profile_mm
    "range20_strain30": {"df_range_mm": 2.0, "df_reach_mm": 2.0, "df_max_strain": 0.30},
    "grid35": {"df_grid_mm": 3.5},
    # boundary evidence only (df_reach_mm/df_profile_mm), df_range_mm left at default -- isolates range27's boundary gain
    # from its interior-search-range change. boundary27 is the one variant with a real, isolated end-zone gain.
    "boundary20": {"df_reach_mm": 2.0},                                               # df_profile_mm default 2.1 >= 2.0, no change needed
    "boundary27": {"df_reach_mm": 2.7, "df_profile_mm": 3.0},
}
# Tried and ruled out, not kept here (bench/results/deform_ablation.md "What's been ruled out"): a lower df_lams floor
# (weights down to 0.03) -- never selected by held-out CV, identical result to the matching strain-cap row above either
# alone or combined with range20. Multi-scale block matching at the ends -- see ablate_deform_multiscale.py instead.


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--main", type=Path, required=True, help="CLI run dir (T_oct2mri.txt) whose pose the field is fitted on")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--folia-mm", type=float, default=10.0, help="exclude MRI axis-1 positions below this (mm from crop start)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    names = args.variants.split(",")
    if set(names) - set(VARIANTS):
        ap.error(f"unknown variants {sorted(set(names) - set(VARIANTS))}")

    out, dev = args.out, args.device
    P0, t0 = Params(), time.time()
    prepare(out, ["texture"])
    (m_arr, m_mask, A_m), (o_arr, o_mask, A_o) = load_grids(out)
    T = np.asarray(E.load_T(args.main / "T_oct2mri.txt"), float)
    h_m, inv_m = float(np.linalg.norm(A_m[:3, 0])), np.linalg.inv(A_m)
    mri, shape = (m_arr, m_mask, A_m), tuple(m_arr.shape)
    mri1 = lambda X: G.apply_affine(inv_m, X)[:, 1] * h_m                       # MRI axis-1 position, mm from the crop start
    keep_pts = lambda X: mri1(X) >= args.folia_mm
    print(f"[{time.time() - t0:.0f} s] grids ready; MRI base {shape}, OCT base {o_arr.shape}", flush=True)

    # sectioning axis and the coordinate s along it (MRI world)
    axis, scores = stripe_axis(o_arr, o_mask)
    d = T[:3, :3] @ (A_o[:3, axis] / np.linalg.norm(A_o[:3, axis]))
    d /= np.linalg.norm(d)
    idx = np.argwhere(o_mask)[:: max(1, int(o_mask.sum() // 200000))]
    s_mask = G.apply_affine(T @ A_o, idx.astype(float)) @ d
    s0, s1 = float(s_mask.min()), float(s_mask.max())
    print(f"section-stripe scores per OCT array axis {np.round(scores, 3).tolist()} -> sectioning axis {axis}; direction in MRI world "
          f"{np.round(d, 3).tolist()} (cos with MRI axes {np.round(np.abs(d), 3).tolist()}); mask extent along it {s1 - s0:.1f} mm", flush=True)
    edges = np.arange(0, (s1 - s0) + 2.0, 2.0)
    zone = lambda s: np.where((s - s0 < 4.0) | (s1 - s < 4.0), "end", "middle")

    summary = {"sectioning_axis": axis, "stripe_scores": scores, "direction_mri_world": d.tolist(), "extent_mm": s1 - s0,
               "folia_exclusion_mri_axis1_mm": args.folia_mm, "main_run": str(args.main), "variants": {}}

    for name in names:
        P = replace(P0, **VARIANTS[name])
        tv = time.time()
        with torch.no_grad():
            evidence = D.Evidence(mri, P, dev)
            src = _on_grid(torch.stack([to_torch(o_arr, dev), to_torch(o_mask, dev)]), A_o, shape, A_m, np.linalg.inv(T))
            inside0 = (src[1] > D.INSIDE).to(torch.float32)
            before = evidence.measure(src[0], inside0)
            diag = None
            if name == "base":
                feat_o = structure_feature(src[0], inside0, evidence.h, P.df_sigma_mm, dev)
                diag = match_diag(evidence.feat_m, feat_o, evidence.core_m * _erode(inside0, evidence.erode), evidence.h,
                                  block_mm=P.df_block_mm, step_mm=P.df_step_mm, range_mm=P.df_range_mm, z_min=P.df_z_min)
        corners = G.apply_affine(A_m, np.array([[i, j, k] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1)
                                                for k in (0, shape[2] - 1)], float))
        lattice = D.Lattice(corners.min(0), corners.max(0), P.df_grid_mm)
        row = {"params": VARIANTS[name], "n_interior": len(before["W"]), "n_boundary": len(before["P"])}
        best, table, after = None, [], None
        if row["n_interior"] >= P.df_min_interior and row["n_boundary"] >= P.df_min_boundary:
            none = [float(np.median(np.linalg.norm(before["D"], axis=1))), float(np.median(np.abs(before["delta"])))]
            best, table = D.choose(lattice, before, sum(none), P)
            row["cv_none"] = none
        row["candidates"] = table
        if best is not None:
            c = D.fit(lattice, before, best[0], P.df_huber_mm)
            field = lattice.on_grid(c, shape, A_m, dev)
            with torch.no_grad():
                moved = D.warp(src, A_m, field)
                after = evidence.measure(moved[0], (moved[1] > D.INSIDE).to(torch.float32))
            fmag = np.linalg.norm(field.cpu().numpy(), axis=0)[m_mask]
            row.update(status="applied", lam=best[0], cv_chosen=best[1], max_strain=lattice.strain(c),
                       field_mm={"median": float(np.median(fmag)), "p95": float(np.percentile(fmag, 95)), "max": float(fmag.max())})
            # where the strain binds: largest first differences of the lattice displacements and their positions
            g3 = c.reshape(3, *lattice.n)
            tops = []
            for ax in range(3):
                dv = np.abs(np.diff(g3, axis=ax + 1)).max(axis=0) / lattice.spacing
                for flat in np.argsort(dv.ravel())[::-1][:3]:
                    ix = np.array(np.unravel_index(flat, dv.shape), float)
                    ix[ax] += 0.5
                    pos = lattice.lo + lattice.spacing * ix
                    tops.append({"strain": float(dv.ravel()[flat]), "lattice_axis": ax, "s_from_end0_mm": float(pos @ d - s0),
                                 "mri_axis1_mm": float(mri1(pos[None])[0])})
            row["strain_top"] = sorted(tops, key=lambda x: -x["strain"])[:6]
            row["independent_F"] = {"affine": f_vs_chance(mri, src[0].cpu().numpy(), (src[1] > D.INSIDE).cpu().numpy(), A_m, P, dev, keep_pts),
                                    "with_field": f_vs_chance(mri, moved[0].cpu().numpy(), (moved[1] > D.INSIDE).cpu().numpy(), A_m, P, dev, keep_pts)}
        else:
            row["status"] = "not_supported"
        row["readouts"] = {k: [before[k], (after or before)[k]] for k in ("interior_mm", "boundary_mm", "within")}
        if after is not None:
            for kind, key, vals in (("interior", "W", lambda e: np.linalg.norm(e["D"], axis=1)), ("boundary", "P", lambda e: np.abs(e["delta"]))):
                zones = {}
                for tag, e in (("before", before), ("after", after)):
                    X, v = e[key], vals(e)
                    ok = keep_pts(X)
                    sX = X @ d
                    zz = zone(sX)
                    for zn in ("end", "middle"):
                        sel = ok & (zz == zn)
                        zones.setdefault(zn, {})[tag] = {"n": int(sel.sum()), "median_mm": float(np.median(v[sel])) if sel.sum() else None}
                row.setdefault("zone_residuals", {})[kind] = zones
        row["seconds"] = time.time() - tv
        summary["variants"][name] = row
        ro = row["readouts"]
        print(f"[{time.time() - t0:.0f} s] {name:18s} {row['status']:13s} lam {row.get('lam')} strain {row.get('max_strain')} | interior {ro['interior_mm'][0]:.3f}->"
              f"{ro['interior_mm'][1]:.3f} boundary {ro['boundary_mm'][0]:.3f}->{ro['boundary_mm'][1]:.3f} | n {row['n_interior']}/{row['n_boundary']}"
              + (f" | F/chance {row['independent_F']['affine']['excl_folia']['ratio']:.3f}->{row['independent_F']['with_field']['excl_folia']['ratio']:.3f}"
                 if 'independent_F' in row else "") + f" ({row['seconds']:.0f} s)", flush=True)

        if name == "base" and diag is not None:
            cen = G.apply_affine(A_m, diag["centre"])
            sc, ok = cen @ d, keep_pts(cen)
            tab = {"edges_from_end0_mm": edges.tolist()}
            for tag, sel in (("blocks", np.ones(len(sc), bool)), ("confident", diag["confident"]), ("border_limited", diag["border"] & ~diag["confident"]),
                             ("low_z", ~diag["border"] & ~diag["confident"])):
                tab[tag] = [int(((sc - s0 >= lo) & (sc - s0 < hi) & sel & ok).sum()) for lo, hi in zip(edges[:-1], edges[1:])]
            summary["block_status_along_sectioning_axis"] = tab
            summary["block_status_totals_excl_folia"] = {k: int(sum(v)) for k, v in tab.items() if k != "edges_from_end0_mm"}
            sB = before["P"] @ d
            summary["boundary_points_along_axis"] = {"count": [int((((sB - s0 >= lo) & (sB - s0 < hi)) & keep_pts(before["P"])).sum()) for lo, hi in zip(edges[:-1], edges[1:])]}
            if after is not None:
                sBa = after["P"] @ d
                prof = {}
                for tag, e, sW, sP in (("before", before, before["W"] @ d, sB), ("after", after, after["W"] @ d, sBa)):
                    prof[tag] = {"interior": binned(sW - s0, np.linalg.norm(e["D"], axis=1), edges),
                                 "boundary": binned(sP - s0, np.abs(e["delta"]), edges)}
                summary["residual_profile_along_axis"] = {"edges": edges.tolist(), **prof}
        io.write_json(summary, out / "deform_variants.json")
    print(f"[{time.time() - t0:.0f} s] wrote {out / 'deform_variants.json'}", flush=True)


if __name__ == "__main__":
    main()

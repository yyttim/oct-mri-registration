#!/usr/bin/env python3
"""Prototype: multi-scale interior block matching for §6, to test whether the block-end evidence gap (ablate_deform.py) is
partly mechanical. octreg.blockmatch.match only tries a block (edge df_block_mm=4.5 mm) when >=70% of it sits in the eroded
core (CORE_FRACTION); within ~4 mm of a true tissue tip a 4.5 mm block usually fails that test before texture quality is
even considered. This script re-tries the failed blocks at smaller sizes (3.0, 2.0 mm), keeps only the extra confident
matches that are not already covered by a baseline (4.5 mm) match nearby, and merges them into octreg's own
Evidence/Lattice/choose/fit/warp exactly like ablate_deform.py. Nothing in octreg is modified; this only adds evidence
before D.fit sees it.

    python bench/ablate_deform_multiscale.py --out OUT --main RUN [--extra-blocks 3.0,2.0] [--extra-z-min 4.0] [--rescue-dist-mm 2.5]

Result (bench/results/deform_ablation.md): even at a z-threshold matched to the baseline's own (so the comparison is
apples-to-apples, not an artificially strict bar on the rescue), smaller blocks rescue only a handful of matches right at
the true ends, and using them does not improve the end-zone residual. The block ends are close to featureless at the
interior-texture scale -- not a block-size or threshold artifact.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate as E                                                    # noqa: E402  (bench/evaluate.py)
from ablate import prepare                                              # noqa: E402  (bench/ablate.py: the §1 prep step)
from deform_common import f_vs_chance, load_grids, stripe_axis          # noqa: E402
from octreg import deform as D                                          # noqa: E402
from octreg import geometry as G, io                                    # noqa: E402
from octreg.blockmatch import _erode, _on_grid, match, structure_feature  # noqa: E402
from octreg.params import Params                                        # noqa: E402
from octreg.search import to_torch                                      # noqa: E402


def multiscale_measure(evidence: D.Evidence, vol, inside, extra_blocks, extra_z_min, rescue_dist_mm):
    """Like Evidence.measure, but the interior matches are the union of the baseline block size (P.df_block_mm) and smaller
    sizes (extra_blocks, mm), each re-tried at extra_z_min and kept only where it is farther than rescue_dist_mm from every
    already-kept match centre. -> (dict like Evidence.measure, per-scale match counts)."""
    P, dev = evidence.P, evidence.device
    core = evidence.core_m * _erode(inside, evidence.erode)
    feat_o = structure_feature(vol, inside, evidence.h, P.df_sigma_mm, dev)
    counts = {}
    c0, d0, *_ = match(evidence.feat_m, feat_o, core, evidence.h, block_mm=P.df_block_mm, step_mm=P.df_step_mm,
                        range_mm=P.df_range_mm, z_min=P.df_z_min)
    W = [G.apply_affine(evidence.A, c0)]
    Dsp = [d0 @ evidence.A[:3, :3].T]
    counts[P.df_block_mm] = len(c0)
    for B in extra_blocks:
        c, d, *_ = match(evidence.feat_m, feat_o, core, evidence.h, block_mm=B, step_mm=P.df_step_mm,
                          range_mm=P.df_range_mm, z_min=extra_z_min)
        Wc = G.apply_affine(evidence.A, c)
        if len(Wc) and len(np.concatenate(W)):
            keep = cKDTree(np.concatenate(W)).query(Wc)[0] >= rescue_dist_mm
        else:
            keep = np.ones(len(Wc), bool)
        counts[B] = int(keep.sum())
        W.append(Wc[keep])
        Dsp.append((d @ evidence.A[:3, :3].T)[keep])
    Wa, Da = np.concatenate(W), np.concatenate(Dsp)
    length = np.linalg.norm(Da, axis=1)
    Wa, Da = Wa[length <= P.df_range_mm], Da[length <= P.df_range_mm]
    Pb, n, delta = evidence.boundary(vol)
    off = np.abs(delta)
    near = cKDTree(Wa).query(Pb)[0] <= P.df_support_mm if len(Wa) and len(Pb) else np.zeros(len(Pb), bool)
    return {"W": Wa, "D": Da, "P": Pb[near], "n": n[near], "delta": delta[near],
            "interior_mm": float(np.median(length)) if len(length) else None,
            "boundary_mm": float(np.median(off)) if len(off) else None,
            "within": float((off < 0.3).mean()) if len(off) else None}, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--main", type=Path, required=True, help="CLI run dir (T_oct2mri.txt) whose pose the field is fitted on")
    ap.add_argument("--extra-blocks", default="3.0,2.0")
    ap.add_argument("--extra-z-min", type=float, default=Params().df_z_min,
                     help="z-score bar for the rescue blocks; default matches P.df_z_min so the comparison to baseline is apples-to-apples "
                          "(try something higher, e.g. 5.5, to see how much of the rescue is noise)")
    ap.add_argument("--rescue-dist-mm", type=float, default=2.5)
    ap.add_argument("--folia-mm", type=float, default=10.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    extra_blocks = [float(x) for x in args.extra_blocks.split(",") if x]

    out, dev = args.out, args.device
    P = Params()
    t0 = time.time()
    prepare(out, ["texture"])
    (m_arr, m_mask, A_m), (o_arr, o_mask, A_o) = load_grids(out)
    T = np.asarray(E.load_T(args.main / "T_oct2mri.txt"), float)
    h_m, inv_m = float(np.linalg.norm(A_m[:3, 0])), np.linalg.inv(A_m)
    mri, shape = (m_arr, m_mask, A_m), tuple(m_arr.shape)
    mri1 = lambda X: G.apply_affine(inv_m, X)[:, 1] * h_m
    keep_pts = lambda X: mri1(X) >= args.folia_mm
    print(f"[{time.time() - t0:.0f} s] grids ready; extra blocks {extra_blocks} mm, extra z_min {args.extra_z_min}, "
          f"rescue dist {args.rescue_dist_mm} mm", flush=True)

    axis, scores = stripe_axis(o_arr, o_mask)
    d = T[:3, :3] @ (A_o[:3, axis] / np.linalg.norm(A_o[:3, axis]))
    d /= np.linalg.norm(d)
    idx = np.argwhere(o_mask)[:: max(1, int(o_mask.sum() // 200000))]
    s_mask = G.apply_affine(T @ A_o, idx.astype(float)) @ d
    s0, s1 = float(s_mask.min()), float(s_mask.max())
    edges = np.arange(0, (s1 - s0) + 2.0, 2.0)
    zone = lambda s: np.where((s - s0 < 4.0) | (s1 - s < 4.0), "end", "middle")
    print(f"sectioning axis {axis}, extent {s1 - s0:.1f} mm", flush=True)

    with torch.no_grad():
        evidence = D.Evidence(mri, P, dev)
        src = _on_grid(torch.stack([to_torch(o_arr, dev), to_torch(o_mask, dev)]), A_o, shape, A_m, np.linalg.inv(T))
        inside0 = (src[1] > D.INSIDE).to(torch.float32)
        before_base = evidence.measure(src[0], inside0)
        before_ms, counts_before = multiscale_measure(evidence, src[0], inside0, extra_blocks, args.extra_z_min, args.rescue_dist_mm)
    print(f"baseline (4.5 mm only): {len(before_base['W'])} interior / {len(before_base['P'])} boundary", flush=True)
    print(f"multi-scale: {counts_before} -> {len(before_ms['W'])} interior / {len(before_ms['P'])} boundary "
          f"(+{len(before_ms['W']) - len(before_base['W'])} interior, +{len(before_ms['P']) - len(before_base['P'])} boundary)", flush=True)

    def zone_report(tag, matches_W):
        sc, ok = matches_W @ d, keep_pts(matches_W)
        end_n = int((((sc - s0 < 4.0) | (s1 - sc < 4.0)) & ok).sum())
        print(f"  {tag}: {end_n} matches in the 4 mm end zones (excl. folia)", flush=True)
        return end_n

    zone_report("baseline", before_base["W"])
    zone_report("multi-scale", before_ms["W"])

    corners = G.apply_affine(A_m, np.array([[i, j, k] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1)
                                            for k in (0, shape[2] - 1)], float))
    lattice = D.Lattice(corners.min(0), corners.max(0), P.df_grid_mm)

    summary = {"sectioning_axis": axis, "extent_mm": s1 - s0, "extra_blocks_mm": extra_blocks,
               "extra_z_min": args.extra_z_min, "rescue_dist_mm": args.rescue_dist_mm, "main_run": str(args.main), "variants": {}}

    for tag, before in (("base", before_base), ("multiscale", before_ms)):
        tv = time.time()
        row = {"n_interior": len(before["W"]), "n_boundary": len(before["P"])}
        none = [float(np.median(np.linalg.norm(before["D"], axis=1))), float(np.median(np.abs(before["delta"])))]
        row["cv_none"] = none
        best, table = D.choose(lattice, before, sum(none), P)
        row["candidates"] = table
        after = None
        if best is not None:
            c = D.fit(lattice, before, best[0], P.df_huber_mm)
            field = lattice.on_grid(c, shape, A_m, dev)
            with torch.no_grad():
                moved = D.warp(src, A_m, field)
                moved_inside = (moved[1] > D.INSIDE).to(torch.float32)
                after = (multiscale_measure(evidence, moved[0], moved_inside, extra_blocks, args.extra_z_min, args.rescue_dist_mm)[0]
                         if tag == "multiscale" else evidence.measure(moved[0], moved_inside))
            fmag = np.linalg.norm(field.cpu().numpy(), axis=0)[m_mask]
            row.update(status="applied", lam=best[0], cv_chosen=best[1], max_strain=lattice.strain(c),
                       field_mm={"median": float(np.median(fmag)), "p95": float(np.percentile(fmag, 95)), "max": float(fmag.max())})
            row["independent_F"] = {"affine": f_vs_chance(mri, src[0].cpu().numpy(), (src[1] > D.INSIDE).cpu().numpy(), A_m, P, dev, keep_pts),
                                    "with_field": f_vs_chance(mri, moved[0].cpu().numpy(), (moved[1] > D.INSIDE).cpu().numpy(), A_m, P, dev, keep_pts)}
        else:
            row["status"] = "not_supported"
        row["readouts"] = {k: [before[k], (after or before)[k]] for k in ("interior_mm", "boundary_mm", "within")}
        if after is not None:
            for kind, key, vals in (("interior", "W", lambda e: np.linalg.norm(e["D"], axis=1)), ("boundary", "P", lambda e: np.abs(e["delta"]))):
                zones = {}
                for bt, e in (("before", before), ("after", after)):
                    X, v = e[key], vals(e)
                    ok = keep_pts(X)
                    sX = X @ d
                    zz = zone(sX)
                    for zn in ("end", "middle"):
                        sel = ok & (zz == zn)
                        zones.setdefault(zn, {})[bt] = {"n": int(sel.sum()), "median_mm": float(np.median(v[sel])) if sel.sum() else None}
                row.setdefault("zone_residuals", {})[kind] = zones
        row["seconds"] = time.time() - tv
        summary["variants"][tag] = row
        ro = row["readouts"]
        print(f"[{time.time() - t0:.0f} s] {tag:12s} {row['status']:13s} lam {row.get('lam')} strain {row.get('max_strain')} | "
              f"interior {ro['interior_mm'][0]:.3f}->{ro['interior_mm'][1]:.3f} boundary {ro['boundary_mm'][0]:.3f}->{ro['boundary_mm'][1]:.3f} "
              f"| n {row['n_interior']}/{row['n_boundary']}"
              + (f" | F/chance {row['independent_F']['affine']['excl_folia']['ratio']:.3f}->{row['independent_F']['with_field']['excl_folia']['ratio']:.3f}"
                 if 'independent_F' in row else "") + f" ({row['seconds']:.0f} s)", flush=True)
        if row.get("zone_residuals"):
            zr = row["zone_residuals"]
            print(f"    end-zone boundary residual: {zr['boundary']['end']['before']['median_mm']} -> {zr['boundary']['end']['after']['median_mm']} "
                  f"(n={zr['boundary']['end']['after']['n']})", flush=True)
            print(f"    end-zone interior residual: {zr['interior']['end']['before']['median_mm']} -> {zr['interior']['end']['after']['median_mm']} "
                  f"(n={zr['interior']['end']['after']['n']})", flush=True)
        io.write_json(summary, out / "deform_multiscale.json")
    print(f"[{time.time() - t0:.0f} s] wrote {out / 'deform_multiscale.json'}", flush=True)


if __name__ == "__main__":
    main()

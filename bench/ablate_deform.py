"""§6 ablation from cached base grids (bench only; the package never reads this): the table of docs/METHOD.md §6, plus rows
that widen the reach of the local evidence (df_reach_mm, the interior search range and the edge window).

    python bench/ngf_lam.py cache --oct OCT --mri MRI -o CACHE        # §1 once, shared with ngf_lam.py
    python bench/ablate_deform.py --cache CACHE --run RUN -o OUT.json  # writes OUT.json and OUT.md

Every row fits §6 once at the affine of RUN (result.json: T_oct2mri, pose.polarity), changing one element of the stage, and
every row is scored the same way, so the rows compare like for like:
  - the stage's read-outs (interior and boundary residual, fraction of boundary offsets within 0.3 mm) measured on the warped
    OCT by the evidence of the default Params, whatever evidence the row was fitted to;
  - two numbers the fit never sees: F of §5 at sigma 0.3 mm (NGFGrid.F, the warped OCT against the MRI on the MRI base grid),
    and the two-class score of §2 (polarity x S_class) over the core 1.5 mm below both surfaces.
The read-outs of the rows that change the evidence are still in-sample for the default evidence they overlap with; F and the
two-class score are not. Nothing is excluded along the block: the boundary residual is also reported per 4 mm along the
sectioning axis (estimated from the section stripes of the OCT), so the end of the block with the torn folia stays in.

The rows of METHOD.md that the package has no switch for are reproduced here: one kind of evidence only (the other kind's
rows dropped from the fit and from the held-out score), the rim ridge instead of the edge (the OCT boundary taken at the one
prominent maximum of the intensity along the normal, not at its steepest fall, while the MRI side keeps the edge), the
support rule of 1.1 (boundary points only within 5 mm of an interior match) and no Huber re-weighting (a Huber threshold
no residual reaches). The rest are Params overrides.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import cKDTree

from octreg import deform as D
from octreg import geometry as G, preprocess as pp
from octreg.blockmatch import _on_grid
from octreg.ngf import NGFGrid
from octreg.params import Params
from octreg.refine import masked_ncc
from octreg.search import to_torch

ROWS = {    # name: (label as in METHOD.md, change); change None = no field
    "none": ("no deformation", None),
    "method": ("§6", {}),
    "interior_only": ("interior evidence only", {"evidence": "interior"}),
    "boundary_only": ("boundary evidence only", {"evidence": "boundary"}),
    "ridge": ("rim ridge instead of the edge", {"edge": "ridge"}),
    "support": ("boundary points only within 5 mm of a match (the rule of 1.1)", {"support": 5.0}),
    "no_huber": ("no Huber re-weighting", {"params": {"df_huber_mm": 1e9}}),
    "lattice7": ("lattice 7 mm", {"params": {"df_grid_mm": 7.0}}),
    "lattice10": ("lattice 10 mm", {"params": {"df_grid_mm": 10.0}}),
    "strain10": ("strain limit 0.10", {"params": {"df_max_strain": 0.10}}),
    "strain20": ("strain limit 0.20", {"params": {"df_max_strain": 0.20}}),
    "reach20": ("reach 2.0 mm", {"params": {"df_reach_mm": 2.0}}),
    "reach27": ("reach 2.7 mm", {"params": {"df_reach_mm": 2.7}}),
}
CORE_MM = 1.5          # the two-class score is taken this far below both surfaces
BIN_MM = 4.0           # bins of the boundary residual along the sectioning axis


def ridge_offsets(vol, affine, points, normals, params: Params):
    """Like deform.edge_offsets, but the boundary is the one prominent maximum of the intensity along the normal (the ridge of
    the rim), not its steepest fall: a local maximum of the smoothed profile more than df_edge_mad MADs above its median,
    within +-df_reach_mm. -> (used bool [M], positions [M] mm)."""
    P, h = params, float(np.linalg.norm(np.asarray(affine, float)[:3, 0]))
    k = max(2, round((P.df_reach_mm + D.PROFILE_EXTRA_MM) / h))
    t = np.arange(-k, k + 1) * h
    pts = torch.as_tensor(points[:, None, :] + normals[:, None, :] * t[None, :, None], dtype=torch.float32, device=vol.device)
    prof = G.sample_world(vol[None], affine, pts)[0].double().cpu().numpy()
    ok = (prof > 0).mean(1) >= D.MEASURED
    s = ndimage.gaussian_filter1d(prof, D.PROFILE_SIGMA, axis=1)
    base = np.median(s, 1, keepdims=True)
    z = (s - base) / (np.median(np.abs(s - base), 1, keepdims=True) + 1e-12)
    peak = ((s[:, 1:-1] > s[:, :-2]) & (s[:, 1:-1] >= s[:, 2:]) & (z[:, 1:-1] > P.df_edge_mad)
            & (np.abs(t[1:-1]) <= P.df_reach_mm + 1e-9))
    ok &= peak.sum(1) == 1
    j, r = peak.argmax(1) + 1, np.arange(len(prof))
    y0, y1, y2 = s[r, j - 1], s[r, j], s[r, j + 1]
    den = y0 - 2 * y1 + y2
    sub = np.where(den < 0, 0.5 * (y0 - y2) / np.where(den < 0, den, -1.0), 0.0)
    return ok, np.where(ok, t[j] + sub * h, 0.0)


def fit_evidence(mri, vol, inside, P, change, device):
    """The evidence the row fits to (Evidence.measure with the row's change applied)."""
    ev = D.Evidence(mri, P, device)
    if change.get("edge") == "ridge":
        def boundary(vol_):
            ok, pos = ridge_offsets(vol_, ev.A, ev.points, ev.normals, P)
            ok &= ev.edge_m[0]
            return ev.points[ok], ev.normals[ok], (pos - ev.edge_m[1])[ok]
        ev.boundary = boundary
    e = ev.measure(vol, inside)
    if change.get("support"):                                 # the support rule of 1.1: a boundary point needs a match nearby
        near = cKDTree(e["W"]).query(e["P"])[0] <= change["support"] if len(e["W"]) and len(e["P"]) else np.zeros(len(e["P"]), bool)
        e.update(P=e["P"][near], n=e["n"][near], delta=e["delta"][near])
    if change.get("evidence") == "interior":
        e.update(P=np.zeros((0, 3)), n=np.zeros((0, 3)), delta=np.zeros(0))
    elif change.get("evidence") == "boundary":
        e.update(W=np.zeros((0, 3)), D=np.zeros((0, 3)))
    return e


def stripe_axis(arr, mask):
    """OCT array axis of the section stripes: the plane-mean intensity is a sawtooth along it."""
    scores = []
    for a in range(3):
        other = tuple(x for x in range(3) if x != a)
        cnt = mask.sum(axis=other)
        prof = ((arr * mask).sum(axis=other) / np.maximum(cnt, 1))[cnt > 50]
        scores.append(float(np.std(np.diff(prof)) / (np.std(prof) + 1e-9)))
    return int(np.argmax(scores))


class Scorer:
    """The same measurement for every row: default-Params evidence, F at sigma 0.3 mm, the two-class score over the core."""

    def __init__(self, mri, polarity, s_of, edges, device):
        self.mri, self.polarity, self.s_of, self.edges, self.device = mri, polarity, s_of, edges, device
        self.P = Params()
        self.ev = D.Evidence(mri, self.P, device)
        m_arr, m_mask, A_m = mri
        self.h = float(np.linalg.norm(np.asarray(A_m)[:3, 0]))
        self.v = pp.mri_channels(pp.two_class(m_arr, m_mask, self.h, self.P), m_mask)
        self.depth_m = ndimage.distance_transform_edt(m_mask) * self.h

    def __call__(self, vol, mask):
        """vol, mask: the (warped) OCT and its specimen mask on the MRI base grid (torch)."""
        inside = (mask > D.INSIDE).to(torch.float32)
        with torch.no_grad():
            e = self.ev.measure(vol, inside)
        arr, msk = vol.cpu().numpy(), inside.cpu().numpy() > 0
        with torch.no_grad():
            F = float(NGFGrid(self.mri, (arr, msk, self.mri[2]), 0.3, self.P, self.device).F(torch.eye(4, device=self.device)))
        u, _ = pp.oct_channels(pp.two_class(arr, msk, self.h, self.P), msk)
        core = (self.depth_m >= CORE_MM) & (ndimage.distance_transform_edt(msk) * self.h >= CORE_MM)
        s_class = float(masked_ncc(torch.as_tensor(self.v[:, core]), torch.as_tensor(u[:, core]),
                                   torch.ones(int(core.sum()), dtype=torch.float32)))
        with torch.no_grad():
            Pb, _, delta = self.ev.boundary(vol)              # every boundary point with one edge, as boundary_mm
        off, sP = np.abs(delta), self.s_of(Pb)
        prof = []
        for lo, hi in zip(self.edges[:-1], self.edges[1:]):
            sel = (sP >= lo) & (sP < hi)
            prof.append(float(np.median(off[sel])) if sel.sum() >= 20 else None)
        return {"interior_mm": e["interior_mm"], "boundary_mm": e["boundary_mm"], "within": e["within"], "F": F,
                "two_class_core": self.polarity * s_class, "core_voxels": int(core.sum()), "boundary_along_axis_mm": prof}


def run_row(name, mri, src, scorer, lattice_for, device):
    label, change = ROWS[name]
    t0 = time.time()
    row = {"label": label, "change": change}
    if change is None:
        row.update(scorer(src[0], src[1]), lam=None, max_strain=None, field_mm={"median": 0.0, "max": 0.0})
        row["seconds"] = time.time() - t0
        return row
    P = replace(Params(), **change.get("params", {}))
    inside0 = (src[1] > D.INSIDE).to(torch.float32)
    with torch.no_grad():
        ev = fit_evidence(mri, src[0], inside0, P, change, device)
    lattice = lattice_for(P)
    kinds = change.get("evidence")
    row["n_interior"], row["n_boundary"] = len(ev["W"]), len(ev["P"])
    enough = ((kinds == "boundary" or len(ev["W"]) >= D.MIN_INTERIOR)
              and (kinds == "interior" or len(ev["P"]) >= D.MIN_BOUNDARY))
    best, score = None, None
    if enough:
        none = [float(np.median(np.linalg.norm(ev["D"], axis=1))) if len(ev["D"]) else 0.0,
                float(np.median(np.abs(ev["delta"]))) if len(ev["delta"]) else 0.0]
        best, score = D.choose(lattice, ev, sum(none), P)
        row["cv_none"] = none
    row["cv_field"] = score
    if best is None:
        row.update(status="not_supported", **scorer(src[0], src[1]), lam=None, max_strain=0.0,
                   field_mm={"median": 0.0, "max": 0.0})
    else:
        c = D.fit(lattice, ev, best[0], P.df_huber_mm)
        field = lattice.on_grid(c, src.shape[1:], mri[2], device)
        with torch.no_grad():
            moved = D.warp(src, mri[2], field)
        mag = np.linalg.norm(field.cpu().numpy(), axis=0)[np.asarray(mri[1], bool)]
        row.update(status="applied", lam=best[0], max_strain=lattice.strain(c),
                   field_mm={"median": float(np.median(mag)), "max": float(mag.max())}, **scorer(moved[0], moved[1]))
    row["seconds"] = time.time() - t0
    return row


def fmt(x, spec):
    return "" if x is None else format(x, spec)


def markdown(res):
    lines = ["| variant | λ | strain | interior (mm) | boundary (mm) | within 0.3 mm | held-out interior / boundary (mm) | F | "
             "two-class, core | field median / max (mm) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"].values():
        fm = r["field_mm"]
        field = "0" if not fm["max"] else f"{fm['median']:.2f} / {fm['max']:.2f}"
        cv = r.get("cv_field")
        cv = "" if cv is None else " / ".join(f"{x:.3f}" for x in cv)
        lines.append(f"| {r['label']} | {fmt(r['lam'], '.2f')} | {fmt(r['max_strain'], '.3f')} | {r['interior_mm']:.3f} | "
                     f"{r['boundary_mm']:.3f} | {100 * r['within']:.0f} % | {cv} | {r['F']:.4f} | {r['two_class_core']:.4f} | "
                     f"{field} |")
    e = res["axis"]["edges_mm"]
    lines += ["", f"Boundary residual (median |offset|, mm) per {BIN_MM:g} mm along the sectioning axis, from end 0 "
              f"({res['axis']['extent_mm']:.1f} mm in all; end 0 is the end at the start of MRI array axis "
              f"{res['axis']['mri_axis']}):", "",
              "| variant | " + " | ".join(f"{a:.0f}-{b:.0f}" for a, b in zip(e[:-1], e[1:])) + " |",
              "|---|" + "---|" * (len(e) - 1)]
    for r in res["rows"].values():
        lines.append(f"| {r['label']} | " + " | ".join(fmt(x, ".3f") for x in r["boundary_along_axis_mm"]) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="base grids of bench/ngf_lam.py cache")
    ap.add_argument("--run", required=True, help="an octreg run directory (result.json)")
    ap.add_argument("--only", default=None, help=f"comma-separated subset of {','.join(ROWS)}")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("-o", "--out", default="deform_ablation.json")
    a = ap.parse_args()
    names = a.only.split(",") if a.only else list(ROWS)
    if set(names) - set(ROWS):
        ap.error(f"unknown rows {sorted(set(names) - set(ROWS))}")
    t0 = time.time()
    z = np.load(a.cache)
    mri = (z["mri_arr"], z["mri_mask"], z["mri_affine"])
    o_arr, o_mask, A_o = z["oct_arr"], z["oct_mask"], z["oct_affine"]
    res_run = json.loads((Path(a.run) / "result.json").read_text())
    T, polarity = np.asarray(res_run["T_oct2mri"], float), int(res_run["pose"]["polarity"])
    A_m, shape = np.asarray(mri[2], float), tuple(mri[0].shape)
    with torch.no_grad():
        src = _on_grid(torch.stack([to_torch(o_arr, a.device), to_torch(o_mask, a.device)]), A_o, shape, A_m, np.linalg.inv(T))

    axis = stripe_axis(o_arr, o_mask)                       # the sectioning axis, as a unit vector in MRI world
    d = T[:3, :3] @ A_o[:3, axis]
    d /= np.linalg.norm(d)
    idx = np.argwhere(o_mask)[:: max(1, int(o_mask.sum() // 200000))]
    s = G.apply_affine(T @ A_o, idx.astype(float)) @ d
    s0, s1 = float(s.min()), float(s.max())
    mri_axis = int(np.argmax(np.abs(np.linalg.inv(A_m[:3, :3]) @ d)))
    if (np.linalg.inv(A_m[:3, :3]) @ d)[mri_axis] < 0:     # end 0 at the start of that MRI array axis
        d, (s0, s1) = -d, (-s1, -s0)
    edges = np.arange(0.0, s1 - s0 + BIN_MM, BIN_MM)
    s_of = lambda X: np.asarray(X, float).reshape(-1, 3) @ d - s0
    scorer = Scorer(mri, polarity, s_of, edges, a.device)
    corners = G.apply_affine(A_m, np.array([[i, j, k] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1)
                                            for k in (0, shape[2] - 1)], float))
    lattice_for = lambda P: D.Lattice(corners.min(0), corners.max(0), P.df_grid_mm)
    print(f"[{time.time() - t0:.0f} s] grids {shape}; sectioning axis: OCT array axis {axis} ~ MRI array axis {mri_axis}, "
          f"{s1 - s0:.1f} mm", flush=True)

    out = Path(a.out)
    res = {"run": str(a.run), "cache": str(a.cache), "params_hash": Params().hash(), "polarity": polarity,
           "axis": {"oct_axis": axis, "mri_axis": mri_axis, "direction_mri_world": d.tolist(), "extent_mm": s1 - s0,
                    "edges_mm": edges.tolist()}, "rows": {}}
    for name in names:
        r = res["rows"][name] = run_row(name, mri, src, scorer, lattice_for, a.device)
        print(f"[{time.time() - t0:.0f} s] {r['label']:32s} λ {fmt(r['lam'], '.2f'):>5s} strain {fmt(r['max_strain'], '.3f'):>5s} | "
              f"interior {r['interior_mm']:.3f} boundary {r['boundary_mm']:.3f} within {100 * r['within']:.0f} % | "
              f"F {r['F']:.4f} two-class {r['two_class_core']:.4f}", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1))
    out.with_suffix(".md").write_text(markdown(res), encoding="utf-8")
    print(f"wrote {out} and {out.with_suffix('.md')} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()

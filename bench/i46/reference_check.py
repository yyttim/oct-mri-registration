#!/usr/bin/env python3
"""Checks of the I46 reference R (bench/i46/README.md), on the inputs of prepare.py -> bench/results/I46/reference_check.json.

lattice  the steps between the 16 lattice centres of the authors' OCT -> LSFM fields, against 33 planes of 12 um
frame    the 96 choices of the rebuilt OCT header (depth sign, slab, order and signs of the plane axes), each scored by
         the correlation of the OCT with each FLASH image and the share of OCT vessels within 0.15 mm of an MRI vessel label
labels   the label-only fits S9 and A12 of labelfit.py from R, compared with R, by thirds of the depth and on split halves
search   the global search of labelfit.py with no transform of the authors, about an hour on 12 processes
vessels  the vessel check of common.vessel_score at R and at R shifted by 0.5 and 1 mm in 12 random directions

    python bench/i46/reference_check.py [--jobs N] [--only STAGE ...]

--only runs the named stages and keeps the others from an existing reference_check.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labelfit as lf  # noqa: E402
from common import (AUTHORS, CHOSEN, FLIPS, LABELS, OCT_TIFF, RESULTS, SLAB, SPACING, TIFF_SHAPE, VESSELS_TIFF,  # noqa: E402
                    error, lta_matrix, mri_file, oct_header, reference, slab_lattices, tissue, vessel_score)

DS = 8                                      # OCT voxels per block and axis for the correlation
OCT_BLOCKS, VESSEL_VOXELS = 120_000, 100_000
NEAR_VESSEL = 0.15                          # mm
LABEL_BOX = 1.0                             # mm around the MRI vessel labels
SPLITS = 10
ROTATIONS, PEAKS, CANDIDATES = 20_000, 3, 40
DISTINCT = 1.5                              # mm
SHIFTS, DIRECTIONS = (0.5, 1.0), 12
ORIENTATION = ("depth_sign", "swap", "sign_y", "sign_x")
STAGES = ("lattice", "frame", "labels", "search", "vessels")


def r4(x):
    return None if x is None else round(float(x), 4)


def mean_distance(T1, T2, pts):
    return float(np.linalg.norm(lf.apply(T1, pts) - lf.apply(T2, pts), axis=1).mean())


def compare(T, R):
    e = error(T, R)
    return {k: e[k] for k in ("mean_mm", "median_mm", "p95_mm", "max_mm", "rotation_deg")}


def lattice_steps():
    """Steps between consecutive slab centres along the depth axis, and the offset of their mean from equal steps."""
    centres, ax = slab_lattices()
    p = centres @ -ax[:, 2]
    step, n = SLAB * 0.012, len(p)
    first, last = p[0] + step * np.arange(n), p[-1] - step * np.arange(n)[::-1]
    return {"steps_mm": [r4(x) for x in np.diff(p)], "plane_step_mm": round(step, 4),
            "mean_offset_mm": {"equal_steps_from_first": r4(p.mean() - first.mean()),
                               "equal_steps_from_last": r4(p.mean() - last.mean())}}


def headers():
    n = TIFF_SHAPE[0] // SLAB - 16                         # 3 spare OCT slabs
    out = []
    for sign in (1, -1):
        for first in range(-1, n + 2):                     # lowest OCT slab of the 16 LSFM slabs
            for swap, sy, sx in itertools.product((True, False), (-1, 1), (-1, 1)):
                out.append({"depth_sign": sign, "slab08": first if sign == 1 else first + 15, "swap": swap, "sign_y": sy,
                            "sign_x": sx})
    return out


def oct_blocks():
    """OCT_BLOCKS random DS^3 OCT blocks at least 90 % non-zero: TIFF index of their centres and their mean intensity."""
    import tifffile
    n = np.array(TIFF_SHAPE) // DS
    val, frac = np.zeros(n), np.zeros(n)
    with tifffile.TiffFile(str(OCT_TIFF)) as tf:
        for b in range(n[0]):
            a = np.stack([tf.pages[z].asarray()[:n[1] * DS, :n[2] * DS] for z in range(b * DS, (b + 1) * DS)]).astype(float)
            nz = a > 0
            med = np.array([np.median(p[m]) if m.sum() > 1000 else 1.0 for p, m in zip(a, nz)])
            a /= med[:, None, None]                 # the OCT intensity falls with depth and changes by section
            val[b] = a.reshape(DS, n[1], DS, n[2], DS).mean((0, 2, 4))
            frac[b] = nz.reshape(DS, n[1], DS, n[2], DS).mean((0, 2, 4))
    idx = np.argwhere(frac > 0.9)
    idx = idx[np.random.default_rng(0).choice(len(idx), OCT_BLOCKS, replace=False)]
    return idx * DS + (DS - 1) / 2, val[tuple(idx.T)]


def vessel_voxels():
    """VESSEL_VOXELS random voxels of the OCT vessel segmentation, TIFF index (plane, y, x)."""
    import tifffile
    v = tifffile.imread(str(VESSELS_TIFF)) > 0
    i = np.flatnonzero(v)
    i = np.sort(i[np.random.default_rng(1).choice(len(i), VESSEL_VOXELS, replace=False)])
    return np.array(np.unravel_index(i, v.shape), float).T


def frame_check(box, box_aff):
    from scipy import ndimage
    f1 = nib.load(str(mri_file("flip1")))
    shape = np.array(f1.shape[:3])
    chain = lta_matrix(AUTHORS / "mri/recon2raw.lta") @ lta_matrix(AUTHORS / "mri2oct/affine.lta")
    hs = headers()
    Ms = [oct_header(**h) for h in hs]
    to_vox = [np.linalg.inv(f1.affine) @ chain @ M for M in Ms]
    n = np.array(TIFF_SHAPE) - 1
    corners = np.array(list(itertools.product((0, n[0]), (0, n[1]), (0, n[2]))), float)
    q = np.concatenate([lf.apply(T, corners) for T in to_vox])
    lo = np.maximum(np.floor(q.min(0)).astype(int) - 2, 0)
    hi = np.minimum(np.ceil(q.max(0)).astype(int) + 3, shape)
    pts, val = oct_blocks()
    ves = vessel_voxels()
    print(f"frame: {len(hs)} headers, {len(pts)} OCT blocks, {len(ves)} vessel voxels, MRI box {lo} {hi}", flush=True)
    edt = ndimage.distance_transform_edt(~box, sampling=np.linalg.norm(box_aff[:3, :3], axis=0)).astype(np.float32)
    to_box = [np.linalg.inv(box_aff) @ chain @ M for M in Ms]
    for h, M, T, Tb in zip(hs, Ms, to_vox, to_box):
        qv = lf.apply(Tb, ves)
        inb = np.all((qv >= 0) & (qv <= np.array(box.shape) - 1), 1)
        d = ndimage.map_coordinates(edt, qv[inb].T, order=1)
        h["vessel_share"] = r4(np.count_nonzero(d <= NEAR_VESSEL) / len(ves))
        q = lf.apply(T, pts)
        h["oct_inside_mri"] = r4(np.all((q >= 0) & (q <= shape - 1), 1).mean())
        h["correlation"] = {}
    for im in FLIPS:
        mri = np.asarray(nib.load(str(mri_file(im))).dataobj[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], dtype=np.float32)
        for h, T in zip(hs, to_vox):
            q = lf.apply(T, pts)
            inside = np.all((q >= 0) & (q <= shape - 1), 1)
            if inside.mean() < 0.5:
                h["correlation"][im] = None
                continue
            m = ndimage.map_coordinates(mri, (q[inside] - lo).T, order=1)
            h["correlation"][im] = r4(np.corrcoef(val[inside], m)[0, 1])
        print(f"frame: {im} done", flush=True)
        del mri
    chosen = next(h for h in hs if all(h[k] == v for k, v in CHOSEN.items()))
    others = [h for h in hs if h is not chosen]
    turned = [h for h in others if any(h[k] != chosen[k] for k in ORIENTATION)]
    slabs = sorted((h for h in others if h not in turned), key=lambda h: h["slab08"])
    corr = {im: max(abs(h["correlation"][im]) for h in turned if h["correlation"][im] is not None) for im in FLIPS}
    share = max(h["vessel_share"] for h in others)
    out = {"headers_scored": len(hs), "oct_blocks": OCT_BLOCKS, "vessel_voxels": VESSEL_VOXELS, "chosen": chosen,
           "other_orientations": {"headers": len(turned), "highest_abs_correlation": corr,
                                  "highest_vessel_share": max(h["vessel_share"] for h in turned)},
           "other_slabs": [{k: h[k] for k in ("slab08", "correlation", "vessel_share")} for h in slabs],
           "others_highest_vessel_share": share,
           "chosen_highest_correlation_over_orientations": {im: bool(chosen["correlation"][im] > corr[im]) for im in FLIPS},
           "chosen_highest_vessel_share": bool(chosen["vessel_share"] > share), "headers": hs}
    print(f"frame: chosen {chosen['correlation']} share {chosen['vessel_share']}, other orientations {corr}, other slabs "
          f"{[(h['slab08'], h['correlation']['flip2'], h['vessel_share']) for h in slabs]}", flush=True)
    return out


def label_box():
    """The MRI vessel labels in their bounding box, LABEL_BOX wider on every face, its voxel -> flip-1 world affine, lo, hi."""
    A = nib.load(str(mri_file("flip1"))).affine
    lab = np.asarray(nib.load(str(LABELS["vessels"])).dataobj) == 255
    v = np.argwhere(lab)
    m = np.ceil(LABEL_BOX / np.linalg.norm(A[:3, :3], axis=0)).astype(int)
    lo, hi = np.maximum(v.min(0) - m, 0), np.minimum(v.max(0) + m + 1, lab.shape)
    box = lab[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].copy()
    aff = A.copy()
    aff[:3, 3] = A[:3, :3] @ lo + A[:3, 3]
    return box, aff, lo, hi


G = {}


def _init(data):
    G["data"] = data
    G["P"] = lf.Problem(data["pts"], data["foot"], lf.cell_affine(SPACING), data["box"], data["box_aff"], lf.CELL * SPACING)


def _fit(job):
    """job: key, start, model, c stages, side of a split half (None, "mri" or "oct") and the components of the half."""
    key, T0, model, cs, side, half = job
    P = G["P"]
    kw = {}
    if side is not None:
        kw = {"idxM" if side == "mri" else "idxO": np.flatnonzero(half[P.mlab if side == "mri" else P.olab])}
    T, info = lf.fit(P, T0, model, cs=cs, **kw)
    return key, T, info


def _refine(job):
    key, T0 = job
    Ts, ds = lf.fit(G["P"], T0, "S9", cs=lf.C_START)
    Ta, da = lf.fit(G["P"], Ts, "A12", cs=lf.C_A12)
    return key, Ts, ds, Ta, da


def _search(job):
    j, L = job
    if "S" not in G:
        d = G["data"]
        G["S"] = lf.Search(d["box"], np.linalg.norm(d["box_aff"][:3, :3], axis=0), d["B"])
    d, S = G["data"], G["S"]
    ncc, tlo = S.run(d["pts"], d["F"], L, d["cen"], d["fcell"])
    return j, [(v, S.pose(L, d["cen"], k, tlo)) for v, k in lf.top_peaks(ncc, PEAKS)]


def by_thirds(T, R, pts):
    e = np.linalg.norm(lf.apply(T, pts) - lf.apply(R, pts), axis=1)
    plane = pts[:, 2] / SPACING[2]
    edges = np.linspace(0, TIFF_SHAPE[0], 4)
    return [{"planes": [int(np.ceil(a)), int(np.ceil(b)) - 1], "points": int(((plane >= a) & (plane < b)).sum()),
             "mean_mm": r4(e[(plane >= a) & (plane < b)].mean())} for a, b in zip(edges[:-1], edges[1:])]


def spacing_um(T):
    """MRI um per OCT voxel along the OCT axes x, y and plane."""
    return (np.linalg.norm(T[:3, :3], axis=0) * SPACING * 1000).round(2).tolist()


def energy_info(info):
    return {k: (r4(v) if isinstance(v, float) else v) for k, v in info.items()}


def reference_fits(pool, R):
    (_, Ts, ds), = pool.map(_fit, [("S9", R, "S9", lf.C_LOCAL, None, None)])
    (_, Ta, da), = pool.map(_fit, [("A12", Ts, "A12", lf.C_A12, None, None)])
    print(f"labels: S9 {compare(Ts, R)}, A12 {compare(Ta, R)}", flush=True)
    return Ts, ds, Ta, da


def label_fit(pool, R, Ts, ds, Ta, da, n_mri, n_oct):
    pts = tissue()
    jobs = []
    for (model, T0, cs), (side, n) in itertools.product((("S9", Ts, lf.C_LOCAL), ("A12", Ta, lf.C_A12)),
                                                        (("mri", n_mri), ("oct", n_oct))):
        for k in range(SPLITS):
            half = np.random.default_rng(100 + k).random(n) < 0.5
            jobs += [((model, side, k, h), T0, model, cs, side, half if h == 0 else ~half) for h in (0, 1)]
    fits = {key: T for key, T, _ in pool.imap_unordered(_fit, jobs)}
    split = {}
    for model, side in itertools.product(("S9", "A12"), ("mri", "oct")):
        ab = np.array([mean_distance(fits[(model, side, k, 0)], fits[(model, side, k, 1)], pts) for k in range(SPLITS)])
        split.setdefault(model, {})[side] = {"rms_mm": r4(np.sqrt((ab ** 2).mean())), "median_mm": r4(np.median(ab)),
                                             "max_mm": r4(ab.max()), "splits_mm": [r4(x) for x in ab]}
    print(f"labels: split halves {json.dumps({m: {s: v['rms_mm'] for s, v in x.items()} for m, x in split.items()})}",
          flush=True)
    out = {"reference": {"spacing_um": spacing_um(R)}}
    for name, T, info in (("S9", Ts, ds), ("A12", Ta, da)):
        out[name] = {**compare(T, R), "by_depth_third": by_thirds(T, R, pts), "spacing_um": spacing_um(T),
                     "objective": energy_info(info), "split_half_spread": split[name],
                     "T_oct2mri": np.asarray(T, float).round(9).tolist()}
    out["reference"]["objective"] = energy_info(pool.apply(_energy, (R,)))
    return out


def _energy(T):
    E, d = G["P"].energy(T, lf.C_LOCAL[-1], detail=True)
    return {"E": E, **d}


def _count():
    """The number of components of the MRI labels and of the OCT points, plus one for label 0."""
    return int(G["P"].mlab.max()) + 1, int(G["P"].olab.max()) + 1


def search(pool, data, Ts, Ta):
    t0 = time.time()
    pts = tissue()
    sub = pts[::400]
    W = data["box_aff"] @ np.diag(list(1 / np.linalg.norm(data["box_aff"][:3, :3], axis=0)) + [1.0])
    rots = lf.super_fibonacci(ROTATIONS)
    jobs = [(j, R @ np.diag([1.0, 1.0, hand])) for j, (hand, R) in enumerate(itertools.product((1.0, -1.0), rots))]
    cand = []
    for n, (j, peaks) in enumerate(pool.imap_unordered(_search, jobs, chunksize=20)):
        cand += [(v, W @ pose) for v, pose in peaks]
        if n % 5000 == 0:
            print(f"search: {n} of {len(jobs)} rotations, {time.time() - t0:.0f} s", flush=True)
    cand.sort(key=lambda c: -c[0])
    kept = []
    for rank, (v, T) in enumerate(cand):
        if all(mean_distance(T, Tk, sub) > DISTINCT for _, _, Tk in kept):
            kept.append((rank, v, T))
        if len(kept) == CANDIDATES:
            break
    print(f"search: {len(cand)} peaks, {len(kept)} distinct candidates (NCC {kept[0][1]:.3f} to {kept[-1][1]:.3f}), "
          f"{time.time() - t0:.0f} s", flush=True)
    rows = []
    for k, Ts_k, ds, Ta_k, da in pool.imap_unordered(_refine, list(enumerate(T for _, _, T in kept))):
        rank, v, T0 = kept[k]
        rows.append({"candidate": k, "ncc": r4(v), "ncc_rank": rank, "handedness": int(np.sign(np.linalg.det(T0[:3, :3]))),
                     "E_S9": r4(ds["E"]), "plausible": ds["plausible"], "E_A12": r4(da["E"]),
                     "S9_to_reference_mm": r4(mean_distance(Ts_k, reference(), pts)), "T": Ts_k})
    ok = sorted((r for r in rows if r["plausible"] and r["E_S9"] < 9), key=lambda r: r["E_S9"])
    best = ok[0]
    nxt = next((r for r in ok[1:] if mean_distance(r["T"], best["T"], pts) > DISTINCT), None)
    mirrored = next((r for r in ok if r["handedness"] != best["handedness"]), None)
    (_, Tb, db), = pool.map(_fit, [("S9", best["T"], "S9", lf.C_LOCAL, None, None)])
    (_, Tab, dab), = pool.map(_fit, [("A12", Tb, "A12", lf.C_A12, None, None)])
    R = reference()
    out = {"rotations": ROTATIONS, "handedness": [1, -1], "grid_mm": lf.GRID, "peaks": len(cand), "candidates": len(kept),
           "best": {"ncc": best["ncc"], "ncc_rank": best["ncc_rank"], "handedness": best["handedness"],
                    "reference_handedness": int(np.sign(np.linalg.det(R[:3, :3]))), "E_S9": r4(db["E"]), "E_A12": r4(dab["E"]),
                    "S9_to_S9_from_reference_mm": round(mean_distance(Tb, Ts, pts), 5),
                    "A12_to_A12_from_reference_mm": round(mean_distance(Tab, Ta, pts), 5),
                    "S9": compare(Tb, R), "A12": compare(Tab, R)},
           "next_distinct": None if nxt is None else {"E_S9": nxt["E_S9"], "handedness": nxt["handedness"],
                                                      "distance_mm": r4(mean_distance(nxt["T"], best["T"], pts))},
           "best_other_handedness": None if mirrored is None else {"E_S9": mirrored["E_S9"]},
           "seconds": round(time.time() - t0),
           "refined": sorted(({k: v for k, v in r.items() if k != "T"} for r in rows), key=lambda r: r["E_S9"])}
    print(f"search: best E {out['best']['E_S9']}, {out['best']['S9_to_S9_from_reference_mm']} mm from the S9 fit from R, "
          f"next {out['next_distinct']}, other handedness {out['best_other_handedness']}, {out['seconds']} s", flush=True)
    return out


def vessel_shifts(R):
    U = np.random.default_rng(5).normal(size=(DIRECTIONS, 3))
    U /= np.linalg.norm(U, axis=1)[:, None]
    out = {"reference": vessel_score(R), "directions": DIRECTIONS}
    for mag in SHIFTS:
        s = []
        for u in U:
            D = np.eye(4)
            D[:3, 3] = mag * u
            s.append(vessel_score(D @ R))
        out[f"shift_{mag:g}mm"] = {"vessel_mean_mm": r4(np.mean([x["vessel_mean_mm"] for x in s])),
                                   "vessel_median_mm": r4(np.mean([x["vessel_median_mm"] for x in s])),
                                   "per_direction": s}
    print(f"vessels: {out['reference']}, " + ", ".join(f"{k} {v['vessel_mean_mm']} / {v['vessel_median_mm']}"
                                                     for k, v in out.items() if k.startswith("shift")), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=12, help="worker processes (default 12)")
    ap.add_argument("--only", nargs="+", choices=STAGES, metavar="STAGE", help="run only these stages")
    args = ap.parse_args()
    stages = set(args.only or STAGES)
    path = RESULTS / "reference_check.json"
    out = json.loads(path.read_text(encoding="utf-8")) if args.only and path.is_file() else {}
    R = reference()
    t0 = time.time()
    if stages & {"frame", "labels", "search"}:
        box, box_aff, lo, hi = label_box()
    if "lattice" in stages:
        out["lattice"] = lattice_steps()
    if "frame" in stages:
        out["frame"] = {**frame_check(box, box_aff), "seconds": round(time.time() - t0)}
    if stages & {"labels", "search"}:
        t0 = time.time()
        pts, foot = lf.oct_points(VESSELS_TIFF, SPACING)
        print(f"{len(pts)} OCT vessel points, {int(box.sum())} MRI vessel labels in the box {lo.tolist()} {hi.tolist()}",
              flush=True)
        F, fvol = lf.template_cells(foot, SPACING)
        data = {"pts": pts, "foot": foot, "box": box, "box_aff": box_aff, "F": F, "cen": F.mean(0),
                "fcell": fvol / lf.GRID ** 3, "B": int(np.ceil(1.35 * np.linalg.norm(F.max(0) - F.min(0)) / lf.GRID)) + 4}
        with Pool(args.jobs, initializer=_init, initargs=(data,)) as pool:
            Ts, ds, Ta, da = reference_fits(pool, R)
            if "labels" in stages:
                out["labels"] = {**label_fit(pool, R, Ts, ds, Ta, da, *pool.apply(_count)), "oct_points": len(pts),
                                 "mri_labels": int(box.sum()), "label_box_voxels": [lo.tolist(), hi.tolist()],
                                 "seconds": round(time.time() - t0)}
            if "search" in stages:
                out["search"] = search(pool, data, Ts, Ta)
    if "vessels" in stages:
        t0 = time.time()
        out["vessels"] = {**vessel_shifts(R), "seconds": round(time.time() - t0)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: out[k] for k in STAGES if k in out}, indent=1), encoding="utf-8")
    print(f"-> {path}", flush=True)


if __name__ == "__main__":
    main()

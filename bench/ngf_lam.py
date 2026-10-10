"""§5 at several prior weights, from cached base grids (bench only, the package never reads this).

Two steps, so the sweep costs minutes instead of a full run:

    python bench/ngf_lam.py cache --oct OCT --mri MRI -o CACHE      # §1 once (about 4 min on a CPU)
    python bench/ngf_lam.py sweep --cache CACHE --run RUN -o OUT.json

`cache` writes the base grids of §1 (register.prepare_mri, register.prepare_oct) that §5 reads, with the measured-OCT
mask and the OCT header affine the outline read-out needs.
`sweep` starts every fit from the §4 pose of RUN (result.json, pose.T_before_ngf) and reports, per weight, F of §5, the
scales and shears of the fitted pose, how far it lands from the run's own §5 pose and from §4, and the outline agreement
of bench/evaluate.py at that pose, which §5 never sees. Default Params otherwise, so only the varied field moves (lam
reaches §5 alone here, since §4 is not run).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))      # bench/evaluate.py and bench/paths.py

import evaluate
from report import portable                                    # bench/report.py: absolute paths without the machine prefix
from octreg import geometry as G, io, ngf, preprocess as pp, register
from octreg.params import Params
from octreg.refine import decompose
from octreg.search import box


def corners(affine, shape):
    ijk = np.array([[i, j, k] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1) for k in (0, shape[2] - 1)], float)
    return G.apply_affine(np.asarray(affine, float), ijk)


def distance(T1, T2, pts):
    d = np.linalg.norm(G.apply_affine(T1, pts) - G.apply_affine(T2, pts), axis=1)
    return float(d.mean()), float(d.max())


def do_cache(args):
    P = Params()
    h = float(args.grid_mm)
    t0 = time.time()
    vo, vm = io.load_volume(args.oct), io.load_volume(args.mri)
    if h == P.base_mm:
        (mri_h, mask_m, A_m), _ = register.prepare_mri(vm, P)
    else:                                                     # the same §1 on a grid of its own, for §5 alone
        mri_h, A_m = G.resample_iso(vm, h)
        mask_m = pp.foreground(mri_h, h, P)[0]
    print(f"mri grid {mri_h.shape} at {h} mm in {time.time() - t0:.0f} s", flush=True)
    fine, A_f = G.resample_iso(vo, P.fine_mm)
    print(f"oct fine grid {fine.shape} in {time.time() - t0:.0f} s", flush=True)
    if h == P.base_mm:
        (oct_h, mask_o, A_o), valid_o, info = register.prepare_oct(fine, A_f, P)
    else:
        mask_f, info = register.fine_mask(fine, A_f, P)
        oct_h, A_o = G.pool_iso(fine, A_f, h)
        mask_o, valid_o = (G.pool_iso(x.astype(np.float32), A_f, h)[0] > 0.5 for x in (mask_f, fine > 0))
        del mask_f
    del fine
    print(f"oct base grid {oct_h.shape}, mask {info['volume_cm3']:.2f} cm3 in {time.time() - t0:.0f} s", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, mri_arr=mri_h, mri_mask=mask_m, mri_affine=A_m, oct_arr=oct_h, oct_mask=mask_o, oct_affine=A_o,
             oct_valid=valid_o, oct_raw_affine=vo.affine, params_hash=P.hash(), grid_mm=h)
    print(f"wrote {out} ({out.stat().st_size / 2**20:.0f} MiB) in {time.time() - t0:.0f} s")


def do_sweep(args):
    z = np.load(args.cache)
    mri = (z["mri_arr"], z["mri_mask"], z["mri_affine"])
    oct_ = (z["oct_arr"], z["oct_mask"], z["oct_affine"])
    res = json.loads((Path(args.run) / "result.json").read_text(encoding="utf-8"))
    T0, T_run = np.asarray(res["pose"]["T_before_ngf"], float), np.asarray(res["T_oct2mri"], float)
    c = box(oct_[2], oct_[0].shape)[0]
    pts = corners(oct_[2], oct_[0].shape)
    bnd = None
    if "oct_valid" in z and not args.no_boundary:
        t = time.time()
        bnd = evaluate.Boundary(oct_[1], oct_[2], z["oct_valid"], mri[1], mri[2], z["oct_raw_affine"])
        print(f"boundary maps: {bnd.n_oct} OCT and {bnd.n_mri} MRI boundary voxels ({time.time() - t:.0f} s); "
              f"the run's pose: {bnd(T_run)['rim_median_mm']}", flush=True)
    elif not args.no_boundary:
        print("cache has no oct_valid: rebuild it for the outline read-out", flush=True)
    rows = []
    key = {"lam": "lam", "sigmas": "ngf_sigmas_mm"}[args.vary]
    if args.vary == "sigmas":
        settings = [(s, {"ngf_sigmas_mm": [float(x) for x in s.split(",")]}) for s in args.sigmas.split("|")]
    else:
        settings = [(f"{v:g}", {key: v}) for v in (float(x) for x in args.lams.split(","))]
    for label, over in settings:
        P = Params.from_dict(over)
        t = time.time()
        T, info = ngf.refine_ngf(mri, oct_, T0, P, args.device)
        _, _, ls, sh = decompose(T, c)
        row = {"lam": P.lam, "ngf_sigmas_mm": list(P.ngf_sigmas_mm),
               "varied": key, "value": label, "F_start": info["F_start"], "F": info["F"], "L": info["L"],
               "scales": np.exp(ls).tolist(), "shears": sh.tolist(),
               "corners_vs_run_mean_mm": distance(T, T_run, pts)[0], "corners_vs_run_max_mm": distance(T, T_run, pts)[1],
               "corners_vs_affine_mean_mm": distance(T, T0, pts)[0], "seconds": time.time() - t, "T": T.tolist()}
        if bnd is not None:
            row["boundary"] = bnd(T)
            row["rim_median_mm"] = row["boundary"]["rim_median_mm"]
        rows.append(row)
        rim = "" if bnd is None else ("  rim " + " / ".join(f"{x:.3f}" for x in row["rim_median_mm"]) + " mm")
        print(f"{key} {label:>16}: F {info['F_start']:.4f} -> {info['F']:.4f}  scales "
              f"{', '.join(f'{s:.4f}' for s in row['scales'])}  moved {row['corners_vs_affine_mean_mm']:.2f} mm from §4, "
              f"{row['corners_vs_run_mean_mm']:.2f} mm from the run{rim}  ({row['seconds']:.0f} s)", flush=True)
    if args.out:                                              # LF on every OS, the paths cut as bench/results/I58 stores them
        io.write_json({"run": portable(str(args.run)), "cache": portable(str(args.cache)), "rows": rows}, args.out)
        print(f"wrote {args.out}")


def do_transforms(args):
    d = json.loads(Path(args.sweep).read_text(encoding="utf-8"))
    out = Path(args.out or Path(args.sweep).parent)
    out.mkdir(parents=True, exist_ok=True)
    for r in d["rows"]:
        f = out / f"T_{r.get('varied', 'lam')}_{str(r.get('value', r['lam'])).replace(',', '-')}.txt"
        np.savetxt(f, np.asarray(r["T"], float), fmt="%.17g")
        print(f"wrote {f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache", help="write the §1 base grids both volumes share")
    c.add_argument("--oct", required=True)
    c.add_argument("--mri", required=True)
    c.add_argument("-o", "--out", required=True)
    c.add_argument("--grid-mm", type=float, default=0.15, help="isotropic grid of the cached volumes (default: Params.base_mm)")
    s = sub.add_parser("sweep", help="§5 from the §4 pose of a run, once per prior weight")
    s.add_argument("--cache", required=True)
    s.add_argument("--run", required=True, help="an octreg run directory (result.json)")
    s.add_argument("--lams", default="2,1,0.5,0.2,0")
    s.add_argument("--vary", choices=("lam", "sigmas"), default="lam", help="what the sweep varies")
    s.add_argument("--sigmas", default="0.6,0.4,0.3|0.6,0.4,0.3,0.25|0.6,0.4,0.3,0.2|0.45,0.3,0.2",
                   help="--vary sigmas: schedules of ngf_sigmas_mm, one per |")
    s.add_argument("--device", default="cpu")
    s.add_argument("--no-boundary", action="store_true", help="skip the outline read-out of bench/evaluate.py")
    s.add_argument("-o", "--out", default=None)
    w = sub.add_parser("transforms", help="write every pose of a sweep as a 4x4 text file for `octreg qc --T`")
    w.add_argument("--sweep", required=True)
    w.add_argument("-o", "--out", default=None)
    args = ap.parse_args()
    {"cache": do_cache, "sweep": do_sweep, "transforms": do_transforms}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())

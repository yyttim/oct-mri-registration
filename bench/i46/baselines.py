#!/usr/bin/env python3
"""Standard registration tools on the I46 crops (bench/i46/README.md), one documented setting each and no mask.

The settings and commands are in bench/baselines/standard_tools.py. The moving image is the OCT box-averaged to 0.15 mm, the
fixed image the C3 crops k = 0..9 of the 10 deg FLASH and k = 0 of the 20 and 30 deg FLASH. Each option changes the runs on
the 10 deg C3 crops, into <project root>/bench_runs/I46/baselines_<kind>:

--search       the tools' own global searches
--mask         the test's specimen mask as the moving mask of the tools that take one
--near         a start near R as the header of the moving image, each tool started from the header
--near-mask    --near and --mask together, for the tools that take a mask
--near-zeroed  --near on the moving OCT zeroed outside the specimen mask, every tool
--atR          the start at R, its closest shear-free pose, as the header, each tool started from the header
--atR-mask     --atR and --mask together

    python bench/i46/baselines.py prepare [--search | --mask | --near | ...]   the inputs, the run.sh of every run and queue.sh
    bash <project root>/bench_runs/I46/baselines/queue.sh                     one run at a time (baselines_<kind>/queue.sh)
    python bench/i46/baselines.py octreg --near                               octreg's refinement from the start of every crop
    python bench/i46/baselines.py collect [--search | --mask | --near | ...] [--runs DIR] [-o FILE]

collect writes <run>/T_oct2mri.txt, scores it against R and checks the conversion of every run that has the tool's own
resampling. The scores go to bench/results/I46/baselines[_<kind>].json. A run without a readable transform is a tool error.
octreg --near runs bench/ablate.py no_search from the start H_k of each crop (--start) into baselines_near/<crop>/octreg_near,
which collect --near scores as the row octreg_near.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (FLIPS, MASK, OCT, PRIMARY, REPO, RESULTS, SEEDS, WORK, crop_file, crop_name, error,  # noqa: E402
                    reference, tissue, vessel_score)

sys.path.insert(0, str(REPO / "bench/baselines"))
import standard_tools as H  # noqa: E402

BL = WORK / "baselines"
VARIANTS = {"search": H.SEARCH, "mask": H.MASK, "near": H.NEAR, "near_mask": H.NEAR_MASK, "near_zeroed": H.NEAR,
            "atR": H.AT_R, "atR_mask": H.AT_R_MASK}
BASE = {kind: WORK / f"baselines_{kind}" for kind in VARIANTS}
HEADER = {"near": "near", "near_mask": "near", "near_zeroed": "near", "atR": "atR", "atR_mask": "atR"}   # whose start H_k
SHARED = {"search": BL, "mask": BL, "near_mask": BASE["near"], "atR_mask": BASE["atR"]}                  # whose inputs
NEAR_SEED, NEAR_DEG, NEAR_MM = 4700, 10.0, 1.0
TURN = {"near": (NEAR_DEG, NEAR_MM), "atR": (0.0, 0.0)}                                                 # degrees, mm
OCTREG_NEAR = {"octreg_near": "§4 from the start, no search"}
ZEROED = "the OCT zeroed outside the specimen mask"


def runs(kind=None):
    if kind:
        return [(crop_name(PRIMARY, "C3", k), t) for k in range(SEEDS) for t in VARIANTS[kind]]
    crops = [crop_name(PRIMARY, "C3", k) for k in range(SEEDS)] + [crop_name(im, "C3", 0) for im in FLIPS if im != PRIMARY]
    return [(c, t) for c in crops for t in H.TOOLS]


def crops(kind):
    return list(dict.fromkeys(c for c, _ in runs(kind)))


def inputs(crop):
    """The input directory of a crop, shared by the main, search and mask runs."""
    return H.prepare_inputs(BL / crop / "inputs", OCT, crop_file(crop))


def near_start(k, A, degrees=NEAR_DEG, mm=NEAR_MM):
    """Start S_k of crop k, R turned by `degrees` about the tissue centre and shifted by `mm`, and its header pose H_k.
    A: the moving header. H_k is the rotation with a scale per axis closest to S_k, as the qform that ANTs, elastix and
    greedy read cannot hold a shear. S and H map OCT world -> MRI world. At 0 degrees and 0 mm, S is R itself."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    g = np.random.default_rng(NEAR_SEED + k)
    axis = g.standard_normal(3)
    axis /= np.linalg.norm(axis)
    shift = g.standard_normal(3)
    shift *= mm / np.linalg.norm(shift)
    X = tissue()
    c = X.mean(0)
    Q, C, D = np.eye(4), np.eye(4), np.eye(4)
    Q[:3, :3] = Rotation.from_rotvec(np.radians(degrees) * axis).as_matrix()
    C[:3, 3], D[:3, 3] = c, shift
    S = D @ reference() @ C @ Q @ np.linalg.inv(C)
    M = S[:3, :3]
    W = np.linalg.cholesky(np.cov((X - c).T, bias=True))          # mean |(L - M)(x - c)|^2 = |(L - M) W|^2
    lin = lambda p: Rotation.from_rotvec(p[:3]).as_matrix() * np.exp(p[3:])
    U, _, Vt = np.linalg.svd(M)
    p0 = np.r_[Rotation.from_matrix(U @ Vt).as_rotvec(), np.log(np.diag((U @ Vt).T @ M))]
    fit = least_squares(lambda p: ((lin(p) - M) @ W).ravel(), p0, xtol=1e-15, ftol=1e-15, gtol=1e-15)
    Hk = np.eye(4)
    Hk[:3, :3] = lin(fit.x)
    Hk[:3, 3] = (S @ np.r_[c, 1])[:3] - Hk[:3, :3] @ c
    Hk = (Hk @ A).astype(np.float32).astype(float) @ np.linalg.inv(A)
    return {"k": k, "seed": NEAR_SEED + k, "degrees": degrees, "axis": axis.tolist(), "shift_mm": shift.tolist(),
            "centre_mm": c.tolist(), "S": S.tolist(), "H": Hk.tolist()}


def start_from_S(st):
    """Mean and max distance (mm) of H_k from S_k over the tissue points."""
    X = np.c_[tissue(), np.ones(len(tissue()))].T
    e = np.linalg.norm(((np.array(st["H"]) - np.array(st["S"])) @ X)[:3], axis=0)
    return {"mean_mm": round(float(e.mean()), 4), "max_mm": round(float(e.max()), 4)}


def with_header(arr, hdr, path):
    """Save arr as path with the qform and sform hdr, and check the saved header holds hdr. -> max |sform - qform|"""
    img = nib.Nifti1Image(arr, hdr)
    img.header.set_xyzt_units("mm")
    img.header.set_qform(hdr, code=1)
    img.header.set_sform(hdr, code=1)
    nib.save(img, str(path))
    back = nib.load(str(path)).header
    q, s = back.get_qform(), back.get_sform()
    if np.abs(s - hdr).max() > 1e-9 or np.abs(q - s).max() > 1e-5:
        raise ValueError(f"{path}: the header does not hold H_k A (sform {np.abs(s - hdr).max():.2g}, "
                         f"qform {np.abs(q - s).max():.2g})")
    return float(np.abs(q - s).max())


def near_inputs(crop, kind="near"):
    """Inputs of the header-start runs (kind near or atR) of a crop: the main inputs with the OCT header H_k A, their x10
    copies and start.json."""
    d, src = BASE[kind] / crop / "inputs", inputs(crop)
    record = json.loads((src / "inputs.json").read_text(encoding="utf-8"))
    if H.made_from(d / "inputs.json", record):
        return d
    from convert_affines import scale_header
    d.mkdir(parents=True, exist_ok=True)
    o = nib.load(str(src / "oct.nii.gz"))
    st = near_start(int(crop.rsplit("_k", 1)[1]), o.affine, *TURN[kind])
    st["header_sform_minus_qform"] = with_header(np.asarray(o.dataobj, np.float32), np.array(st["H"]) @ o.affine,
                                                 d / "oct.nii.gz")
    shutil.copy(src / "mri.nii.gz", d / "mri.nii.gz")
    for f in ("oct", "mri"):
        nib.save(scale_header(nib.load(str(d / f"{f}.nii.gz")), 10.0), str(d / f"{f}_x10.nii.gz"))
    st["start_from_S"] = start_from_S(st)
    (d / "start.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
    (d / "inputs.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return d


def header_mask(crop, kind):
    """Write the specimen mask of the main inputs into the header-start inputs of a crop with their OCT header H_k A (the
    mask resampled through the pose would be misplaced), and its x10 copy."""
    from convert_affines import scale_header
    d, m = near_inputs(crop, kind), H.prepare_mask(inputs(crop), MASK) / "oct_mask.nii.gz"
    src = {"mask": str(m.resolve())}
    if H.made_from(d / "mask.json", src):
        return d
    o = nib.load(str(d / "oct.nii.gz"))
    with_header(np.asarray(nib.load(str(m)).dataobj, np.uint8), o.affine, d / "oct_mask.nii.gz")
    nib.save(scale_header(nib.load(str(d / "oct_mask.nii.gz")), 10.0), str(d / "oct_mask_x10.nii.gz"))
    (d / "mask.json").write_text(json.dumps(src, indent=1), encoding="utf-8")
    return d


def zeroed_inputs(crop):
    """Inputs of the near runs of a crop on the OCT zeroed outside the specimen mask: the near inputs with that OCT."""
    from convert_affines import scale_header
    d, src = BASE["near_zeroed"] / crop / "inputs", near_inputs(crop)
    m = H.prepare_mask(inputs(crop), MASK) / "oct_mask.nii.gz"
    record = {**json.loads((src / "inputs.json").read_text(encoding="utf-8")), "mask": str(m.resolve())}
    if H.made_from(d / "inputs.json", record):
        return d
    d.mkdir(parents=True, exist_ok=True)
    o = nib.load(str(src / "oct.nii.gz"))
    arr = np.asarray(o.dataobj, np.float32)
    arr[np.asarray(nib.load(str(m)).dataobj) == 0] = 0
    with_header(arr, o.affine, d / "oct.nii.gz")
    shutil.copy(src / "mri.nii.gz", d / "mri.nii.gz")
    for f in ("oct", "mri"):
        nib.save(scale_header(nib.load(str(d / f"{f}.nii.gz")), 10.0), str(d / f"{f}_x10.nii.gz"))
    shutil.copy(src / "start.json", d / "start.json")
    (d / "inputs.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return d


def kind_inputs(crop, kind):
    """The input directory of a crop for the runs of a kind."""
    if kind == "near_zeroed":
        return zeroed_inputs(crop)
    return near_inputs(crop, HEADER[kind]) if kind in HEADER else inputs(crop)


def start_of(crop, kind="near"):
    return json.loads((BASE[kind] / crop / "inputs" / "start.json").read_text(encoding="utf-8"))


def prepare(kind=None):
    if kind == "mask":
        for crop in crops(kind):
            H.prepare_mask(inputs(crop), MASK)
    elif kind in ("near_mask", "atR_mask"):
        for crop in crops(kind):
            header_mask(crop, HEADER[kind])
    H.prepare(BASE[kind] if kind else BL, runs(kind), lambda crop: kind_inputs(crop, kind))


def octreg_run(crop):
    return BASE["near"] / crop / "octreg_near"


def octreg(kind):
    """octreg's refinement alone (bench/ablate.py no_search) from the start H_k of every crop of the near kind, with octreg's
    own masks, one at a time. A run already made by the checkout's octreg code is skipped."""
    from analyses import ablate_one
    if kind != "near":
        sys.exit("octreg runs from the --near starts")
    failed = [crop for crop in crops(kind) if not ablate_one(crop, octreg_run(crop), "no_search",
                                                             ["--start", str(BASE[kind] / crop / "inputs" / "start.json")])]
    if failed:
        sys.exit(f"{len(failed)} runs failed: " + ", ".join(failed))


def octreg_row(crop, score):
    """The row of octreg's refinement from H_k in octreg_run(crop), or None where it has not run."""
    from bench.ablate import code_id, made_by
    run = octreg_run(crop) / "no_search"
    if made_by(run) is None:
        return None
    if made_by(run) != code_id():
        sys.exit(f"{run}: made by octreg code {made_by(run)}, not {code_id()}")
    res = json.loads((run / "result.json").read_text(encoding="utf-8"))
    T = np.loadtxt(run / "T_oct2mri.txt")
    if np.abs(np.array(res["pose"]["T_search"]) - np.array(start_of(crop)["H"])).max() > 1e-9:
        sys.exit(f"{run}: not started from H_k of {crop}")
    return {"crop": crop, "tool": "octreg_near", "setting": OCTREG_NEAR["octreg_near"], **score(crop, T),
            "T_oct2mri": H.as_list(T), "code": code_id(), "seconds": round(res["seconds"]["total"], 1)}


def collect(base=BL, out=RESULTS / "baselines.json", kind=None):
    """Score every finished run under base and write the rows to out. A run from a header start also scores its start H_k,
    and the near kind adds octreg's refinement from that start (octreg_near)."""
    hk = HEADER.get(kind)

    def extra(crop):
        """The start H_k of the crop, scored, and the input where it is not the main one: in every row, tool errors too."""
        out = {}
        if hk:
            st = start_of(crop, hk)
            out.update(H=H.as_list(st["H"]), start_pose=error(np.array(st["H"])), start_from_S=st["start_from_S"])
        if kind == "near_zeroed":
            out["input"] = ZEROED
        return out

    def score(crop, T):
        return {**error(T), **vessel_score(T), **extra(crop)}
    start = (lambda crop: np.array(start_of(crop, hk)["H"])) if hk else None
    rows = H.collect(base, score, lambda crop: True, tools=list(VARIANTS[kind]) if kind else H.TOOLS,
                     inputs=(lambda crop: SHARED[kind] / crop / "inputs") if kind in SHARED else None, start=start)
    for r in rows:
        if r["tier"] == "tool error":
            r.update(extra(r["crop"]))
    expected = set(runs(kind))
    if kind == "near":
        expected |= {(crop, "octreg_near") for crop in crops(kind)}
        rows += [r for r in (octreg_row(crop, score) for crop in crops(kind)) if r is not None]
    missing = sorted(expected - {(r["crop"], r["tool"]) for r in rows})
    if missing:
        sys.exit(f"{len(missing)} runs never ran in {base}: " + ", ".join(f"{c}/{t}" for c, t in missing))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(len(rows), "runs ->", out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prepare", "octreg", "collect"])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--search", dest="kind", action="store_const", const="search", help="the tools' own global searches")
    g.add_argument("--mask", dest="kind", action="store_const", const="mask", help="the test's specimen mask as moving mask")
    g.add_argument("--near", dest="kind", action="store_const", const="near", help="the start near R in the moving header")
    g.add_argument("--near-mask", dest="kind", action="store_const", const="near_mask", help="--near with the moving mask")
    g.add_argument("--near-zeroed", dest="kind", action="store_const", const="near_zeroed",
                   help="--near on the OCT zeroed outside the specimen mask")
    g.add_argument("--atR", dest="kind", action="store_const", const="atR", help="the start at R in the moving header")
    g.add_argument("--atR-mask", dest="kind", action="store_const", const="atR_mask", help="--atR with the moving mask")
    ap.add_argument("--runs", type=Path, help="collect: the folder of runs (default "
                                              "<project root>/bench_runs/I46/baselines, or baselines_<kind>)")
    ap.add_argument("-o", "--out", type=Path, help="collect: the JSON file to write (default bench/results/I46/baselines.json, "
                                                   "or baselines_<kind>.json)")
    a = ap.parse_args()
    if a.step == "prepare":
        prepare(a.kind)
    elif a.step == "octreg":
        octreg(a.kind)
    else:
        collect(a.runs or BASE.get(a.kind, BL),
                a.out or RESULTS / (f"baselines_{a.kind}.json" if a.kind else "baselines.json"), a.kind)

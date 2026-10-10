#!/usr/bin/env python3
"""The freeview figures of the I58 pair in docs/figures (bench only, not part of the package).

    python bench/figures.py [--octreg-run DIR] [--baselines DIR] [--ablation DIR] [-o docs/figures] [--work DIR]

  result/       octreg's result on a sagittal, a coronal and an axial plane through the middle of the block: the MRI, the
                registered OCT (oct_in_mri.nii.gz, affine and §6), and the same OCT with the MRI tissue boundary
  affine/       the affine baselines on the axial plane of result/, each with the MRI tissue boundary
  deformable/   the deformable baselines on the sagittal plane of result/, each with the MRI tissue boundary
  ablation/     the ablations of bench/ablate.py and bench/ablate_deform.py (--ablation, their output directory, by default
                bench_runs/I58/octreg_ablate as in bench/run_i58.py) on the same
                sagittal plane, each with the MRI tissue boundary. The OCT is first resampled through each variant's affine
                (and field) by octreg apply, into DIR/overlays/<panel>/ and DIR/deform/<row>/.

The MRI tissue boundary is octreg's MRI foreground rule (octreg.preprocess.foreground) applied to the MRI crop on its own
0.08 mm grid, made a surface by FreeSurfer's mri_mc and mris_smooth -n 3 -nw, and drawn by freeview as a 2-pixel red line.
Every panel is one freeview screenshot as written: freeview is set to show the whole plane of the MRI crop at exactly one
screen pixel per MRI voxel, with the MRI at grey window 0.3864 to 1.5868 and the OCT at 0.00115 to 0.00367. Nothing is
resampled, cropped or drawn on afterwards. freeview's default orientation applies: sagittal planes have anterior on the
right, coronal and axial planes show the subject's right on the left.

The baseline overlays are the oct_in_mri_affine.nii.gz (affine) and oct_in_mri.nii.gz (deformable) of the run directories
of bench/baselines, under --baselines (default OCTREG_PROJECT_ROOT/baselines). FreeSurfer 8.2 (freeview, mri_mc, mris_smooth)
must be set up in the shell with its licence (FS_LICENSE), or on Windows in WSL (FREESURFER_HOME, default
/usr/local/freesurfer/8.2.0 there, and FS_LICENSE passed through WSLENV).
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from bench.paths import BENCH_RUNS, MRI_I58, OCT_I58, PROJECT_ROOT  # noqa: E402
from octreg import preprocess as pp  # noqa: E402
from octreg.params import Params  # noqa: E402

GREY_MRI, GREY_OCT = "0.3864,1.5868", "0.00115,0.00367"
MIDDLE = (2.31, -0.82, 22.70)   # RAS mm: the three planes of result/, the axial one also for affine/, the sagittal for deformable/
AFFINE = [  # panel name, overlay: RUN/... in the octreg run, else under --baselines
    ("start", "affine/reference/centres"), ("octreg", "RUN/oct_in_mri_affine.nii.gz"), ("reg_aladin", "affine/niftyreg/center"),
    ("greedy", "affine/greedy/centers_fov"), ("elastix", "affine/elastix/cog_fov"),
    ("mri_robust_register", "affine/mri_robust_register/default_com"), ("ants", "affine/ants/com_fov"),
    ("mri_coreg", "affine/mri_coreg/default"), ("flirt", "affine/flirt/search_x10_fov"),
    ("synthmorph", "affine/synthmorph/affine")]
DEFORM = [
    ("affine", "RUN/oct_in_mri_affine.nii.gz"), ("octreg", "RUN/oct_in_mri.nii.gz"), ("syn_fov", "deform/ants/syn_cc_fov"),
    ("syn_default", "deform/ants/syn_cc"), ("greedy", "deform/greedy/wncc_fov"), ("convexadam", "deform/convexadam/default_fov"),
    ("elastix", "deform/elastix/default_fov"), ("reg_f3d", "deform/niftyreg/sx5mm")]
ABLATION = [  # panel name, what it shows under --ablation: an affine (a 4x4 text file) or a §6 run directory with a field
    ("start", "starts/T_image_centres.txt"), ("intensity_mask", "variants/A0/T_oct2mri.txt"),
    ("no_outline", "variants/A9/T_oct2mri.txt"), ("outline_only", "variants/A13/T_oct2mri.txt"),
    ("no_search", "variants/A14/T_oct2mri.txt"), ("no_penalty", "variants/A6p/T_oct2mri.txt"),
    ("no_refinement", "variants/A12/T_oct2mri.txt"), ("turned_start", "starts/axis2_90/T_start.txt"),
    ("turned_no_search", "starts/axis2_90/A14/T_oct2mri.txt"), ("turned_octreg", "starts/axis2_90/method/T_oct2mri.txt"),
    ("interior_only", "deform/interior_only"), ("boundary_only", "deform/boundary_only")]
PLANES = {"sagittal": (0, 1, 2), "coronal": (1, 0, 2), "axial": (2, 0, 1)}   # normal, horizontal, vertical world axis
WSL = platform.system() == "Windows"


def host_path(p: Path) -> str:
    """The path as the FreeSurfer shell sees it (/mnt/<drive>/... in WSL)."""
    s = str(Path(p).resolve()).replace("\\", "/")
    return "/mnt/" + s[0].lower() + s[2:] if WSL and len(s) > 1 and s[1] == ":" else s


def view(img, plane, point):
    """freeview commands that show `plane` of the MRI grid through `point` (RAS mm), the whole crop at one screen pixel per
    voxel. -> (lines, (W, H))."""
    A, shape = img.affine, img.shape[:3]
    ax = [int(np.argmax(np.abs(A[w, :3]))) for w in range(3)]      # array axis along each world axis
    centre, size = list(point), [0, 0]
    for slot, w in enumerate(PLANES[plane][1:]):
        centre[w] = A[w, 3] + A[w, ax[w]] * (shape[ax[w]] - 1) / 2   # centre of the crop
        size[slot] = shape[ax[w]]
    zoom = (max(shape) - 1) / size[1]     # at zoom 1 the largest extent of the crop fills the height of the view
    return [f"-viewport {plane}", f"-viewsize {size[0]} {size[1]}", f"-zoom {zoom:.6f}", "-nocursor",
            "-ras {:.3f} {:.3f} {:.3f} -cc".format(*centre)], tuple(size)


def overlays(abl, run, oct_path, mri):
    """The OCT on the MRI grid for every ABLATION panel, made again when missing or older than the affine or field it shows.
    -> {panel: overlay path}."""
    import json
    from octreg import io
    from octreg.register import apply
    missing = [str(abl / sub) for _, sub in ABLATION if not (abl / sub).exists()]
    if missing:
        sys.exit("missing ablation outputs (bench/README.md gives the commands that write them): " + ", ".join(missing))
    inputs = json.loads((run / "result.json").read_text(encoding="utf-8"))["inputs"]
    out = {}
    for name, sub in ABLATION:
        if (abl / sub).is_dir():                                               # a §6 row: its affine and field
            d, f, affine_only = abl / sub, "oct_in_mri.nii.gz", False
            made_from = [d / "T_oct2mri.txt", d / "oct2mri_warp.nii.gz"]
        else:
            T = np.loadtxt(abl / sub)
            d, f, affine_only = abl / "overlays" / name, "oct_in_mri_affine.nii.gz", True
            res = {"inputs": inputs, "T_oct2mri": T.tolist()}
            made_from = [d / "result.json"]
            if not made_from[0].is_file() or json.loads(made_from[0].read_text(encoding="utf-8")) != res:
                d.mkdir(parents=True, exist_ok=True)          # written only when the affine changes, so its time dates it
                io.write_transform_txt(T, d / "T_oct2mri.txt")
                io.write_transform_txt(np.linalg.inv(T), d / "T_mri2oct.txt")
                io.write_json(res, made_from[0])
        if not (d / f).is_file() or (d / f).stat().st_mtime < max(p.stat().st_mtime for p in made_from):
            apply(d, oct_path, mri, d / f, affine_only=affine_only)
        out[name] = d / f
    return out


def sessions(mri, run, baselines, out, surf, ablation=None):
    """freeview -cmd sessions, one per panel size. ablation: {panel: overlay} or None.
    -> [(name, lines)], {png path: (W, H)}."""
    img = nib.load(str(mri))
    v_mri, f_line = f"-v {host_path(mri)}:grayscale={GREY_MRI}", f"-f {host_path(surf)}:edgecolor=red:edgethickness=2"
    oct_v = lambda f: f"-v {host_path(f)}:grayscale={GREY_OCT}"
    ss = lambda p: f"-ss {host_path(p)} 1 0"
    S, expect = [], {}
    for plane in PLANES:
        lines, wh = view(img, plane, MIDDLE)
        d = out / "result"
        S.append((f"result_{plane}", [v_mri, *lines, ss(d / f"{plane}_mri.png"), "-hide volume", oct_v(run / "oct_in_mri.nii.gz"),
                                      ss(d / f"{plane}_oct.png"), f_line, ss(d / f"{plane}_oct_boundary.png"), "-quit"]))
        expect.update({d / f"{plane}_{n}.png": wh for n in ("mri", "oct", "oct_boundary")})
    for stage, rows, plane, file in (("affine", AFFINE, "axial", "oct_in_mri_affine.nii.gz"),
                                     ("deformable", DEFORM, "sagittal", "oct_in_mri.nii.gz")):
        lines, wh = view(img, plane, MIDDLE)
        d = out / stage
        cmd = [v_mri, f_line, *lines, ss(d / "mri.png"), "-hide volume"]
        for name, sub in rows:
            src = run / sub[4:] if sub.startswith("RUN/") else baselines / sub / file
            if not src.is_file():
                raise FileNotFoundError(src)
            cmd += [oct_v(src), ss(d / f"{name}.png"), "-unload volume"]
        S.append((stage, cmd + ["-quit"]))
        expect.update({d / f"{n}.png": wh for n in ["mri"] + [r[0] for r in rows]})
    if ablation:
        lines, wh = view(img, "sagittal", MIDDLE)
        d = out / "ablation"
        cmd = [v_mri, f_line, *lines, "-hide volume"]
        for name, src in ablation.items():
            cmd += [oct_v(src), ss(d / f"{name}.png"), "-unload volume"]
        S.append(("ablation", cmd + ["-quit"]))
        expect.update({d / f"{n}.png": wh for n in ablation})
    return S, expect


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mri", type=Path, default=MRI_I58)
    ap.add_argument("--octreg-run", type=Path, default=BENCH_RUNS / "I58" / "octreg")
    ap.add_argument("--baselines", type=Path, default=PROJECT_ROOT / "baselines")
    ap.add_argument("--ablation", type=Path, default=BENCH_RUNS / "I58" / "octreg_ablate",
                    help="the output of bench/ablate.py --starts and bench/ablate_deform.py --save-fields, by default the ABL of "
                    "bench/run_i58.py. Its panels are skipped when it does not exist")
    ap.add_argument("--oct", type=Path, default=OCT_I58)
    ap.add_argument("-o", "--out", type=Path, default=REPO / "docs" / "figures")
    ap.add_argument("--work", type=Path, default=BENCH_RUNS / "I58" / "figures")
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    for d in ("result", "affine", "deformable", "ablation"):
        (a.out / d).mkdir(parents=True, exist_ok=True)

    mi = nib.load(str(a.mri))
    tissue, info = pp.foreground(np.asarray(mi.dataobj, np.float32), float(abs(mi.affine[0, :3]).max()), Params())
    print(f"MRI tissue on the 0.08 mm grid: threshold {info['threshold']:.4f}, {info['volume_cm3']:.2f} cm3")
    hdr = mi.header.copy()
    hdr.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(tissue.astype(np.uint8), mi.affine, hdr), str(a.work / "mri_tissue.nii.gz"))

    surf = a.work / "mri_tissue.surf"
    abl = overlays(a.ablation, a.octreg_run, a.oct, a.mri) if a.ablation.is_dir() else None
    if abl is None:
        print(f"no ablation directory {a.ablation}: the panels in {a.out / 'ablation'} are not made again")
    S, expect = sessions(a.mri, a.octreg_run, a.baselines, a.out, surf, abl)
    sh = ["#!/bin/bash", "set -e", 'if [ -n "$FREESURFER_HOME" ] || [ -d /usr/local/freesurfer/8.2.0 ]; then',
          '  export FREESURFER_HOME=${FREESURFER_HOME:-/usr/local/freesurfer/8.2.0}',
          '  source $FREESURFER_HOME/SetUpFreeSurfer.sh >/dev/null 2>&1; fi', f"cd {host_path(a.work)}",
          "mri_mc mri_tissue.nii.gz 1 mri_tissue_raw.surf > mri_mc.log 2>&1",
          "mris_smooth -n 3 -nw mri_tissue_raw.surf mri_tissue.surf > mris_smooth.log 2>&1"]
    for name, lines in S:
        (a.work / f"{name}.cmd").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        sh.append(f"freeview -cmd {name}.cmd > {name}.log 2>&1")
    (a.work / "figures.sh").write_text("\n".join(sh) + "\n", encoding="utf-8", newline="\n")
    for p in expect:
        p.unlink(missing_ok=True)
    subprocess.run((["wsl"] if WSL else []) + ["bash", host_path(a.work / "figures.sh")], check=True)

    from PIL import Image
    bad = [f"{p}: {Image.open(p).size if p.is_file() else 'missing'}, expected {wh}" for p, wh in expect.items()
           if not p.is_file() or Image.open(p).size != wh]
    if bad:
        sys.exit("\n".join(bad))
    print(f"{len(expect)} panels in {a.out}")


if __name__ == "__main__":
    main()

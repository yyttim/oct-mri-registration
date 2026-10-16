#!/usr/bin/env python3
"""The freeview figures of the I58 pair in docs/figures.

    python bench/figures.py [--octreg-run DIR] [--baselines DIR] [--deform-baselines DIR] [--ablation DIR] [--stages S]
                            [--window-from OCT] [-o docs/figures] [--work DIR]

  result/       octreg's result on a sagittal, a coronal and an axial plane through the centroid of the MRI tissue: the MRI,
                the registered OCT (oct_in_mri.nii.gz, affine and §6), and the same OCT with the MRI tissue boundary
  affine/       the affine baselines, each at its setting in bench/baselines/README.md, on the three planes of result/, each
                with the MRI tissue boundary (<panel>_<plane>.png)
  deformable/   the deformable baselines at their settings, started from octreg's affine, on the three planes of result/,
                each with the MRI tissue boundary (<panel>_<plane>.png)
  ablation/     the ablations of bench/ablate.py --starts (--ablation, its output directory, by default
                bench_runs/I58/octreg_ablate as in bench/run_i58.py) on the same sagittal plane, each with the MRI tissue
                boundary. The OCT is first resampled through each variant's affine by octreg apply, into
                DIR/overlays/<panel>/.

The MRI tissue boundary is the MRI foreground rule of §1 (octreg.preprocess.foreground, a histogram-valley threshold of the
MRI) applied to the MRI crop on its own 0.08 mm grid, made a surface by FreeSurfer's mri_mc and mris_smooth -n 3 -nw, and
drawn by freeview as a 2-pixel red line, the same in every panel. The three planes pass through the voxel centre nearest the
centroid of this tissue.

Every panel is one freeview screenshot of the whole plane of the MRI crop at one screen pixel per MRI voxel, with linear grey
levels. The MRI takes freeview's default window (freeview_window), and every OCT panel takes the default window of the moving
image every method received (--window-from, the OCT box-averaged to 0.08 mm of bench/baselines). The windows, the plane centre
and the MRI tissue are written to figures.json in -o. freeview's default orientation applies: sagittal planes have anterior on
the right, coronal and axial planes show the subject's right on the left.

The baseline overlays are the oct_in_mri_affine.nii.gz (affine) and oct_in_mri.nii.gz (deformable) of the run directories
of bench/baselines, under --baselines (default OCTREG_PROJECT_ROOT/baselines). FreeSurfer 8.2 (freeview, mri_mc, mris_smooth)
must be set up in the shell with its licence (FS_LICENSE, default ~/freesurfer/license.txt), or on Windows in WSL
(FREESURFER_HOME, default /usr/local/freesurfer/8.2.0 there, and FS_LICENSE passed through WSLENV or at the default path).
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from bench.paths import BENCH_RUNS, MRI_I58, OCT_I58, PROJECT_ROOT  # noqa: E402

AFFINE = [  # panel name, overlay: RUN/... in the octreg run, else under --baselines, each tool at its documented setting
    ("start", "affine/reference/centres"), ("octreg", "RUN/oct_in_mri_affine.nii.gz"), ("reg_aladin", "affine/niftyreg/center"),
    ("greedy", "affine/greedy/centers"), ("elastix", "affine/elastix/center"),
    ("mri_robust_register", "affine/mri_robust_register/default_com"), ("ants", "affine/ants/com"),
    ("mri_coreg", "affine/mri_coreg/default"), ("flirt", "affine/flirt/search_x10"), ("synthmorph", "affine/synthmorph/affine")]
DEFORM = [  # the same for the deformable stage, under --deform-baselines
    ("affine", "RUN/oct_in_mri_affine.nii.gz"), ("octreg", "RUN/oct_in_mri.nii.gz"), ("ants_syn", "deform/ants/syn_cc"),
    ("elastix", "deform/elastix/default"), ("reg_f3d", "deform/niftyreg/default"), ("convexadam", "deform/convexadam/default"),
    ("greedy", "deform/greedy/wncc")]
ABLATION = [  # panel name, the affine it shows (a 4x4 text file under --ablation)
    ("intensity_mask", "variants/A0/T_oct2mri.txt"), ("no_outline", "variants/A9/T_oct2mri.txt"),
    ("no_penalty", "variants/A6p/T_oct2mri.txt"), ("no_refinement", "variants/A12/T_oct2mri.txt"),
    ("outline_only", "variants/A13/T_oct2mri.txt"), ("turned_start", "starts/axis2_90/T_start.txt"),
    ("turned_no_search", "starts/axis2_90/A14/T_oct2mri.txt"), ("turned_octreg", "starts/axis2_90/method/T_oct2mri.txt")]
PLANES = {"sagittal": (0, 1, 2), "coronal": (1, 0, 2), "axial": (2, 0, 1)}   # normal, horizontal, vertical world axis
STAGES = ("result", "affine", "deformable", "ablation")
WSL = platform.system() == "Windows"


def host_path(p: Path) -> str:
    """The path as the FreeSurfer shell sees it (/mnt/<drive>/... in WSL)."""
    s = str(Path(p).resolve()).replace("\\", "/")
    return "/mnt/" + s[0].lower() + s[2:] if WSL and len(s) > 1 and s[1] == ":" else s


def freeview_window(path):
    """freeview's default grey window of a volume with no negative values (FreeSurfer 8.2, LayerPropertyMRI and FSVolume):
    a 1000-bin histogram, from the minimum to the maximum of the volume, of the voxels that are not 0, each split linearly
    between its two nearest bins, and the window from the bin nearest the 5th percentile to the bin nearest the 99th plus a
    fifth of the difference. -> (lo, hi)."""
    img = nib.load(str(path))
    slabs = lambda: (np.asarray(img.dataobj[:, :, i:i + 32], np.float32).ravel() for i in range(0, img.shape[2], 32))
    vmin = min(float(a.min()) for a in slabs())
    vmax = max(float(a.max()) for a in slabs())
    step = (vmax - vmin) / 999
    counts = np.zeros(1000)
    for a in slabs():
        d = (a[np.abs(a) >= 1e-7].astype(np.float64) - vmin) / step
        lo_bin, hi_bin = np.floor(d), np.ceil(d)
        counts += (np.bincount(np.minimum(lo_bin, 999).astype(int), hi_bin - d, 1000)
                   + np.bincount(np.minimum(hi_bin, 999).astype(int), d - lo_bin, 1000))
    cdf = np.cumsum(counts) / counts.sum()
    bins = vmin + step * np.arange(1000)
    p5, p99 = (float(bins[np.argmin(np.abs(cdf - p))]) for p in (0.05, 0.99))
    return p5, p99 + (p99 - p5) / 5


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
    """The OCT on the MRI grid for every ABLATION panel, made again when missing or older than the affine it shows.
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
        T = np.loadtxt(abl / sub)
        d, f = abl / "overlays" / name, "oct_in_mri_affine.nii.gz"
        res, made_from = {"inputs": inputs, "T_oct2mri": T.tolist()}, d / "result.json"
        if not made_from.is_file() or json.loads(made_from.read_text(encoding="utf-8")) != res:
            d.mkdir(parents=True, exist_ok=True)              # written only when the affine changes, so its time dates it
            io.write_transform_txt(T, d / "T_oct2mri.txt")
            io.write_transform_txt(np.linalg.inv(T), d / "T_mri2oct.txt")
            io.write_json(res, made_from)
        if not (d / f).is_file() or (d / f).stat().st_mtime < made_from.stat().st_mtime:
            apply(d, oct_path, mri, d / f, affine_only=True)
        out[name] = d / f
    return out


def figure_info(mri, window_from, tissue, threshold, middle):
    """What the panels are drawn with, for figures.json: the grey windows, the plane centre and the MRI tissue of the line."""
    img = nib.load(str(mri))
    return {"mri": Path(mri).name, "mri_window": [round(x, 6) for x in freeview_window(mri)],
            "oct_window_from": Path(window_from).name, "oct_window": [round(x, 6) for x in freeview_window(window_from)],
            "planes_through_ras_mm": [round(x, 4) for x in middle],
            "mri_tissue": {"threshold": round(float(threshold), 6),
                           "volume_cm3": round(float(tissue.sum() * np.prod(np.linalg.norm(img.affine[:3, :3], axis=0)) / 1e3), 4)}}


def mri_tissue(img):
    """The MRI tissue of the red line: octreg's MRI foreground rule on the grid of the crop. -> (bool array, threshold)."""
    from octreg import preprocess as pp
    from octreg.params import Params
    tissue, info = pp.foreground(np.asarray(img.dataobj, np.float32), float(np.linalg.norm(img.affine[:3, :3], axis=0).min()),
                                 Params())
    return tissue, info["threshold"]


def sessions(mri, run, baselines, out, surf, middle, window_from, ablation=None, deform_baselines=None, stages=STAGES):
    """freeview -cmd sessions, one per panel size, for the folders in stages. middle: RAS mm of the three planes. ablation:
    {panel: overlay} or None. -> [(name, lines)], {png path: (W, H)}."""
    img = nib.load(str(mri))
    grey_mri, grey_oct = freeview_window(mri), freeview_window(window_from)
    print("grey windows: MRI {:.4g} to {:.4g}, OCT {:.4g} to {:.4g}".format(*grey_mri, *grey_oct))
    v_mri = "-v {}:grayscale={!r},{!r}".format(host_path(mri), *grey_mri)
    f_line = f"-f {host_path(surf)}:edgecolor=red:edgethickness=2"
    oct_v = lambda f: "-v {}:grayscale={!r},{!r}".format(host_path(f), *grey_oct)
    ss = lambda p: f"-ss {host_path(p)} 1 0"
    S, expect = [], {}
    for plane in PLANES if "result" in stages else ():
        lines, wh = view(img, plane, middle)
        d = out / "result"
        S.append((f"result_{plane}", [v_mri, *lines, ss(d / f"{plane}_mri.png"), "-hide volume", oct_v(run / "oct_in_mri.nii.gz"),
                                      ss(d / f"{plane}_oct.png"), f_line, ss(d / f"{plane}_oct_boundary.png"), "-quit"]))
        expect.update({d / f"{plane}_{n}.png": wh for n in ("mri", "oct", "oct_boundary")})
    for stage, rows, file in (("affine", AFFINE, "oct_in_mri_affine.nii.gz"), ("deformable", DEFORM, "oct_in_mri.nii.gz")):
        if stage not in stages:
            continue
        d = out / stage
        root = (deform_baselines or baselines) if stage == "deformable" else baselines
        srcs = {name: run / sub[4:] if sub.startswith("RUN/") else root / sub / file for name, sub in rows}
        missing = [str(f) for f in srcs.values() if not f.is_file()]
        if missing:
            raise FileNotFoundError(", ".join(missing))
        for plane in PLANES:
            lines, wh = view(img, plane, middle)
            cmd = [v_mri, f_line, *lines, ss(d / f"mri_{plane}.png"), "-hide volume"]
            for name, src in srcs.items():
                cmd += [oct_v(src), ss(d / f"{name}_{plane}.png"), "-unload volume"]
            S.append((f"{stage}_{plane}", cmd + ["-quit"]))
            expect.update({d / f"{n}_{plane}.png": wh for n in ["mri"] + list(srcs)})
    if ablation and "ablation" in stages:
        lines, wh = view(img, "sagittal", middle)
        d = out / "ablation"
        cmd = [v_mri, f_line, *lines, "-hide volume"]
        for name, src in ablation.items():
            cmd += [oct_v(src), ss(d / f"{name}.png"), "-unload volume"]
        S.append(("ablation", cmd + ["-quit"]))
        expect.update({d / f"{n}.png": wh for n in ablation})
    return S, expect


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mri", type=Path, default=MRI_I58)
    ap.add_argument("--octreg-run", type=Path, default=BENCH_RUNS / "I58" / "octreg")
    ap.add_argument("--baselines", type=Path, default=PROJECT_ROOT / "baselines")
    ap.add_argument("--deform-baselines", type=Path, default=None,
                    help="the root of the deformable baseline runs, started from this octreg run's affine (default: --baselines)")
    ap.add_argument("--ablation", type=Path, default=BENCH_RUNS / "I58" / "octreg_ablate",
                    help="the output of bench/ablate.py --starts, by default the ABL of bench/run_i58.py. Its panels are "
                    "skipped when it does not exist")
    ap.add_argument("--oct", type=Path, default=OCT_I58)
    ap.add_argument("--stages", default=",".join(STAGES), help="comma-separated folders to make (default: all four)")
    ap.add_argument("--window-from", type=Path, default=PROJECT_ROOT / "baselines" / "inputs" / "oct_0.08mm.nii.gz",
                    help="the volume whose freeview default window every OCT panel takes (default: the moving image of "
                    "bench/baselines)")
    ap.add_argument("-o", "--out", type=Path, default=REPO / "docs" / "figures")
    ap.add_argument("--work", type=Path, default=BENCH_RUNS / "I58" / "figures")
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    for d in ("result", "affine", "deformable", "ablation"):
        (a.out / d).mkdir(parents=True, exist_ok=True)

    mi = nib.load(str(a.mri))
    tissue, t = mri_tissue(mi)
    c = np.floor(np.argwhere(tissue).mean(0) + 0.5)                       # the voxel centre nearest the tissue centroid
    middle = tuple(float(x) for x in (mi.affine @ np.r_[c, 1])[:3])
    vol = tissue.sum() * np.prod(np.linalg.norm(mi.affine[:3, :3], axis=0)) / 1e3
    print(f"MRI tissue: threshold {t:.4f}, {vol:.2f} cm3, planes through RAS " + "{:.2f} {:.2f} {:.2f}".format(*middle))
    hdr = mi.header.copy()
    hdr.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(tissue.astype(np.uint8), mi.affine, hdr), str(a.work / "mri_tissue.nii.gz"))

    surf = a.work / "mri_tissue.surf"
    stages = a.stages.split(",")
    abl = overlays(a.ablation, a.octreg_run, a.oct, a.mri) if a.ablation.is_dir() and "ablation" in stages else None
    if abl is None:
        print(f"no ablation directory {a.ablation}: the panels in {a.out / 'ablation'} are not made again"
              if "ablation" in stages else "the ablation panels are not made (--stages)")
    S, expect = sessions(a.mri, a.octreg_run, a.baselines, a.out, surf, middle, a.window_from, abl, a.deform_baselines, stages)
    sh = ["#!/bin/bash", "set -e", 'if [ -n "$FREESURFER_HOME" ] || [ -d /usr/local/freesurfer/8.2.0 ]; then',
          '  export FREESURFER_HOME=${FREESURFER_HOME:-/usr/local/freesurfer/8.2.0}',
          '  source $FREESURFER_HOME/SetUpFreeSurfer.sh >/dev/null 2>&1; fi',
          "export FS_LICENSE=${FS_LICENSE:-$HOME/freesurfer/license.txt}", f"cd {host_path(a.work)}",
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
    (a.out / "figures.json").write_text(json.dumps(figure_info(a.mri, a.window_from, tissue, t, middle), indent=1) + "\n",
                                    encoding="utf-8")
    print(f"{len(expect)} panels in {a.out}")


if __name__ == "__main__":
    main()

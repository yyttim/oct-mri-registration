#!/usr/bin/env python3
"""freeview panels of the DANDI:000874 evaluation in docs/figures/sp002, made the way bench/figures.py makes the I58 panels.

For each block, on its C3 k = 0 crop of the 5 deg FLASH, the axial plane through the middle slice of the crop: the MRI
(<b>_mri.png), and the OCT on the MRI grid through the authors' reference (<b>_reference.png) and through octreg's affine
(<b>_octreg.png), each with the MRI tissue boundary as a 2-pixel red line. The OCT is box-averaged to the MRI voxel size and
resampled by octreg's own resampler. The boundary is octreg's MRI foreground (octreg.preprocess.foreground) on the crop, made a
surface by FreeSurfer's mri_mc and mris_smooth -n 3 -nw. Every panel is one freeview screenshot of the whole plane at two screen
pixels per MRI voxel (0.15 mm), nothing drawn afterwards. Grey levels are linear at freeview's default window of each file
(bench.figures.freeview_window): the MRI crop's for the MRI, and the reference-resampled OCT's for both OCT panels of a block.

    python bench/dandi874/figures.py [-o docs/figures/sp002]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import BLOCKS, REPO, RUNS, WORK, crop_file, crop_name, oct_nifti, reference  # noqa: E402
from bench.figures import WSL, freeview_window, host_path, view  # noqa: E402
from octreg import geometry as G, io, preprocess as pp  # noqa: E402
from octreg.params import Params  # noqa: E402

K = 2   # screen pixels per MRI voxel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", type=Path, default=REPO / "docs/figures/sp002")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    work = WORK / "figures"
    work.mkdir(parents=True, exist_ok=True)
    sh = ["#!/bin/bash", 'if [ -n "$FREESURFER_HOME" ] || [ -d /usr/local/freesurfer/8.2.0 ]; then',
          "  export FREESURFER_HOME=${FREESURFER_HOME:-/usr/local/freesurfer/8.2.0}",
          "  source $FREESURFER_HOME/SetUpFreeSurfer.sh >/dev/null 2>&1; fi", "set -e"]
    expect = {}
    for b in BLOCKS:
        name = crop_name(b, "flip2", "C3", 0)
        mri, d = crop_file(name), work / f"s{b}"
        d.mkdir(exist_ok=True)
        m = nib.load(str(mri))
        vo, vm = io.load_volume(oct_nifti(b)), io.load_volume(mri)
        arr, A = G.resample_iso(vo, max(float(vm.spacing_mm.min()), float(vo.spacing_mm.min())))
        for tag, T in (("reference", reference(b)), ("octreg", np.loadtxt(RUNS / name / "T_oct2mri.txt"))):
            io.save_nifti(G.resample_to(arr, A, vm.shape, vm.affine, np.linalg.inv(T)), vm.affine, d / f"oct_{tag}.nii.gz")
        tissue, _ = pp.foreground(np.asarray(m.dataobj, np.float32), float(np.linalg.norm(m.affine[:3, 0])), Params())
        hdr = m.header.copy(); hdr.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(tissue.astype(np.uint8), m.affine, hdr), str(d / "mri_tissue.nii.gz"))
        gm, go = freeview_window(mri), freeview_window(d / "oct_reference.nii.gz")
        c = (np.array(m.shape[:3]) - 1) / 2
        n = int(np.argmax(np.abs(m.affine[2, :3])))         # the array axis along world z, the normal of the axial plane
        c[n] = np.floor(c[n])                                 # a voxel centre, so that the plane is one slice
        centre = (m.affine @ np.r_[c, 1])[:3]
        lines, wh = view(m, "axial", centre)
        lines = [f"-viewsize {wh[0] * K} {wh[1] * K}" if l.startswith("-viewsize") else l for l in lines]
        ss = lambda p: f"-ss {host_path(p)} 1 0"
        cmd = ["-v {}:grayscale={!r},{!r}".format(host_path(mri), *gm), *lines, ss(a.out / f"s{b}_mri.png"), "-hide volume",
               f"-f {host_path(d / 'mri_tissue.surf')}:edgecolor=red:edgethickness=2"]
        for tag in ("reference", "octreg"):
            cmd += ["-v {}:grayscale={!r},{!r}".format(host_path(d / f"oct_{tag}.nii.gz"), *go), ss(a.out / f"s{b}_{tag}.png"), "-unload volume"]
        (d / "panels.cmd").write_text("\n".join(cmd + ["-quit"]) + "\n", encoding="utf-8", newline="\n")
        sh += [f"cd {host_path(d)}", "mri_mc mri_tissue.nii.gz 1 mri_tissue_raw.surf > mri_mc.log 2>&1",
               "mris_smooth -n 3 -nw mri_tissue_raw.surf mri_tissue.surf > mris_smooth.log 2>&1", "freeview -cmd panels.cmd > freeview.log 2>&1"]
        expect.update({a.out / f"s{b}_{t}.png": (wh[0] * K, wh[1] * K) for t in ("mri", "reference", "octreg")})
    (work / "figures.sh").write_text("\n".join(sh) + "\n", encoding="utf-8", newline="\n")
    for p in expect:
        p.unlink(missing_ok=True)
    subprocess.run((["wsl"] if WSL else []) + ["bash", host_path(work / "figures.sh")], check=True)
    from PIL import Image
    bad = [f"{p}: {Image.open(p).size if p.is_file() else 'missing'}, expected {wh}" for p, wh in expect.items()
           if not p.is_file() or Image.open(p).size != wh]
    if bad:
        sys.exit("\n".join(bad))
    print(f"{len(expect)} panels in {a.out}")


if __name__ == "__main__":
    main()

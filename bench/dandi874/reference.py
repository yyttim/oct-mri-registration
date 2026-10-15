#!/usr/bin/env python3
"""How well the authors' reference agrees with the image data (bench/dandi874/README.md, Reference).

local     local refinements started at the reference on each block's C3 k = 0 crop, and how far each moves from it:
            ants     antsRegistration, Affine, MI (32 bins, regular 25 %), the two finest levels (shrink 2x1, smoothing 1x0 vox)
            aladin   reg_aladin -inaff, two levels (-ln 2: half and full resolution)
            octreg   octreg's §4 fit from the reference (the polarity with the lower loss), then §5
          The inputs are those of the baselines (<bench_runs>/SP002/baselines/<crop>/inputs).
restarts  mri_robust_register --cost NMI --affine, the authors' registration as their sidecars name it, restarted from the
          reference moved by 2 deg about a random axis through the tissue centroid and 0.5 mm in a random direction (numpy
          default_rng(block), ten starts), on the OCT box-averaged to 80 um and the whole 5 deg FLASH.
Tools run in WSL on Windows and in the shell elsewhere. They need ~/baselines/tools/env.sh and ~/baselines/fs_env.sh
(bench/baselines/INSTALL.md).
score     -> bench/results/SP002/reference.json

    python bench/dandi874/reference.py local|restarts|score
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import BLOCKS, INPUTS, REPO, RESULTS, WORK, crop_file, crop_name, error, mri_file, oct_nifti, reference, tissue  # noqa: E402

sys.path.insert(0, str(REPO / "bench/baselines"))
OUT = WORK / "reference"
WIN = platform.system() == "Windows"


def wsl(p):
    s = str(Path(p).resolve()).replace("\\", "/")
    return "/mnt/" + s[0].lower() + s[2:] if WIN and len(s) > 1 and s[1] == ":" else s


def local():
    import torch
    from octreg import geometry as G, io, preprocess as pp, refine as Rf
    from octreg.ngf import refine_ngf
    from octreg.params import Params
    from octreg.register import prepare_mri, prepare_oct
    sh = ["#!/bin/bash", "source ~/baselines/tools/env.sh", "export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=12"]
    for b in BLOCKS:
        crop = crop_name(b, "flip2", "C3", 0)
        inp, d = WORK / "baselines" / crop / "inputs", OUT / f"s{b}"
        d.mkdir(parents=True, exist_ok=True)
        R = reference(b)
        io.write_itk(R, io.load_volume(inp / "oct.nii.gz"), io.load_volume(inp / "mri.nii.gz"), d / "ref_itk.txt")
        np.savetxt(d / "ref_niftyreg.txt", np.linalg.inv(R), fmt="%.10f")
        F, M, D = wsl(inp / "mri.nii.gz"), wsl(inp / "oct.nii.gz"), wsl(d)
        sh += [f"cd {D}",
               f"antsRegistration -d 3 --float 0 --collapse-output-transforms 1 -o [{D}/ants_] -n Linear --initial-moving-transform "
               f"{D}/ref_itk.txt -m MI[{F},{M},1,32,Regular,0.25] -t Affine[0.1] -c [200x100,1e-6,10] -f 2x1 -s 1x0vox > ants.log 2>&1",
               f"reg_aladin -ref {F} -flo {M} -inaff {D}/ref_niftyreg.txt -ln 2 -aff {D}/aladin_aff.txt -res {D}/aladin_res.nii.gz "
               f"-omp 8 > aladin.log 2>&1"]
        P = Params()
        vo, vm = io.load_volume(oct_nifti(b)), io.load_volume(crop_file(crop))
        (mri_h, mask_m, A_m), _ = prepare_mri(vm, P)
        fine, A_f = G.resample_iso(vo, P.fine_mm)
        (oct_h, mask_o, A_o), valid_o, _ = prepare_oct(fine, A_f, P)
        del fine
        u, w = pp.oct_channels(pp.two_class(oct_h, mask_o, P.base_mm, P), mask_o)
        v = pp.mri_channels(pp.two_class(mri_h, mask_m, P.base_mm, P), mask_m)
        grid = Rf.BaseGrid((v, mask_m, A_m), (u, w, valid_o, A_o), "cuda")
        T4 = min((Rf.fit(grid, R, p, P) for p in (1, -1)), key=lambda f: f[4])[0]
        T5, _ = refine_ngf((mri_h, mask_m, A_m), (oct_h, mask_o, A_o), T4, P, device="cuda")
        np.savetxt(d / "octreg_T.txt", T5, fmt="%.10f")
        del grid; torch.cuda.empty_cache()
    (OUT / "local.sh").write_text("\n".join(sh) + "\n", encoding="utf-8", newline="\n")
    subprocess.run((["wsl"] if WIN else []) + ["bash", wsl(OUT / "local.sh")], check=True)


def rot(axis, deg):
    a, k = np.radians(deg), axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def restarts():
    from octreg import geometry as G, io
    vm = io.load_volume(mri_file("flip2"))
    sh = ["#!/bin/bash", "source ~/baselines/fs_env.sh"]
    for b in BLOCKS:
        o80 = INPUTS / f"sample-{b}_oct_80um.nii.gz"
        if not o80.is_file():
            arr, A = G.resample_iso(io.load_volume(oct_nifti(b)), 0.08)
            io.save_nifti(arr, A, o80)
        vo, R, c = io.load_volume(o80), reference(b), tissue(b).mean(0)
        rng = np.random.default_rng(int(b))
        for k in range(10):
            Rk = rot(rng.normal(size=3), 2.0)
            t = rng.normal(size=3)
            Pm = np.eye(4)
            Pm[:3, :3], Pm[:3, 3] = Rk, c - Rk @ c + 0.5 * t / np.linalg.norm(t)
            d = OUT / "restarts" / f"s{b}_k{k}"
            d.mkdir(parents=True, exist_ok=True)
            np.savetxt(d / "T_init.txt", R @ Pm, fmt="%.10f")
            io.write_lta(R @ Pm, vo, vm, d / "init.lta")
            sh.append(f"cd {wsl(d)} && mri_robust_register --mov {wsl(o80)} --dst {wsl(mri_file('flip2'))} --lta out.lta --cost NMI --affine "
                      f"--ixform init.lta > log.txt 2>&1; echo s{b}_k{k} rc=$?")
    (OUT / "restarts.sh").write_text("\n".join(sh) + "\n", encoding="utf-8", newline="\n")
    subprocess.run((["wsl"] if WIN else []) + ["bash", wsl(OUT / "restarts.sh")], check=True)


def score():
    from convert_affines import ants_mat_to_mov2fix_ras, lta_to_mov2fix_ras, niftyreg_txt_to_mov2fix_ras
    rows = []
    for b in BLOCKS:
        d = OUT / f"s{b}"
        for name, f in (("ants", lambda: ants_mat_to_mov2fix_ras(d / "ants_0GenericAffine.mat")),
                        ("aladin", lambda: niftyreg_txt_to_mov2fix_ras(d / "aladin_aff.txt")),
                        ("octreg", lambda: np.loadtxt(d / "octreg_T.txt"))):
            rows.append({"check": "local", "block": b, "method": name, **error(b, f())})
        mri, src = nib.load(str(mri_file("flip2"))), nib.load(str(INPUTS / f"sample-{b}_oct_80um.nii.gz"))
        for k in range(10):
            d = OUT / "restarts" / f"s{b}_k{k}"
            row = {"check": "restart", "block": b, "k": k, "start_mean_mm": error(b, np.loadtxt(d / "T_init.txt"))["mean_mm"]}
            if (d / "out.lta").is_file():
                row.update(error(b, lta_to_mov2fix_ras(d / "out.lta", src, mri)))
            else:
                row["note"] = "no output: the tool stopped (see log.txt)"
            rows.append(row)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "reference.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    for r in rows:
        print(json.dumps(r)[:160])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["local", "restarts", "score"])
    {"local": local, "restarts": restarts, "score": score}[ap.parse_args().step]()

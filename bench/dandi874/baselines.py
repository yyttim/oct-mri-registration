#!/usr/bin/env python3
"""The baseline comparison of bench/baselines on the DANDI:000874 crops: each tool at the setting reported for it on I58
(bench/baselines/BASELINES.md), with the same commands and the same conversion of its transform
(bench/baselines/convert_affines.py).

Inputs per crop, as in the I58 protocol: the fixed image is the MRI crop, the moving image the 20 um OCT box-averaged to the MRI
voxel size with its own header, and the field-of-view mask of the OCT the whole array (these OCT arrays have no unmeasured
region). FLIRT gets copies with headers scaled x10, as on I58. The elastix parameter files are those of the I58 run
(bench/dandi874/elastix/: the model-zoo rigid and affine files with the centre-of-gravity start).

    python bench/dandi874/baselines.py prepare     inputs and run.sh of every tool on the C3 k = 0 crops and of reg_aladin
                                                   on all C3 crops
    bash <bench_runs>/SP002/baselines/queue.sh     runs them one at a time (WSL tools in WSL, greedy with Windows Python)
    python bench/dandi874/baselines.py collect     converts each transform, checks its convention and scores it
-> bench/results/SP002/baselines.json

Convention check (as bench/baselines/rundir.py check on I58): the tool's own resampling of the OCT onto the crop against
octreg's resampling through the converted transform, Pearson r over the voxels where either is non-zero and the Dice of the two
supports (reg_aladin resampled linearly with reg_resample, the LTA tools with mri_vol2vol, FLIRT with flirt -applyxfm, greedy
with greedy -r). The tools run in WSL on Windows (greedy with the Windows Python) and in the shell elsewhere. They need
~/baselines/tools/env.sh and ~/baselines/fs_env.sh (bench/baselines/INSTALL.md). The two-step tools (mri_robust_register,
mri_coreg, FLIRT) are timed over both steps, as on I58.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import BLOCKS, REPO, RESULTS, SEEDS, WORK, crop_file, crop_name, error, oct_nifti  # noqa: E402

sys.path.insert(0, str(REPO / "bench/baselines"))
BL = WORK / "baselines"
TOOLS = ["reg_aladin", "greedy", "elastix", "ants", "mri_robust_register", "mri_coreg", "flirt", "synthmorph"]
SETTING = {"reg_aladin": "center", "greedy": "centers_fov", "elastix": "cog_fov", "ants": "com_fov", "mri_robust_register": "default_com",
           "mri_coreg": "default", "flirt": "search_x10_fov", "synthmorph": "affine"}
PY = sys.executable
WIN = platform.system() == "Windows"


def wsl(p):
    s = str(Path(p).resolve()).replace("\\", "/")
    return "/mnt/" + s[0].lower() + s[2:] if WIN and len(s) > 1 and s[1] == ":" else s


def prepare_inputs(crop):
    from octreg import geometry as G, io
    from convert_affines import scale_header
    d = BL / crop / "inputs"
    if (d / "mri_x10.nii.gz").is_file():
        return d
    d.mkdir(parents=True, exist_ok=True)
    vm = io.load_volume(crop_file(crop))
    arr, A = G.resample_iso(io.load_volume(oct_nifti(crop[1:3])), float(vm.spacing_mm.min()))
    io.save_nifti(arr, A, d / "oct.nii.gz")
    io.save_nifti(np.ones(arr.shape, np.uint8), A, d / "oct_fov.nii.gz")
    shutil.copy(crop_file(crop), d / "mri.nii.gz")
    m = nib.load(str(d / "mri.nii.gz"))
    io.save_nifti(np.ones(m.shape, np.uint8), m.affine, d / "mri_ones.nii.gz")
    for f in ("oct", "oct_fov", "mri"):
        nib.save(scale_header(nib.load(str(d / f"{f}.nii.gz")), 10.0), str(d / f"{f}_x10.nii.gz"))
    return d


def script(tool, run, inp):
    R, I = wsl(run), wsl(inp)
    fs, tl = "source ~/baselines/fs_env.sh", "source ~/baselines/tools/env.sh"
    t = "/usr/bin/time -f %e -o time.txt"
    # the two-step tools, summed as on I58
    t1, t2 = "/usr/bin/time -f %e -o time_rigid.txt", "/usr/bin/time -f %e -o time_affine.txt"
    total = "awk '{s+=$1} END {printf \"%.2f\\n\", s}' time_rigid.txt time_affine.txt > time.txt"
    head = ["#!/bin/bash", "set -o pipefail", f"cd {R} || exit 1"]
    if tool == "reg_aladin":
        return head + [tl, f"{t} reg_aladin -ref {I}/mri.nii.gz -flo {I}/oct.nii.gz -aff {R}/aff.txt -res {R}/warped_tool.nii.gz -omp 8 > log.txt 2>&1",
                       f"reg_resample -ref {I}/mri.nii.gz -flo {I}/oct.nii.gz -trans {R}/aff.txt -inter 1 -pad 0 -res {R}/warped_check.nii.gz -omp 8 >> log.txt 2>&1"]
    if tool == "elastix":
        return head + [tl, f"{t} elastix -f {I}/mri.nii.gz -m {I}/oct.nii.gz -mMask {I}/oct_fov.nii.gz -p {R}/rigid.txt -p {R}/affine.txt "
                           f"-out {R} -threads 12 > log.txt 2>&1", "cp result.1.nii.gz warped_check.nii.gz"]
    if tool == "ants":
        return head + [tl, "export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=12",
                       f"{t} antsRegistrationSyN.sh -d 3 -f {I}/mri.nii.gz -m {I}/oct.nii.gz -o {R}/ants_ -t a -n 12 "
                       f"-x {I}/mri_ones.nii.gz,{I}/oct_fov.nii.gz > log.txt 2>&1", "cp ants_Warped.nii.gz warped_check.nii.gz"]
    if tool == "mri_robust_register":
        return head + [fs, f"{t1} mri_robust_register --mov {I}/oct.nii.gz --dst {I}/mri.nii.gz --lta rigid.lta --cost NMI > log.txt 2>&1",
                       f"{t2} mri_robust_register --mov {I}/oct.nii.gz --dst {I}/mri.nii.gz --lta affine.lta --cost NMI --affine --ixform rigid.lta >> log.txt 2>&1",
                       total,
                       f"mri_vol2vol --mov {I}/oct.nii.gz --targ {I}/mri.nii.gz --lta affine.lta --o warped_check.nii.gz --trilin >> log.txt 2>&1"]
    if tool == "mri_coreg":
        return head + [fs, f"{t1} mri_coreg --mov {I}/oct.nii.gz --ref {I}/mri.nii.gz --reg rigid.lta --dof 6 --threads 20 > log.txt 2>&1",
                       f"{t2} mri_coreg --mov {I}/oct.nii.gz --ref {I}/mri.nii.gz --reg affine.lta --dof 12 --init-reg rigid.lta --threads 20 >> log.txt 2>&1",
                       total,
                       "lta_convert --inlta affine.lta --outlta affine_ras.lta >> log.txt 2>&1",
                       f"mri_vol2vol --mov {I}/oct.nii.gz --targ {I}/mri.nii.gz --lta affine_ras.lta --o warped_check.nii.gz --trilin >> log.txt 2>&1"]
    if tool == "flirt":
        c = "-cost normmi -searchcost normmi -bins 256 -interp trilinear"
        return head + [tl, "export OMP_NUM_THREADS=8",
                       f"{t1} flirt -v -in {I}/oct_x10.nii.gz -ref {I}/mri_x10.nii.gz -dof 6 -inweight {I}/oct_fov_x10.nii.gz {c} -omat rigid.mat > log.txt 2>&1",
                       f"{t2} flirt -v -in {I}/oct_x10.nii.gz -ref {I}/mri_x10.nii.gz -dof 12 -init rigid.mat -nosearch -inweight {I}/oct_fov_x10.nii.gz {c} -omat affine.mat >> log.txt 2>&1",
                       total,
                       f"flirt -in {I}/oct_x10.nii.gz -ref {I}/mri_x10.nii.gz -applyxfm -init affine.mat -out warped_check_x10.nii.gz -interp trilinear >> log.txt 2>&1"]
    if tool == "synthmorph":
        return head + [fs, f"{t} mri_synthmorph register -m affine -j 20 -t aff.lta {I}/oct.nii.gz {I}/mri.nii.gz > log.txt 2>&1",
                       f"mri_vol2vol --mov {I}/oct.nii.gz --targ {I}/mri.nii.gz --lta aff.lta --o warped_check.nii.gz --trilin >> log.txt 2>&1"]
    if tool == "greedy":     # picsl_greedy under Windows Python, as on I58
        Iw, Rw, Pw = (str(x).replace("\\", "/") for x in (inp, run, PY))
        g = lambda a: f'"{Pw}" -c "import picsl_greedy; picsl_greedy.Greedy3D().execute(\'{a}\')"'
        return ["#!/bin/bash", f'cd "{Rw}" || exit 1', "T0=$(date +%s.%N)",
                g(f"-d 3 -a -dof 6 -m NMI -n 100x50x10 -threads 8 -seed 42 -ia-image-centers -mm {Iw}/oct_fov.nii.gz -i {Iw}/mri.nii.gz {Iw}/oct.nii.gz -o {Rw}/rigid.mat") + " > log.txt 2>&1",
                g(f"-d 3 -a -dof 12 -m NMI -n 100x50x10 -threads 8 -seed 42 -ia {Rw}/rigid.mat -mm {Iw}/oct_fov.nii.gz -i {Iw}/mri.nii.gz {Iw}/oct.nii.gz -o {Rw}/affine.mat") + " >> log.txt 2>&1",
                f'T1=$(date +%s.%N); "{Pw}" -c "print($T1 - $T0)" > time.txt',
                g(f"-d 3 -threads 8 -rf {Iw}/mri.nii.gz -rm {Iw}/oct.nii.gz {Rw}/warped_check.nii.gz -r {Rw}/affine.mat") + " >> log.txt 2>&1"]
    raise ValueError(tool)


def prepare():
    runs = [(crop_name(b, "flip2", "C3", 0), t) for b in BLOCKS for t in TOOLS]
    runs += [(crop_name(b, "flip2", "C3", k), "reg_aladin") for b in BLOCKS for k in range(1, SEEDS)]
    lines = ["#!/bin/bash", "# every prepared baseline run without time.txt, one at a time"]
    for crop, tool in sorted(runs, key=lambda r: (TOOLS.index(r[1]), r[0])):
        inp = prepare_inputs(crop)
        run = BL / crop / tool
        run.mkdir(parents=True, exist_ok=True)
        if tool == "elastix":
            for f in ("rigid.txt", "affine.txt"):
                shutil.copy(HERE / "elastix" / f, run / f)
        (run / "run.sh").write_text("\n".join(script(tool, run, inp)) + "\n", encoding="utf-8", newline="\n")
        launch = (f"MSYS_NO_PATHCONV=1 wsl bash {wsl(run / 'run.sh')}" if WIN and tool != "greedy"
                  else f'bash "{str(run / "run.sh").replace(chr(92), "/")}"')
        lines.append(f'[ -f "{str(run / "time.txt").replace(chr(92), "/")}" ] || {launch}; echo "{crop}/{tool} done"')
    (BL / "queue.sh").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(len(runs), "runs prepared:", BL / "queue.sh")


def convert(tool, run, inp):
    import convert_affines as C
    src, dst = nib.load(str(inp / "oct.nii.gz")), nib.load(str(inp / "mri.nii.gz"))
    if tool == "elastix":                 # the parameter files name each other by the path elastix saw (WSL on Windows)
        for n in (0, 1):
            txt = (run / f"TransformParameters.{n}.txt").read_text()
            txt = re.sub(r"/mnt/([a-z])/", lambda m: m.group(1).upper() + ":/", txt) if WIN else txt
            txt = txt.replace("TransformParameters.0.txt", "TransformParameters.0.win.txt")
            (run / f"TransformParameters.{n}.win.txt").write_text(txt)
        return C.elastix_tp_to_mov2fix_ras(run / "TransformParameters.1.win.txt")
    if tool == "flirt":
        s10, d10 = nib.load(str(inp / "oct_x10.nii.gz")), nib.load(str(inp / "mri_x10.nii.gz"))
        return C.unscale_world_transform(C.flirt_mat_to_mov2fix_world(run / "affine.mat", s10, d10), 10.0)
    return {"reg_aladin": lambda: C.niftyreg_txt_to_mov2fix_ras(run / "aff.txt"),
            "greedy": lambda: C.greedy_mat_to_mov2fix_ras(run / "affine.mat"),
            "ants": lambda: C.ants_mat_to_mov2fix_ras(run / "ants_0GenericAffine.mat"),
            "mri_robust_register": lambda: C.lta_to_mov2fix_ras(run / "affine.lta", src, dst),
            "mri_coreg": lambda: C.lta_to_mov2fix_ras(run / "affine_ras.lta", src, dst),
            "synthmorph": lambda: C.lta_to_mov2fix_ras(run / "aff.lta", src, dst)}[tool]()


def check(tool, run, inp, T):
    from octreg import geometry as G
    f = run / ("warped_check_x10.nii.gz" if tool == "flirt" else "warped_check.nii.gz")
    if not f.is_file():
        return {}
    o, m = nib.load(str(inp / "oct.nii.gz")), nib.load(str(inp / "mri.nii.gz"))
    oa = np.asarray(o.dataobj, np.float32)
    mine = G.resample_to(oa, o.affine, m.shape, m.affine, np.linalg.inv(T))
    w = np.nan_to_num(np.asarray(nib.load(str(f)).dataobj, np.float32))
    sel = (w != 0) | (mine != 0)
    out = {"check_r": round(float(np.corrcoef(w[sel], mine[sel])[0, 1]), 4),
           "check_dice": round(float(2 * ((w != 0) & (mine != 0)).sum() / ((w != 0).sum() + (mine != 0).sum())), 4)}
    if tool == "flirt" and (w > np.percentile(oa, 60)).mean() < 1e-3:
        out["check_note"] = "the tool's own resampling holds no OCT tissue inside the crop, as the converted pose puts it outside"
    return out


def collect():
    rows = []
    for crop in sorted(p for p in BL.iterdir() if p.is_dir()):
        inp = crop / "inputs"
        for tool in TOOLS:
            run = crop / tool
            if not (run / "time.txt").is_file():
                continue
            T = convert(tool, run, inp)
            np.savetxt(run / "T_oct2mri.txt", T, fmt="%.10f")
            rows.append({"crop": crop.name, "block": crop.name[1:3], "tool": tool, "setting": SETTING[tool], **error(crop.name[1:3], T),
                         **(check(tool, run, inp, T) if crop.name.endswith("C3_k0") else {}),
                         "seconds": round(float((run / "time.txt").read_text().split()[0]), 1)})
            print(json.dumps(rows[-1])[:200], flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "baselines.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prepare", "collect"])
    {"prepare": prepare, "collect": collect}[ap.parse_args().step]()

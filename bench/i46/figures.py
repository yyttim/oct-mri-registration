#!/usr/bin/env python3
"""freeview panels of the I46 test, made the way bench/figures.py makes the I58 panels.

main/        the C3 crop of the 10 deg FLASH whose distance from the reference under octreg is the lower median among
             k = 0..9 (bench/results/I46/octreg.json, or --scores), on three planes, axial, coronal and sagittal: the MRI
             (<crop>_mri_<plane>.png), the MRI with the line (<crop>_mri_line_<plane>.png), and the OCT on the MRI grid
             through the reference, through octreg (octreg, or octreg<major><minor> when several --octreg folders are given) and
             through each standard tool with a transform, at its declared setting (<tool>) and with its own search
             (<tool> of --search)
conditions/  for every condition of the test, its crop with the lower median distance under octreg, on the axial
             plane: the MRI with the line and the OCT through the reference and through every --octreg folder of runs (the
             runs without --oct-mask as <crop>_nomask)
strip/       every C3 crop of the 10 deg FLASH on the axial plane: the MRI with the line and the OCT through the reference and
             through the first --octreg folder of runs
enface/      the crop of main/ seen in the OCT's own imaging planes, as the authors show the OCT: three OCT planes
             (ENFACE_PLANES, one near the top of three sections) of the masked OCT every method received, at one screen
             pixel per two OCT voxels and freeview's default window, alone (<crop>_p<plane>_oct.png), with the line of main/
             placed through the reference and through octreg (<crop>_p<plane>_oct_line_<tag>.png), and the MRI crop placed
             through each (<crop>_p<plane>_mri_<tag>.png), at freeview's default window and interpolation. FreeSurfer
             places everything: lta_convert writes each transform as an LTA, and freeview resamples the MRI and moves the
             boundary surface through it (:reg=). On the middle plane also the line through each standard tool of main/
             (<crop>_p300_oct_line_<tool>.png): where its boundary does not cross the plane there is no line (listed as
             no_line in figures.json)
The planes pass through the MRI voxel centre nearest the centre of the OCT tissue placed by the reference. The OCT is
box-averaged to the MRI voxel size and resampled by octreg's own resampler, and black means no OCT. The 2-pixel red line is
the white-matter boundary of the authors' manual MRI layer labels, made a surface by FreeSurfer's mri_mc and mris_smooth -n 3
-nw: it comes from no method that is compared. Every panel is one freeview screenshot of the whole plane at two screen pixels
per MRI voxel (0.15 mm), nothing drawn afterwards. Grey levels are linear: each MRI panel at freeview's default window of its
crop (bench.figures.freeview_window), and every OCT panel at the default window of the moving image every method received,
the masked OCT box-averaged to 0.15 mm. The windows, the plane centre and the plane sizes of each crop are written to
figures.json in -o. Needs FreeSurfer 8.2 with its licence (in WSL on Windows).

    python bench/i46/figures.py [-o docs/figures/i46] [--octreg RUNS ...] [--baselines DIR] [--search DIR] [--scores FILE]
                                [--stages main,conditions,strip,enface]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (CONDITIONS, FLIPS, INPUTS, LABELS, OCT, PRIMARY, REPO, RESULTS, RUNS, SEEDS, WORK, crop_file,  # noqa: E402
                    crop_name, reference, tissue)
from bench.figures import WSL, freeview_window, host_path, view  # noqa: E402
from octreg import geometry as G, io  # noqa: E402

K = 2                                           # screen pixels per MRI voxel
ENFACE_PLANES = (102, 300, 498)                 # OCT planes of enface/: the fourth plane of sections 3, 9 and 15
INFO = {}                                       # figures.json: <folder>/<crop> -> what its panels are drawn with


def panel_info(m, crop, work, centre_world, planes, key):
    """The grey windows of the crop and of the moving OCT (work/oct_moving.nii.gz), the voxel centre nearest centre_world
    and the size of each plane, recorded in INFO[key]. -> (crop window, OCT window, centre in MRI world mm)."""
    gm, go = freeview_window(crop), freeview_window(work / "oct_moving.nii.gz")
    v = np.linalg.inv(m.affine) @ np.r_[centre_world, 1]
    c = np.clip(np.floor(v[:3] + 0.5), 0, np.array(m.shape[:3]) - 1)                    # the nearest voxel centre
    centre = (m.affine @ np.r_[c, 1])[:3]
    vox = float(np.linalg.norm(m.affine[:3, :3], axis=0).min())
    INFO[key] = {"mri_window": [round(x, 4) for x in gm], "oct_window": [round(x, 4) for x in go],
                 "plane_centre_mm": [round(float(x), 4) for x in centre],
                 "plane_mm": {p: [round(x * vox, 2) for x in view(m, p, centre)[1]] for p in planes}}
    return gm, go, centre


def panels(crop, transforms, boundary, centre_world, out_dir, prefix, work, planes=("axial", "coronal", "sagittal"),
           plain_mri=True):
    """The panels of one crop: transforms {tag: 4 x 4 OCT world -> MRI world} and boundary a boolean array on the crop grid.
    -> the panel files"""
    out_dir.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    m = nib.load(str(crop))
    vo, vm = io.load_volume(OCT), io.load_volume(crop)
    arr, A = G.resample_iso(vo, max(float(vm.spacing_mm.min()), float(vo.spacing_mm.min())))
    io.save_nifti(arr, A, work / "oct_moving.nii.gz")
    for tag, T in transforms.items():
        io.save_nifti(G.resample_to(arr, A, vm.shape, vm.affine, np.linalg.inv(np.asarray(T, float))), vm.affine,
                      work / f"oct_{tag}.nii.gz")
    hdr = m.header.copy()
    hdr.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(np.asarray(boundary, np.uint8), m.affine, hdr), str(work / "boundary.nii.gz"))
    gm, go, centre = panel_info(m, crop, work, centre_world, planes, f"{out_dir.name}/{prefix}")
    ss = lambda p: f"-ss {host_path(p)} 1 0"
    cmd, expect = [], {}
    for plane in planes:
        lines, wh = view(m, plane, centre)
        lines = [f"-viewsize {wh[0] * K} {wh[1] * K}" if line.startswith("-viewsize") else line for line in lines]
        cmd += ["-v {}:grayscale={!r},{!r}".format(host_path(crop), *gm), *lines]
        if plain_mri:
            cmd.append(ss(out_dir / f"{prefix}_mri_{plane}.png"))
            expect[out_dir / f"{prefix}_mri_{plane}.png"] = (wh[0] * K, wh[1] * K)
        cmd += [f"-f {host_path(work / 'boundary.surf')}:edgecolor=red:edgethickness=2",
                ss(out_dir / f"{prefix}_mri_line_{plane}.png"), "-hide volume"]
        expect[out_dir / f"{prefix}_mri_line_{plane}.png"] = (wh[0] * K, wh[1] * K)
        for tag in transforms:
            cmd += ["-v {}:grayscale={!r},{!r}".format(host_path(work / f"oct_{tag}.nii.gz"), *go),
                    ss(out_dir / f"{prefix}_{tag}_{plane}.png"), "-unload volume"]
            expect[out_dir / f"{prefix}_{tag}_{plane}.png"] = (wh[0] * K, wh[1] * K)
        cmd += ["-unload volume", "-unload surface"]
    (work / "panels.cmd").write_text("\n".join(cmd + ["-quit"]) + "\n", encoding="utf-8", newline="\n")
    sh = ["#!/bin/bash", "export FREESURFER_HOME=${FREESURFER_HOME:-/usr/local/freesurfer/8.2.0}",
          "export FS_LICENSE=${FS_LICENSE:-$HOME/freesurfer/license.txt}",
          "source $FREESURFER_HOME/SetUpFreeSurfer.sh >/dev/null 2>&1", "set -e", f"cd {host_path(work)}",
          "mri_mc boundary.nii.gz 1 boundary_raw.surf > mri_mc.log 2>&1",
          "mris_smooth -n 3 -nw boundary_raw.surf boundary.surf > mris_smooth.log 2>&1",
          "freeview -cmd panels.cmd > freeview.log 2>&1"]
    (work / "panels.sh").write_text("\n".join(sh) + "\n", encoding="utf-8", newline="\n")
    for p in expect:
        p.unlink(missing_ok=True)
    subprocess.run((["wsl"] if WSL else []) + ["bash", host_path(work / "panels.sh")], check=True)
    from PIL import Image
    bad = [f"{p}: {Image.open(p).size if p.is_file() else 'missing'}, expected {wh}" for p, wh in expect.items()
           if not p.is_file() or Image.open(p).size != wh]
    if bad:
        sys.exit("\n".join(bad))
    return sorted(expect)


def tkr(img):
    """FreeSurfer's tkregister vox2ras of a grid, the frame of the surfaces mri_mc writes from it."""
    n, z = np.array(img.shape[:3], float), np.array(img.header.get_zooms()[:3], float)
    return np.array([[-z[0], 0, 0, z[0] * n[0] / 2], [0, 0, z[2], -z[2] * n[2] / 2], [0, -z[1], 0, z[1] * n[1] / 2],
                     [0, 0, 0, 1]])


def crosses(surf, m, T, o, plane):
    """Whether the boundary surface (made on the crop grid m), placed in the OCT world by T, crosses OCT plane `plane` inside
    the OCT field of view."""
    v, f = nib.freesurfer.read_geometry(str(surf))
    v = (np.linalg.inv(o.affine) @ np.linalg.inv(T) @ m.affine @ np.linalg.inv(tkr(m)) @ np.c_[v, np.ones(len(v))].T)[:3].T
    z, xy = v[f, 2], v[f, :2].mean(1)
    inside = (xy >= -0.5).all(1) & (xy <= np.array(o.shape[:2]) - 0.5).all(1)
    return bool((inside & (z.min(1) <= plane) & (z.max(1) >= plane)).any())


def enface(crop, transforms, wm, out_dir, prefix, work, planes=ENFACE_PLANES, mri=True):
    """enface/ (module docstring): transforms {tag: 4 x 4 OCT world -> MRI world}, wm the white-matter label on the crop grid,
    mri whether to make the MRI panels too. -> the panel files"""
    out_dir.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    m, o = nib.load(str(crop)), nib.load(str(OCT))
    hdr = m.header.copy()
    hdr.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(np.asarray(wm, np.uint8), m.affine, hdr), str(work / "boundary.nii.gz"))
    for tag, T in transforms.items():               # lta_convert reads NiftyReg's direction, here MRI world -> OCT world
        np.savetxt(work / f"{tag}_niftyreg.txt", np.linalg.inv(np.asarray(T, float)), fmt="%.10f")
    go, gm = freeview_window(OCT), freeview_window(crop)      # what freeview computes itself, for figures.json
    ss = lambda name: f"-ss {host_path(out_dir / name)} 1 0"
    surf = host_path(work / "boundary.surf")
    reg = lambda tag: host_path(work / f"{tag}_mri2oct.lta")
    line_cmd, mri_cmd, expect = [f"-v {host_path(OCT)}"], [f"-v {host_path(OCT)}:visible=0"], {}
    for plane in planes:
        lines, wh = view(o, "axial", (o.affine @ np.r_[0, 0, plane, 1])[:3])
        size = (wh[0] // 2, wh[1] // 2)                                  # one screen pixel per two OCT voxels
        lines = [f"-viewsize {size[0]} {size[1]}" if line.startswith("-viewsize") else line for line in lines]
        INFO[f"{out_dir.name}/{prefix}_p{plane}"] = {
            "oct_window": [round(x, 4) for x in go], "mri_window": [round(x, 4) for x in gm],
            "plane_mm": [round(o.shape[k] * float(o.header.get_zooms()[k]), 2) for k in (0, 1)]}
        line_cmd += [*lines, ss(f"{prefix}_p{plane}_oct.png")]
        expect[out_dir / f"{prefix}_p{plane}_oct.png"] = size
        for tag in transforms:
            line_cmd += [f"-f {surf}:reg={reg(tag)}:edgecolor=red", ss(f"{prefix}_p{plane}_oct_line_{tag}.png"),
                         "-unload surface"]
            expect[out_dir / f"{prefix}_p{plane}_oct_line_{tag}.png"] = size
        for tag in transforms if mri else ():
            mri_cmd += [f"-v {host_path(crop)}:reg={reg(tag)}", *lines, ss(f"{prefix}_p{plane}_mri_{tag}.png"),
                        "-unload volume"]
            expect[out_dir / f"{prefix}_p{plane}_mri_{tag}.png"] = size
    (work / "lines.cmd").write_text("\n".join(line_cmd + ["-quit"]) + "\n", encoding="utf-8", newline="\n")
    (work / "mri.cmd").write_text("\n".join(mri_cmd + ["-quit"]) + "\n", encoding="utf-8", newline="\n")
    O, C = host_path(OCT), host_path(crop)
    sh = ["#!/bin/bash", "export FREESURFER_HOME=${FREESURFER_HOME:-/usr/local/freesurfer/8.2.0}",
          "export FS_LICENSE=${FS_LICENSE:-$HOME/freesurfer/license.txt}",
          "source $FREESURFER_HOME/SetUpFreeSurfer.sh >/dev/null 2>&1", "set -e", f"cd {host_path(work)}",
          "mri_mc boundary.nii.gz 1 boundary_raw.surf > mri_mc.log 2>&1",
          "mris_smooth -n 3 -nw boundary_raw.surf boundary.surf > mris_smooth.log 2>&1",
          *(f"lta_convert --inniftyreg {tag}_niftyreg.txt --src {O} --trg {C} --outlta {tag}_oct2mri.lta "
            f"> lta_{tag}.log 2>&1 && lta_convert --inlta {tag}_oct2mri.lta --invert --outlta {tag}_mri2oct.lta "
            f">> lta_{tag}.log 2>&1"
            for tag in transforms),
          "freeview -cmd lines.cmd > freeview_lines.log 2>&1",
          *(["freeview -cmd mri.cmd > freeview_mri.log 2>&1"] if mri else [])]
    (work / "enface.sh").write_text("\n".join(sh) + "\n", encoding="utf-8", newline="\n")
    for f in expect:
        f.unlink(missing_ok=True)
    subprocess.run((["wsl"] if WSL else []) + ["bash", host_path(work / "enface.sh")], check=True)
    for plane in planes:
        no = [tag for tag, T in transforms.items() if not crosses(work / "boundary.surf", m, np.asarray(T, float), o, plane)]
        if no:
            INFO[f"{out_dir.name}/{prefix}_p{plane}"]["no_line"] = no
    from PIL import Image
    bad = [f"{f}: {Image.open(f).size if f.is_file() else 'missing'}, expected {wh}" for f, wh in expect.items()
           if not f.is_file() or Image.open(f).size != wh]
    if bad:
        sys.exit("\n".join(bad))
    return sorted(expect)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", type=Path, default=REPO / "docs/figures/i46")
    ap.add_argument("--octreg", type=Path, nargs="+", default=[RUNS],
                    help="folders of octreg runs, one panel column each (default <bench_runs>/I46/runs)")
    ap.add_argument("--baselines", type=Path, default=WORK / "baselines", help="the folder of the standard-tool runs")
    ap.add_argument("--search", type=Path, default=WORK / "baselines_search",
                    help="the folder of the runs of the tools with their own search")
    ap.add_argument("--scores", type=Path, default=RESULTS / "octreg.json", help="the octreg scores that choose the crop")
    ap.add_argument("--stages", default="main,conditions,strip,enface", help="comma-separated folders to make (default: all)")
    a = ap.parse_args()
    stages = a.stages.split(",")
    scores = {x["name"]: x for x in json.loads(a.scores.read_text(encoding="utf-8"))}
    rows = {x["name"]: x for x in json.loads((INPUTS / "crops.json").read_text(encoding="utf-8"))}
    centre = (reference() @ np.r_[tissue().mean(0), 1])[:3]

    def octreg(crop):
        T = {}
        for runs in a.octreg:
            res = json.loads((runs / crop / "result.json").read_text(encoding="utf-8"))
            tag = "octreg" if len(a.octreg) == 1 else "octreg" + "".join(res["octreg_version"].split(".")[:2])
            T[tag] = np.loadtxt(runs / crop / "T_oct2mri.txt")
        return T

    def wm(crop):
        lo, hi = (np.array(x) for x in rows[crop]["crop_voxels"])
        return np.asarray(nib.load(str(LABELS["layers"])).dataobj[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]) == 1   # flip-1 grid

    def median_crop(image, cond, suffix=""):
        errs = sorted((scores[n]["mean_mm"], n) for n in (crop_name(image, cond, k) + suffix for k in range(SEEDS)))
        return errs[(len(errs) - 1) // 2][1]                                               # the lower median

    crop = median_crop(PRIMARY, "C3")
    print("main crop:", crop)
    T = {"reference": reference(), **{k: v for k, v in list(octreg(crop).items())[:1]}}
    for folder in (a.baselines, a.search):
        for d in sorted((folder / crop).glob("*/T_oct2mri.txt")) if (folder / crop).is_dir() else []:
            T[d.parent.name if d.parent.name not in T else d.parent.name + "_search"] = np.loadtxt(d)
    files = panels(crop_file(crop), T, wm(crop), centre, a.out / "main", crop, WORK / "figures" / crop) if "main" in stages else []
    if "enface" in stages:
        files += enface(crop_file(crop), {"reference": reference(), **dict(list(octreg(crop).items())[:1])}, wm(crop),
                        a.out / "enface", crop, WORK / "figures" / "enface" / crop)
        files += enface(crop_file(crop), {k: v for k, v in T.items() if k != "reference" and not k.startswith("octreg")},
                        wm(crop), a.out / "enface", crop, WORK / "figures" / "enface" / crop / "tools", planes=(300,),
                        mri=False)
    conds = [(im, "C3", "") for im in FLIPS] + [(PRIMARY, c, "") for c in CONDITIONS if c != "C3"]
    conds.append((PRIMARY, "C3", "_nomask"))
    for image, cond, suffix in conds if "conditions" in stages else []:
        name = median_crop(image, cond, suffix)
        c = name.removesuffix(suffix) if suffix else name
        files += panels(crop_file(c), {"reference": reference(), **octreg(name)}, wm(c), centre, a.out / "conditions", name,
                        WORK / "figures" / "conditions" / name, planes=("axial",), plain_mri=False)
    for k in range(SEEDS) if "strip" in stages else []:
        c = crop_name(PRIMARY, "C3", k)
        files += panels(crop_file(c), {"reference": reference(), **dict(list(octreg(c).items())[:1])}, wm(c), centre,
                        a.out / "strip", c, WORK / "figures" / "strip" / c, planes=("axial",), plain_mri=False)
    info = a.out / "figures.json"                                     # a partial --stages run keeps the other entries
    old = json.loads(info.read_text(encoding="utf-8")) if info.is_file() else {}
    info.write_text(json.dumps({**old, **INFO}, indent=1) + "\n", encoding="utf-8")
    print(len(files), "panels in", a.out)


if __name__ == "__main__":
    main()

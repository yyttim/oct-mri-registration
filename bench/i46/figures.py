#!/usr/bin/env python3
"""freeview panels of the I46 test (bench/i46/README.md, Pictures), made like the I58 panels of bench/figures.py.

The crop is the C3 crop of the 10 deg FLASH at the lower median distance from R under octreg (--scores, default
bench/results/I46/octreg.json). On OCT planes 102, 300 and 498 the panels show the OCT, the OCT with the white-matter boundary
placed through R and octreg, one panel each and both in one (R red, octreg yellow), and the MRI crop placed through each.
Plane 300 also has the boundary through each standard tool
(--baselines, --search). The panels go to -o/enface and figures.json to -o. Needs FreeSurfer 8.2 with its licence (in WSL on
Windows).

    python bench/i46/figures.py [-o docs/figures/i46] [--octreg RUNS] [--baselines DIR] [--search DIR] [--scores FILE]
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
from common import (INPUTS, LABELS, OCT, PRIMARY, REPO, RESULTS, RUNS, SEEDS, WORK, crop_file, crop_name,  # noqa: E402
                    reference)
from bench.figures import check_sizes, freeview_window, fs_script, view  # noqa: E402
from bench.paths import WSL, host_path  # noqa: E402

ENFACE_PLANES = (102, 300, 498)                 # the fourth plane of slabs 11, 17 and 23
TOOL_PLANE = 300
INFO = {}                                       # figures.json: enface/<crop>_p<plane> -> windows, plane size, no_line


def tkr(img):
    """FreeSurfer's tkregister vox2ras of a grid, the frame of the surfaces mri_mc writes from it."""
    n, z = np.array(img.shape[:3], float), np.array(img.header.get_zooms()[:3], float)
    return np.array([[-z[0], 0, 0, z[0] * n[0] / 2], [0, 0, z[2], -z[2] * n[2] / 2], [0, -z[1], 0, z[1] * n[1] / 2],
                     [0, 0, 0, 1]])


def crosses(surf, m, T, o, plane):
    """Whether the surface made on the crop grid m, placed by T, crosses OCT plane `plane` inside the OCT field of view."""
    v, f = nib.freesurfer.read_geometry(str(surf))
    v = (np.linalg.inv(o.affine) @ np.linalg.inv(T) @ m.affine @ np.linalg.inv(tkr(m)) @ np.c_[v, np.ones(len(v))].T)[:3].T
    z, xy = v[f, 2], v[f, :2].mean(1)
    inside = (xy >= -0.5).all(1) & (xy <= np.array(o.shape[:2]) - 0.5).all(1)
    return bool((inside & (z.min(1) <= plane) & (z.max(1) >= plane)).any())


def enface(crop, main, tools, wm, out_dir, prefix, work):
    """Draw the panels and return their files: main on every plane with MRI panels, tools as lines on TOOL_PLANE only.
    main, tools: {tag: 4 x 4 OCT world -> MRI world}. wm: the white-matter label on the crop grid."""
    out_dir.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    m, o = nib.load(str(crop)), nib.load(str(OCT))
    transforms = {**main, **tools}
    hdr = m.header.copy()
    hdr.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(np.asarray(wm, np.uint8), m.affine, hdr), str(work / "boundary.nii.gz"))
    for tag, T in transforms.items():               # lta_convert --inniftyreg reads MRI world -> OCT world
        np.savetxt(work / f"{tag}_niftyreg.txt", np.linalg.inv(np.asarray(T, float)), fmt="%.10f")
    go, gm = freeview_window(OCT), freeview_window(crop)      # freeview's defaults, explicit so no remembered window wins
    ss = lambda name: f"-ss {host_path(out_dir / name)} 1 0"
    surf = host_path(work / "boundary.surf")
    reg = lambda tag: host_path(work / f"{tag}_mri2oct.lta")
    tags = {p: list(main) + (list(tools) if p == TOOL_PLANE else []) for p in ENFACE_PLANES}
    expect = {}
    line_cmd = ["-v {}:grayscale={!r},{!r}".format(host_path(OCT), *go)]
    mri_cmd = ["-v {}:visible=0".format(host_path(OCT))]
    # -zoom once per freeview session: it multiplies
    once = lambda cmd, lines: [x for x in lines if not (x.startswith("-zoom") and any(c.startswith("-zoom") for c in cmd))]
    for plane in ENFACE_PLANES:
        lines, wh = view(o, "axial", (o.affine @ np.r_[0, 0, plane, 1])[:3])
        size = (wh[0] // 2, wh[1] // 2)                                  # one screen pixel per two OCT voxels
        lines = [f"-viewsize {size[0]} {size[1]}" if line.startswith("-viewsize") else line for line in lines]
        INFO[f"{out_dir.name}/{prefix}_p{plane}"] = {
            "oct_window": [round(x, 4) for x in go], "mri_window": [round(x, 4) for x in gm],
            "plane_mm": [round(o.shape[k] * float(o.header.get_zooms()[k]), 2) for k in (0, 1)]}
        line_cmd += once(line_cmd, lines) + [ss(f"{prefix}_p{plane}_oct.png")]
        expect[out_dir / f"{prefix}_p{plane}_oct.png"] = size
        for tag in tags[plane]:
            line_cmd += [f"-f {surf}:reg={reg(tag)}:edgecolor=red", ss(f"{prefix}_p{plane}_oct_line_{tag}.png"),
                         "-unload surface"]
            expect[out_dir / f"{prefix}_p{plane}_oct_line_{tag}.png"] = size
        line_cmd += [f"-f {surf}:reg={reg(tag)}:edgecolor={colour}" for tag, colour in zip(main, ("red", "yellow"))]
        line_cmd += [ss(f"{prefix}_p{plane}_oct_lines.png")] + ["-unload surface"] * len(main)
        expect[out_dir / f"{prefix}_p{plane}_oct_lines.png"] = size
        for tag in main:
            mri_cmd += ["-v {}:reg={}:grayscale={!r},{!r}".format(host_path(crop), reg(tag), *gm), *once(mri_cmd, lines),
                        ss(f"{prefix}_p{plane}_mri_{tag}.png"), "-unload volume"]
            expect[out_dir / f"{prefix}_p{plane}_mri_{tag}.png"] = size
    (work / "lines.cmd").write_text("\n".join(line_cmd + ["-quit"]) + "\n", encoding="utf-8", newline="\n")
    (work / "mri.cmd").write_text("\n".join(mri_cmd + ["-quit"]) + "\n", encoding="utf-8", newline="\n")
    O, C = host_path(OCT), host_path(crop)
    sh = fs_script(work, [
        "mri_mc boundary.nii.gz 1 boundary_raw.surf > mri_mc.log 2>&1",
        "mris_smooth -n 3 -nw boundary_raw.surf boundary.surf > mris_smooth.log 2>&1",
        *(f"lta_convert --inniftyreg {tag}_niftyreg.txt --src {O} --trg {C} --outlta {tag}_oct2mri.lta "
          f"> lta_{tag}.log 2>&1 && lta_convert --inlta {tag}_oct2mri.lta --invert --outlta {tag}_mri2oct.lta "
          f">> lta_{tag}.log 2>&1"
          for tag in transforms),
        "freeview -cmd lines.cmd > freeview_lines.log 2>&1", "freeview -cmd mri.cmd > freeview_mri.log 2>&1"])
    (work / "enface.sh").write_text("\n".join(sh) + "\n", encoding="utf-8", newline="\n")
    for f in expect:
        f.unlink(missing_ok=True)
    subprocess.run((["wsl"] if WSL else []) + ["bash", host_path(work / "enface.sh")], check=True)
    for plane in ENFACE_PLANES:
        INFO[f"{out_dir.name}/{prefix}_p{plane}"]["no_line"] = [
            tag for tag in tags[plane] if not crosses(work / "boundary.surf", m, np.asarray(transforms[tag], float), o, plane)]
    check_sizes(expect)
    return sorted(expect)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", type=Path, default=REPO / "docs/figures/i46")
    ap.add_argument("--octreg", type=Path, default=RUNS, help="the folder of the octreg runs "
                                                                 "(default <project root>/bench_runs/I46/runs)")
    ap.add_argument("--baselines", type=Path, default=WORK / "baselines", help="the folder of the standard-tool runs")
    ap.add_argument("--search", type=Path, default=WORK / "baselines_search",
                    help="the folder of the runs of the tools with their own search")
    ap.add_argument("--scores", type=Path, default=RESULTS / "octreg.json", help="the octreg scores that choose the crop")
    a = ap.parse_args()
    scores = {x["name"]: x for x in json.loads(a.scores.read_text(encoding="utf-8"))}
    errs = sorted((scores[n]["mean_mm"], n) for n in (crop_name(PRIMARY, "C3", k) for k in range(SEEDS)))
    crop = errs[(len(errs) - 1) // 2][1]                                                  # the lower median
    print("crop:", crop)
    box = next(x for x in json.loads((INPUTS / "crops.json").read_text(encoding="utf-8")) if x["name"] == crop)
    lo, hi = (np.array(x) for x in box["crop_voxels"])
    wm = np.asarray(nib.load(str(LABELS["layers"])).dataobj[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]) == 1   # flip-1 grid
    main = {"reference": reference(), "octreg": np.loadtxt(a.octreg / crop / "T_oct2mri.txt")}
    tools = {f.parent.name: np.loadtxt(f) for folder in (a.baselines, a.search)
             for f in sorted((folder / crop).glob("*/T_oct2mri.txt"))}
    files = enface(crop_file(crop), main, tools, wm, a.out / "enface", crop, WORK / "figures" / "enface" / crop)
    (a.out / "figures.json").write_text(json.dumps(INFO, indent=1) + "\n", encoding="utf-8")
    print(len(files), "panels in", a.out / "enface")


if __name__ == "__main__":
    main()

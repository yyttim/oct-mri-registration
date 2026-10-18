"""The standard registration tools on the I46 test data, each at one setting from its own documentation, and variants
that each make one change (SEARCH, MASK, NEAR). For each run: the inputs, the tool's script, the conversion of its
transform to OCT world -> MRI world and the check of that conversion against the tool's own resampling."""
from __future__ import annotations

import json
import platform
import re
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TOOLS = ["reg_aladin", "greedy", "elastix", "ants", "mri_robust_register", "mri_coreg", "flirt", "synthmorph"]
SETTING = {"reg_aladin": "defaults (image centres)", "greedy": "NMI, -ia-image-centers",
           "elastix": "model zoo, geometric centre", "ants": "antsRegistrationSyN -t a",
           "mri_robust_register": "NMI, default start", "mri_coreg": "defaults",
           "flirt": "normmi, default search, x10 headers", "synthmorph": "-m affine"}
SEARCH = {"flirt_search": "normmi, search -180 180 about every axis, x10 headers",
          "ants_antsai": "antsAI start, antsRegistrationSyN -t a",
          "greedy_search": "NMI, -ia-image-centers, -search 1000 any 20"}
MASK = {"reg_aladin_mask": "defaults (image centres), -fmask", "greedy_mask": "NMI, -ia-image-centers, -mm",
        "elastix_mask": "model zoo, geometric centre, -mMask", "ants_mask": "antsRegistrationSyN -t a, -x NULL,mask",
        "mri_robust_register_mask": "NMI, default start, --maskmov", "mri_coreg_mask": "defaults, --mov-mask",
        "flirt_mask": "normmi, default search, x10 headers, -inweight"}
NO_MASK = {"synthmorph": "mri_synthmorph register has no option for a mask"}
NEAR = {"reg_aladin_near": "defaults, -nac", "greedy_near": "NMI, -ia-identity", "elastix_near": "model zoo, no initialisation",
        "ants_near": "antsRegistrationSyN -t a, -i Identity", "mri_robust_register_near": "NMI, --noinit",
        "mri_coreg_near": "defaults, --regheader", "flirt_near": "normmi, default search, -usesqform, x10 headers",
        "synthmorph_near": "-m affine"}
EXTENDS = {"flirt_search": "flirt", "ants_antsai": "ants", "greedy_search": "greedy",
           **{t: t.removesuffix("_mask") for t in MASK}, **{t: t.removesuffix("_near") for t in NEAR}}
ELASTIX = HERE / "elastix_defaults"
PY = sys.executable
WIN = platform.system() == "Windows"


def wsl(p):
    """The path p as WSL sees it on Windows (/mnt/<drive>/...), else as it is."""
    s = str(Path(p).resolve()).replace("\\", "/")
    return "/mnt/" + s[0].lower() + s[2:] if WIN and len(s) > 1 and s[1] == ":" else s


def made_from(record, sources):
    """Whether the record file exists with these sources. ValueError if it holds other sources."""
    if not record.is_file():
        return False
    if json.loads(record.read_text(encoding="utf-8")) != sources:
        raise ValueError(f"{record}: made from {record.read_text(encoding='utf-8')}, not from {sources}")
    return True


def prepare_inputs(d, oct_path, mri_path):
    """Write d/oct.nii.gz, the OCT box-averaged to the MRI voxel size, d/mri.nii.gz and their x10 copies. Returns d."""
    from octreg import geometry as G, io
    from convert_affines import scale_header
    d, src = Path(d), {"oct": str(Path(oct_path).resolve()), "mri": str(Path(mri_path).resolve())}
    if made_from(d / "inputs.json", src):
        return d
    d.mkdir(parents=True, exist_ok=True)
    vm = io.load_volume(mri_path)
    arr, A = G.resample_iso(io.load_volume(oct_path), float(vm.spacing_mm.min()))
    io.save_nifti(arr, A, d / "oct.nii.gz")
    shutil.copy(mri_path, d / "mri.nii.gz")
    for f in ("oct", "mri"):
        nib.save(scale_header(nib.load(str(d / f"{f}.nii.gz")), 10.0), str(d / f"{f}_x10.nii.gz"))
    (d / "inputs.json").write_text(json.dumps(src, indent=1), encoding="utf-8")
    return d


def prepare_mask(d, mask):
    """Write d/oct_mask.nii.gz, the mask file on the grid of d/oct.nii.gz, and its x10 copy. Returns d."""
    from octreg import geometry as G, io
    from convert_affines import scale_header
    d, src = Path(d), {"mask": str(Path(mask).resolve())}
    if made_from(d / "mask.json", src):
        return d
    o, v = nib.load(str(d / "oct.nii.gz")), io.load_volume(mask)
    # the same resampling as octreg's --oct-mask
    frac, A = G.resample_iso(v, max(float(np.linalg.norm(o.affine[:3, 0])), float(v.spacing_mm.min())), binary=True)
    m = G.resample_to(frac, A, o.shape, o.affine, np.eye(4)) > 0.5
    io.save_nifti(m.astype(np.uint8), o.affine, d / "oct_mask.nii.gz")
    nib.save(scale_header(nib.load(str(d / "oct_mask.nii.gz")), 10.0), str(d / "oct_mask_x10.nii.gz"))
    (d / "mask.json").write_text(json.dumps(src, indent=1), encoding="utf-8")
    return d


def script(tool, run, inp):
    """The lines of run.sh for one tool or variant."""
    R, I = wsl(run), wsl(inp)
    fs, tl = "source ~/baselines/fs_env.sh", "source ~/baselines/tools/env.sh"
    t = "/usr/bin/time -f %e -o time.txt"
    t1, t2 = "/usr/bin/time -f %e -o time_rigid.txt", "/usr/bin/time -f %e -o time_affine.txt"
    times = lambda a, b: "awk '{s+=$1} END {printf \"%.2f\\n\", s}' " + f"{a} {b} > time.txt"
    total = times("time_rigid.txt", "time_affine.txt")
    head = ["#!/bin/bash", "set -o pipefail", f"cd {R} || exit 1"]
    base, mask, near = EXTENDS.get(tool, tool), tool in MASK, tool in NEAR
    if base == "reg_aladin":    # NiftyReg 2.1.2 ignores -fmask (INSTALL.md)
        x = f" -fmask {I}/oct_mask.nii.gz" if mask else " -nac" if near else ""
        return head + [tl, f"{t} reg_aladin -ref {I}/mri.nii.gz -flo {I}/oct.nii.gz{x} -aff {R}/aff.txt -res {R}/warped_tool.nii.gz -omp 8 > log.txt 2>&1",
                       f"reg_resample -ref {I}/mri.nii.gz -flo {I}/oct.nii.gz -trans {R}/aff.txt -inter 1 -pad 0 -res {R}/warped_check.nii.gz -omp 8 >> log.txt 2>&1"]
    if base == "elastix":       # elastix_near differs only in its parameter files (prepare)
        x = f" -mMask {I}/oct_mask.nii.gz" if mask else ""
        return head + [tl, f"{t} elastix -f {I}/mri.nii.gz -m {I}/oct.nii.gz{x} -p {R}/rigid.txt -p {R}/affine.txt "
                           f"-out {R} -threads 12 > log.txt 2>&1", "cp result.1.nii.gz warped_check.nii.gz"]
    if tool == "ants_antsai":    # ants_0GenericAffine.mat includes init.mat (--collapse-output-transforms 1)
        return head + [tl, "export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=12",
                       f"/usr/bin/time -f %e -o time_antsai.txt antsAI -d 3 -m Mattes[ {I}/mri.nii.gz,{I}/oct.nii.gz,32,"
                       f"Regular,0.2 ] -t Rigid[ 0.1 ] -s [ 20,1 ] -p 0 -c 10 -o {R}/init.mat --random-seed 1 -v 1 "
                       f"> log.txt 2>&1",
                       f"/usr/bin/time -f %e -o time_ants.txt antsRegistrationSyN.sh -d 3 -f {I}/mri.nii.gz "
                       f"-m {I}/oct.nii.gz -o {R}/ants_ -t a -n 12 -i {R}/init.mat >> log.txt 2>&1",
                       times("time_antsai.txt", "time_ants.txt"), "cp ants_Warped.nii.gz warped_check.nii.gz"]
    if base == "ants":
        x = f" -x NULL,{I}/oct_mask.nii.gz" if mask else " -i Identity" if near else ""
        return head + [tl, "export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=12",
                       f"{t} antsRegistrationSyN.sh -d 3 -f {I}/mri.nii.gz -m {I}/oct.nii.gz -o {R}/ants_ -t a -n 12{x} "
                       f"> log.txt 2>&1", "cp ants_Warped.nii.gz warped_check.nii.gz"]
    if base == "mri_robust_register":
        m = f" --maskmov {I}/oct_mask.nii.gz" if mask else ""
        x = m + (" --noinit" if near else "")
        return head + [fs, f"{t1} mri_robust_register --mov {I}/oct.nii.gz --dst {I}/mri.nii.gz --lta rigid.lta --cost NMI{x} > log.txt 2>&1",
                       f"{t2} mri_robust_register --mov {I}/oct.nii.gz --dst {I}/mri.nii.gz --lta affine.lta --cost NMI --affine --ixform rigid.lta{m} >> log.txt 2>&1",
                       total,
                       f"mri_vol2vol --mov {I}/oct.nii.gz --targ {I}/mri.nii.gz --lta affine.lta --o warped_check.nii.gz --trilin >> log.txt 2>&1"]
    if base == "mri_coreg":
        m = f" --mov-mask {I}/oct_mask.nii.gz" if mask else ""
        x = m + (" --regheader" if near else "")
        return head + [fs, f"{t1} mri_coreg --mov {I}/oct.nii.gz --ref {I}/mri.nii.gz --reg rigid.lta --dof 6 --threads 20{x} > log.txt 2>&1",
                       f"{t2} mri_coreg --mov {I}/oct.nii.gz --ref {I}/mri.nii.gz --reg affine.lta --dof 12 --init-reg rigid.lta --threads 20{m} >> log.txt 2>&1",
                       total,
                       "lta_convert --inlta affine.lta --outlta affine_ras.lta >> log.txt 2>&1",
                       f"mri_vol2vol --mov {I}/oct.nii.gz --targ {I}/mri.nii.gz --lta affine_ras.lta --o warped_check.nii.gz --trilin >> log.txt 2>&1"]
    if base == "flirt":
        c = "-cost normmi -searchcost normmi -bins 256 -interp trilinear"
        s = "-searchrx -180 180 -searchry -180 180 -searchrz -180 180 " if tool == "flirt_search" else ""
        m = f"-inweight {I}/oct_mask_x10.nii.gz " if mask else ""
        x = m + ("-usesqform " if near else "")
        return head + [tl, "export OMP_NUM_THREADS=8",
                       f"{t1} flirt -v -in {I}/oct_x10.nii.gz -ref {I}/mri_x10.nii.gz -dof 6 {x}{s}{c} -omat rigid.mat "
                       f"> log.txt 2>&1",
                       f"{t2} flirt -v -in {I}/oct_x10.nii.gz -ref {I}/mri_x10.nii.gz -dof 12 -init rigid.mat -nosearch {m}{c} -omat affine.mat >> log.txt 2>&1",
                       total,
                       f"flirt -in {I}/oct_x10.nii.gz -ref {I}/mri_x10.nii.gz -applyxfm -init affine.mat -out warped_check_x10.nii.gz -interp trilinear >> log.txt 2>&1"]
    if base == "synthmorph":
        return head + [fs, f"{t} mri_synthmorph register -m affine -j 20 -t aff.lta {I}/oct.nii.gz {I}/mri.nii.gz > log.txt 2>&1",
                       f"mri_vol2vol --mov {I}/oct.nii.gz --targ {I}/mri.nii.gz --lta aff.lta --o warped_check.nii.gz --trilin >> log.txt 2>&1"]
    if base == "greedy":     # picsl_greedy with this Python, not in WSL
        Iw, Rw, Pw = (str(x).replace("\\", "/") for x in (inp, run, PY))
        g = lambda a: f'"{Pw}" -c "import picsl_greedy; picsl_greedy.Greedy3D().execute(\'{a}\')"'
        s = " -search 1000 any 20" if tool == "greedy_search" else ""
        m = f" -mm {Iw}/oct_mask.nii.gz" if mask else ""
        ia = "-ia-identity" if near else "-ia-image-centers"
        return ["#!/bin/bash", f'cd "{Rw}" || exit 1', "T0=$(date +%s.%N)",
                g(f"-d 3 -a -dof 6 -m NMI -n 100x50x10 -threads 8 -seed 42 {ia}{s}{m} -i {Iw}/mri.nii.gz "
                  f"{Iw}/oct.nii.gz -o {Rw}/rigid.mat") + " > log.txt 2>&1",
                g(f"-d 3 -a -dof 12 -m NMI -n 100x50x10 -threads 8 -seed 42 -ia {Rw}/rigid.mat{m} -i {Iw}/mri.nii.gz {Iw}/oct.nii.gz -o {Rw}/affine.mat") + " >> log.txt 2>&1",
                f'T1=$(date +%s.%N); "{Pw}" -c "print($T1 - $T0)" > time.txt',
                g(f"-d 3 -threads 8 -rf {Iw}/mri.nii.gz -rm {Iw}/oct.nii.gz {Rw}/warped_check.nii.gz -r {Rw}/affine.mat") + " >> log.txt 2>&1"]
    raise ValueError(tool)


def prepare(base, runs, inputs):
    """Write base/<crop>/<tool>/run.sh for every (crop, tool) in runs, and base/queue.sh that runs them."""
    lines = ["#!/bin/bash", "# every prepared baseline run without time.txt, one at a time"]
    order = TOOLS + list(SEARCH) + list(MASK) + list(NEAR)
    for crop, tool in sorted(runs, key=lambda r: (order.index(r[1]), r[0])):
        inp = inputs(crop)
        run = Path(base) / crop / tool
        run.mkdir(parents=True, exist_ok=True)
        if EXTENDS.get(tool, tool) == "elastix":
            for f in ("rigid.txt", "affine.txt"):
                shutil.copy(ELASTIX / f, run / f)
                if tool in NEAR:
                    b, on = (run / f).read_bytes(), b'(AutomaticTransformInitialization "true")'
                    if b.count(on) != 1:
                        raise ValueError(f"{ELASTIX / f}: not one AutomaticTransformInitialization line")
                    (run / f).write_bytes(b.replace(on, b'(AutomaticTransformInitialization "false")'))
        (run / "run.sh").write_text("\n".join(script(tool, run, inp)) + "\n", encoding="utf-8", newline="\n")
        launch = (f"MSYS_NO_PATHCONV=1 wsl bash {wsl(run / 'run.sh')}" if WIN and EXTENDS.get(tool, tool) != "greedy"
                  else f'bash "{str(run / "run.sh").replace(chr(92), "/")}"')
        lines.append(f'[ -f "{str(run / "time.txt").replace(chr(92), "/")}" ] || {launch}; echo "{crop}/{tool} done"')
    (Path(base) / "queue.sh").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(len(runs), "runs prepared:", Path(base) / "queue.sh")


def convert(tool, run, inp):
    """The tool's transform as a 4x4 OCT world -> MRI world (NIfTI RAS mm)."""
    import convert_affines as C
    tool = EXTENDS.get(tool, tool)
    if tool == "elastix":                 # on Windows the files name each other by WSL paths
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
            "mri_robust_register": lambda: C.lta_to_mov2fix_ras(run / "affine.lta"),
            "mri_coreg": lambda: C.lta_to_mov2fix_ras(run / "affine_ras.lta"),
            "synthmorph": lambda: C.lta_to_mov2fix_ras(run / "aff.lta")}[tool]()


def check(tool, run, inp, T):
    """The convention check of a converted transform T: check_r, check_dice and the non-zero voxel counts check_voxels.
    check_r and check_dice are None where neither image holds OCT inside the crop."""
    from octreg import geometry as G
    tool = EXTENDS.get(tool, tool)
    f = run / ("warped_check_x10.nii.gz" if tool == "flirt" else "warped_check.nii.gz")
    if not f.is_file():
        return {}
    o, m = nib.load(str(inp / "oct.nii.gz")), nib.load(str(inp / "mri.nii.gz"))
    mine = G.resample_to(np.asarray(o.dataobj, np.float32), o.affine, m.shape, m.affine, np.linalg.inv(T))
    w = np.nan_to_num(np.asarray(nib.load(str(f)).dataobj, np.float32))
    a, b = w != 0, mine != 0
    sel = a | b
    out = {"check_r": None, "check_dice": None,
           "check_voxels": {"tool": int(a.sum()), "converted": int(b.sum()), "crop": int(w.size)}}
    if sel.any():
        out.update(check_r=round(float(np.corrcoef(w[sel], mine[sel])[0, 1]), 4),
                   check_dice=round(float(2 * (a & b).sum() / (a.sum() + b.sum())), 4))
    return out


def tool_error(e, run):
    """The exception and the last 4 log lines of a run without a readable transform."""
    msg = f"{type(e).__name__}: {e.strerror}: {Path(e.filename).name}" if isinstance(e, OSError) and e.filename else \
        f"{type(e).__name__}: {e}"
    log = run / "log.txt"
    return [msg] + (log.read_text(errors="replace").strip().splitlines()[-4:] if log.is_file() else [])


LOG_ERROR = re.compile(r"ITK ERROR|Internal Error|ERROR:")


def log_error(run):
    """The first line of the run's log that names an error, memory addresses removed, or None."""
    log = run / "log.txt"
    lines = log.read_text(errors="replace").splitlines() if log.is_file() else []
    return next((re.sub(r"\(0x[0-9a-f]+\)", "", s).strip() for s in lines if LOG_ERROR.search(s)), None)


def collect(base, score, checked, tools=TOOLS, inputs=None, start=None):
    """Convert, write and score every finished run base/<crop>/<tool>, with the check where checked(crop). -> list of rows.
    start(crop) is the start S that the moving header carries: the check runs on T, and T S is written and scored."""
    rows, setting = [], {**SETTING, **SEARCH, **MASK, **NEAR}
    for crop in sorted(p for p in Path(base).iterdir() if p.is_dir()):
        inp = Path(inputs(crop.name)) if inputs else crop / "inputs"
        for tool in tools:
            run = crop / tool
            if not (run / "time.txt").is_file():
                continue
            try:
                T = convert(tool, run, inp)
            except (OSError, ValueError, IndexError) as e:
                rows.append({"crop": crop.name, "tool": tool, "setting": setting[tool], "tier": "tool error",
                             "error": tool_error(e, run), "log_error": log_error(run)})
                print(crop.name, tool, "tool error", flush=True)
                continue
            chk = check(tool, run, inp, T) if checked(crop.name) else {}
            if start is not None:
                T = T @ start(crop.name)
            np.savetxt(run / "T_oct2mri.txt", T, fmt="%.10f")
            rows.append({"crop": crop.name, "tool": tool, "setting": setting[tool], **score(crop.name, T), **chk,
                         "seconds": round(float((run / "time.txt").read_text().split()[-1]), 1)})
            print(crop.name, tool, rows[-1]["tier"], flush=True)
    return rows

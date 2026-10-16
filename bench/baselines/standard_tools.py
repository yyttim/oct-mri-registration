"""The standard registration tools on the I46 test data, each at one setting fixed before any run from its own documentation,
with no masks, and three sets of variants, each with one change: the inputs of a run, the command of every tool, the conversion
of its transform and the check of that conversion. The caller (bench/i46/baselines.py) passes its own files and scorer.

    reg_aladin           NiftyReg reg_aladin with its defaults: rigid then affine block matching from the image centres
    greedy               greedy NMI, -dof 6 then -dof 12 from the rigid result, -ia-image-centers
    elastix              the model-zoo rigid and affine parameter files with the geometric-centre start (elastix_defaults/)
    ants                 antsRegistrationSyN.sh -t a: rigid then affine, Mattes MI, its own centre-of-mass start
    mri_robust_register  --cost NMI rigid, then --affine from the rigid result, its default start
    mri_coreg            --dof 6, then --dof 12 from the rigid result, its brute-force start
    flirt                -dof 6 then -dof 12, normmi, default search, headers scaled x10 for its pyramid in mm
    synthmorph           mri_synthmorph register -m affine, which registers in its own space of 256 voxels of 1 mm, where the
                         block spans about 14 x 14 x 9 voxels (FLIRT, whose schedule is also in mm, gets headers scaled x10)

The tools' own global searches, each one of the settings above with the search option of the I58 comparison added and nothing
else changed (SEARCH):

    flirt_search         flirt with -searchrx -180 180 -searchry -180 180 -searchrz -180 180 in the rigid step
    ants_antsai          antsAI rotation search (Mattes MI with 32 bins and 20 % regular sampling, Rigid[0.1], the full circle
                         about each axis in 20 deg steps, no principal axes, 10 iterations per start, seed 1) on the run
                         inputs, then ants with its result as the initial transform (-i) in place of the centre of mass
    greedy_search        greedy with -search 1000 any 20 in the rigid step: 1000 random rigid candidates around the image
                         centres, any rotation and a 20 mm standard deviation of the offset

The tools that take a mask of the moving image, each declared setting with that mask added and nothing else changed (MASK).
The mask is the OCT tissue, the voxels > 0 of the moving image (prepare_mask: oct_mask.nii.gz next to the inputs, and
oct_mask_x10.nii.gz for FLIRT), given to both steps of the two-step tools:

    reg_aladin_mask           -fmask, which NiftyReg 2.1.2 does not use (bench/baselines/INSTALL.md)
    greedy_mask               -mm
    elastix_mask              -mMask, with ErodeMask "false" as in the parameter files
    ants_mask                 -x NULL,<mask>: no fixed mask and the moving mask in both stages, the script's documented form
    mri_robust_register_mask  --maskmov
    mri_coreg_mask            --mov-mask
    flirt_mask                -inweight

mri_synthmorph register has no mask option (NO_MASK).

The declared settings started from the header alignment of the two images instead of their own start (NEAR), for a moving
image whose header carries the start (bench/i46/baselines.py, the near start):

    reg_aladin_near           -nac
    greedy_near               -ia-identity in place of -ia-image-centers
    elastix_near              AutomaticTransformInitialization "false" in both parameter files
    ants_near                 -i Identity in place of the script's centre-of-mass start
    mri_robust_register_near  --noinit in the rigid step
    mri_coreg_near            --regheader in the rigid step, its brute-force search kept
    flirt_near                -usesqform in the rigid step, its default search kept
    synthmorph_near           unchanged

Inputs of a run (prepare_inputs): the fixed image is the MRI crop, the moving image the OCT box-averaged to the MRI voxel size
with its own header (octreg.geometry.resample_iso). FLIRT gets copies of both with headers scaled x10.

Every transform is converted to OCT world -> MRI world (NIfTI RAS mm) by bench/baselines/convert_affines.py. A run whose
transform cannot be read, because the tool stopped, is kept as tier "tool error" with the last lines of its log and the first
log line that names the error (log_error). The convention check (as bench/baselines/rundir.py check on I58) compares the tool's
own resampling of the OCT onto the crop with octreg's resampling through the converted transform: Pearson r over the voxels
where either is non-zero, and the Dice of the two supports. reg_aladin resamples with reg_resample, the LTA tools with
mri_vol2vol, FLIRT with flirt -applyxfm, greedy with greedy -r, elastix and ANTs write their own result image. A variant writes
the same files as the setting it extends, and is converted and checked the same way, on the files it ran on. When the moving
header carries a start S (collect(..., start)), the converted transform maps that header's world, and T S maps the OCT world of
the original header.

The tools run in WSL on Windows (greedy with the Windows Python) and in the shell elsewhere. They need ~/baselines/tools/env.sh
and ~/baselines/fs_env.sh (bench/baselines/INSTALL.md). The two-step tools (greedy, mri_robust_register, mri_coreg, FLIRT) are
timed over both steps, and ants_antsai over the search and the registration.
"""
from __future__ import annotations

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
DECLARED = {"flirt_search": "flirt", "ants_antsai": "ants", "greedy_search": "greedy",     # the setting each variant extends
            **{t: t.removesuffix("_mask") for t in MASK}, **{t: t.removesuffix("_near") for t in NEAR}}
ELASTIX = HERE / "elastix_defaults"
PY = sys.executable
WIN = platform.system() == "Windows"


def wsl(p):
    s = str(Path(p).resolve()).replace("\\", "/")
    return "/mnt/" + s[0].lower() + s[2:] if WIN and len(s) > 1 and s[1] == ":" else s


def prepare_inputs(d, oct_path, mri_path):
    """The inputs of every tool on one crop, in d: oct.nii.gz, mri.nii.gz and their x10 copies. Kept if already there."""
    from octreg import geometry as G, io
    from convert_affines import scale_header
    d = Path(d)
    if (d / "mri_x10.nii.gz").is_file():
        return d
    d.mkdir(parents=True, exist_ok=True)
    vm = io.load_volume(mri_path)
    arr, A = G.resample_iso(io.load_volume(oct_path), float(vm.spacing_mm.min()))
    io.save_nifti(arr, A, d / "oct.nii.gz")
    shutil.copy(mri_path, d / "mri.nii.gz")
    for f in ("oct", "mri"):
        nib.save(scale_header(nib.load(str(d / f"{f}.nii.gz")), 10.0), str(d / f"{f}_x10.nii.gz"))
    return d


def prepare_mask(d):
    """The moving mask of the MASK variants in the input directory d: oct_mask.nii.gz, 1 where oct.nii.gz > 0 (uint8, the same
    header), and oct_mask_x10.nii.gz with the header scaled x10. Kept if already there."""
    from convert_affines import scale_header
    d = Path(d)
    if (d / "oct_mask_x10.nii.gz").is_file():
        return d
    o = nib.load(str(d / "oct.nii.gz"))
    m = nib.Nifti1Image((np.asarray(o.dataobj) > 0).astype(np.uint8), o.affine)
    m.header.set_xyzt_units("mm")
    m.set_qform(o.header.get_qform(), code=int(o.header["qform_code"]))
    m.set_sform(o.header.get_sform(), code=int(o.header["sform_code"]))
    nib.save(m, str(d / "oct_mask.nii.gz"))
    nib.save(scale_header(nib.load(str(d / "oct_mask.nii.gz")), 10.0), str(d / "oct_mask_x10.nii.gz"))
    return d


def script(tool, run, inp):
    """The lines of run.sh for one tool or variant: the registration, time.txt, and the tool's own resampling for the check."""
    R, I = wsl(run), wsl(inp)
    fs, tl = "source ~/baselines/fs_env.sh", "source ~/baselines/tools/env.sh"
    t = "/usr/bin/time -f %e -o time.txt"
    t1, t2 = "/usr/bin/time -f %e -o time_rigid.txt", "/usr/bin/time -f %e -o time_affine.txt"
    times = lambda a, b: "awk '{s+=$1} END {printf \"%.2f\\n\", s}' " + f"{a} {b} > time.txt"
    total = times("time_rigid.txt", "time_affine.txt")
    head = ["#!/bin/bash", "set -o pipefail", f"cd {R} || exit 1"]
    base, mask, near = DECLARED.get(tool, tool), tool in MASK, tool in NEAR
    if base == "reg_aladin":
        x = f" -fmask {I}/oct_mask.nii.gz" if mask else " -nac" if near else ""
        return head + [tl, f"{t} reg_aladin -ref {I}/mri.nii.gz -flo {I}/oct.nii.gz{x} -aff {R}/aff.txt -res {R}/warped_tool.nii.gz -omp 8 > log.txt 2>&1",
                       f"reg_resample -ref {I}/mri.nii.gz -flo {I}/oct.nii.gz -trans {R}/aff.txt -inter 1 -pad 0 -res {R}/warped_check.nii.gz -omp 8 >> log.txt 2>&1"]
    if base == "elastix":       # the start of elastix_near is in its parameter files (prepare)
        x = f" -mMask {I}/oct_mask.nii.gz" if mask else ""
        return head + [tl, f"{t} elastix -f {I}/mri.nii.gz -m {I}/oct.nii.gz{x} -p {R}/rigid.txt -p {R}/affine.txt "
                           f"-out {R} -threads 12 > log.txt 2>&1", "cp result.1.nii.gz warped_check.nii.gz"]
    if tool == "ants_antsai":    # with --collapse-output-transforms 1 the script's ants_0GenericAffine.mat holds init.mat
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
    if base == "greedy":     # picsl_greedy under the Windows Python, as on I58
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
    """run.sh of every (crop, tool) in runs under base/<crop>/<tool>, with inputs(crop) giving the input directory of the crop,
    and base/queue.sh, which runs every run without time.txt, one at a time."""
    lines = ["#!/bin/bash", "# every prepared baseline run without time.txt, one at a time"]
    order = TOOLS + list(SEARCH) + list(MASK) + list(NEAR)
    for crop, tool in sorted(runs, key=lambda r: (order.index(r[1]), r[0])):
        inp = inputs(crop)
        run = Path(base) / crop / tool
        run.mkdir(parents=True, exist_ok=True)
        if DECLARED.get(tool, tool) == "elastix":
            for f in ("rigid.txt", "affine.txt"):
                shutil.copy(ELASTIX / f, run / f)
                if tool in NEAR:        # the header start: no automatic initialisation in either stage
                    b, on = (run / f).read_bytes(), b'(AutomaticTransformInitialization "true")'
                    if b.count(on) != 1:
                        raise ValueError(f"{ELASTIX / f}: not one AutomaticTransformInitialization line")
                    (run / f).write_bytes(b.replace(on, b'(AutomaticTransformInitialization "false")'))
        (run / "run.sh").write_text("\n".join(script(tool, run, inp)) + "\n", encoding="utf-8", newline="\n")
        launch = (f"MSYS_NO_PATHCONV=1 wsl bash {wsl(run / 'run.sh')}" if WIN and DECLARED.get(tool, tool) != "greedy"
                  else f'bash "{str(run / "run.sh").replace(chr(92), "/")}"')
        lines.append(f'[ -f "{str(run / "time.txt").replace(chr(92), "/")}" ] || {launch}; echo "{crop}/{tool} done"')
    (Path(base) / "queue.sh").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(len(runs), "runs prepared:", Path(base) / "queue.sh")


def convert(tool, run, inp):
    """The tool's transform as a 4x4 OCT world -> MRI world (NIfTI RAS mm)."""
    import convert_affines as C
    tool = DECLARED.get(tool, tool)
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
    """The convention check of one converted transform T: check_r, check_dice (and check_note for an empty FLIRT resampling)."""
    from octreg import geometry as G
    tool = DECLARED.get(tool, tool)
    f = run / ("warped_check_x10.nii.gz" if tool == "flirt" else "warped_check.nii.gz")
    if not f.is_file():
        return {}
    o, m = nib.load(str(inp / "oct.nii.gz")), nib.load(str(inp / "mri.nii.gz"))
    oa = np.asarray(o.dataobj, np.float32)
    mine = G.resample_to(oa, o.affine, m.shape, m.affine, np.linalg.inv(T))
    w = np.nan_to_num(np.asarray(nib.load(str(f)).dataobj, np.float32))
    sel = (w != 0) | (mine != 0)
    if not sel.any():
        return {"check_r": None, "check_dice": None,
                "check_note": "neither the tool's own resampling nor the converted one holds OCT inside the crop"}
    out = {"check_r": round(float(np.corrcoef(w[sel], mine[sel])[0, 1]), 4),
           "check_dice": round(float(2 * ((w != 0) & (mine != 0)).sum() / ((w != 0).sum() + (mine != 0).sum())), 4)}
    n_tool, n_mine = int((w != 0).sum()), int((mine != 0).sum())
    if tool == "flirt" and (w > np.percentile(oa, 60)).mean() < 1e-3:
        out.update(check_voxels={"tool": n_tool, "converted": n_mine, "crop": int(w.size)},
                   check_note=f"the tool's own resampling holds OCT in only {n_tool} of the {w.size} crop voxels, and the "
                              f"converted pose in {n_mine}, so r and Dice rest on them")
    return out


def tool_error(e, run):
    """The message of a run without a readable transform: the exception (a missing file by its name) and the last 4 lines of
    its log."""
    msg = f"{type(e).__name__}: {e.strerror}: {Path(e.filename).name}" if isinstance(e, OSError) and e.filename else \
        f"{type(e).__name__}: {e}"
    log = run / "log.txt"
    return [msg] + (log.read_text(errors="replace").strip().splitlines()[-4:] if log.is_file() else [])


LOG_ERROR = re.compile(r"ITK ERROR|Internal Error|ERROR:")       # a log line where a tool names its error


def log_error(run):
    """The first line of the run's log that names an error, memory addresses removed, or None."""
    log = run / "log.txt"
    lines = log.read_text(errors="replace").splitlines() if log.is_file() else []
    return next((re.sub(r"\(0x[0-9a-f]+\)", "", s).strip() for s in lines if LOG_ERROR.search(s)), None)


def collect(base, score, checked, tools=TOOLS, inputs=None, start=None):
    """Every finished run of tools under base (it holds time.txt), converted, written as <run>/T_oct2mri.txt and scored:
    score(crop, T) -> dict, and the convention check on the crops for which checked(crop) is true. inputs(crop) gives the
    input directory of a crop, by default base/<crop>/inputs. start(crop), if given, is the start S that the moving header of
    the crop carries: the check runs on the converted transform as it is, and T S is written and scored. -> list of rows"""
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

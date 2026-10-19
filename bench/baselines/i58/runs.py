#!/usr/bin/env python3
"""The affine baseline runs on the I58 pair: the tool script of every run, the queue that runs them, and the conversion of
each tool's transform into a run directory of octreg with its check.

    python bench/baselines/i58/runs.py prepare [RUN ...]
    bash <project root>/baselines/affine/queue.sh
    python bench/baselines/i58/runs.py convert [RUN ...] [--template RESULT.json] [--force]

RUN is a run directory under <project root>/baselines/affine, for example ants/com (default every run of the table). The
project root is that of bench/paths.py, and the shared inputs are those of prepare_inputs.py in <project root>/baselines/inputs.
prepare  writes RUN/run.sh, the files the runs read besides the shared inputs (the elastix parameter files from
         elastix_defaults, the fixed mask of ones of ants/com_fov, the x10 copies of FLIRT in flirt/inputs_x10 and
         elastix/resample_linear.sh, each kept when it exists) and queue.sh, which runs every run without a time.txt.
         The scripts source the tool environment of INSTALL.md. On Windows the Linux tools run in WSL, and greedy runs with
         this Python.
convert  converts the tool's transform file with convert_affines.py to OCT world -> MRI world (NIfTI RAS mm), writes
         RUN/T_oct2mri.txt and RUN/result.json (rundir.py init) and RUN/check.json, the check of rundir.py against the tool's
         own resampling of the OCT (warped_tool_linear.nii.gz for elastix and NiftyReg, warped_tool.nii.gz otherwise). A
         run with a check.json is kept unless --force is given.
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "bench"), str(REPO / "bench" / "baselines")]
import convert_affines as C  # noqa: E402
import rundir  # noqa: E402
from paths import PROJECT_ROOT, WSL, host_path  # noqa: E402

BASELINES = PROJECT_ROOT / "baselines"
ELASTIX = REPO / "bench" / "baselines" / "elastix_defaults"
PY = Path(sys.executable).as_posix()
TOOLS, FS = "source ~/baselines/tools/env.sh", "source ~/baselines/fs_env.sh"
SUM = "awk '{s+=$1} END {printf \"%.2f\\n\", s}' "
VOL2VOL = 'mri_vol2vol --mov "$MOV" --targ "$DST" --lta {} --o warped_tool.nii.gz --interp trilin > log_vol2vol.txt 2>&1'
FOV = "$IN/oct_fov_0.08mm.nii.gz"

ANTS_HEAD = [TOOLS, "export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=12"]
ANTS_APPLY = ["antsApplyTransforms -d 3 -i $IN/oct_0.08mm.nii.gz -r $IN/mri_0.08mm.nii.gz -t $DIR/ants_0GenericAffine.mat \\",
              "  -o $DIR/warped_tool.nii.gz -n Linear >> log.txt 2>&1"]
REHEAD = ["$BASELINE_PY - <<PY", "import nibabel as nib, numpy as np", 'w = nib.load("warped_tool_x10.nii.gz")',
          'm = nib.load("$BASELINES/inputs/mri_0.08mm.nii.gz")', "assert w.shape == m.shape, (w.shape, m.shape)",
          "out = nib.Nifti1Image(np.asarray(w.dataobj, dtype=np.float32), m.affine)",
          "out.set_qform(m.affine, code=1); out.set_sform(m.affine, code=1)", 'nib.save(out, "warped_tool.nii.gz")',
          'print("warped_tool.nii.gz: header rewritten to the 0.08 mm MRI affine; x10-header copy removed")', "PY",
          "rm -f warped_tool_x10.nii.gz"]
RESAMPLE_LINEAR = [
    "set -e", ". ~/baselines/tools/env.sh", "E=$BASELINES/affine/elastix", "D=$E/$1", "mkdir -p $D/transformix_linear",
    "sed 's/^(FinalBSplineInterpolationOrder 3)/(FinalBSplineInterpolationOrder 1)/' $D/TransformParameters.1.txt "
    "> $D/transformix_linear/TransformParameters.1.linear.txt",
    "transformix -in $BASELINES/inputs/oct_0.08mm.nii.gz -out $D/transformix_linear \\",
    "  -tp $D/transformix_linear/TransformParameters.1.linear.txt -threads 12 > $D/transformix_linear/log.txt 2>&1",
    "mv $D/transformix_linear/result.nii.gz $D/warped_tool_linear.nii.gz"]


def in_dir(run, threads):
    return [f"DIR=$BASELINES/affine/{run}", "IN=$BASELINES/inputs", 'cd "$DIR"', f"echo {threads} > threads.txt"]


def ants_registration(init, redirect):
    mi = "MI[ $IN/mri_0.08mm.nii.gz,$IN/oct_0.08mm.nii.gz,1,32,Regular,0.25 ]"
    conv = "      --convergence [ 1000x500x250x100,1e-6,10 ] --shrink-factors 12x8x4x2 --smoothing-sigmas 4x3x2x1vox \\"
    return ["/usr/bin/time -f %e -o time.txt \\",
            "  antsRegistration --verbose 1 --dimensionality 3 --float 0 --collapse-output-transforms 1 \\",
            "    --output [ $DIR/ants_,$DIR/ants_Warped.nii.gz,$DIR/ants_InverseWarped.nii.gz ] \\",
            "    --interpolation Linear --use-histogram-matching 0 --winsorize-image-intensities [ 0.005,0.995 ] \\",
            f"    --initial-moving-transform {init} \\", f"    --transform Rigid[ 0.1 ] --metric {mi} \\", conv,
            f"    --transform Affine[ 0.1 ] --metric {mi} \\", conv, f"  {redirect} log.txt 2>&1"]


def ants_syn(run, mask=False):
    """antsRegistrationSyN.sh -t a, with mask the fixed mask of ones and the moving FOV mask (-x)."""
    x = ["    -x $DIR/mri_ones_0.08mm.nii.gz,$IN/oct_fov_0.08mm.nii.gz \\"] if mask else []
    return ["set -euo pipefail", *ANTS_HEAD, *in_dir(run, 12), "/usr/bin/time -f %e -o time.txt \\",
            "  antsRegistrationSyN.sh -d 3 -f $IN/mri_0.08mm.nii.gz -m $IN/oct_0.08mm.nii.gz -o $DIR/ants_ -t a -n 12 \\",
            *x, "  > log.txt 2>&1", *ANTS_APPLY]


def ants_identity(run):
    """The antsRegistration command of antsRegistrationSyN.sh -t a, started from the header alignment."""
    return ["set -euo pipefail", *ANTS_HEAD, *in_dir(run, 12), *ants_registration("Identity", ">"), *ANTS_APPLY]


def ants_antsai(run):
    """The antsAI rotation search on 0.15 mm copies, then the antsRegistration command of ants_identity from its result."""
    return ["set -euo pipefail", *ANTS_HEAD, *in_dir(run, 12),
            "[ -f $DIR/mri_0.15mm.nii.gz ] || ResampleImageBySpacing 3 $IN/mri_0.08mm.nii.gz $DIR/mri_0.15mm.nii.gz "
            "0.15 0.15 0.15 1 0 0",
            "/usr/bin/time -f %e -o time_antsai.txt \\",
            "  antsAI -d 3 -m Mattes[ $DIR/mri_0.15mm.nii.gz,$IN/oct_0.15mm.nii.gz,32,Regular,0.2 ] -t Rigid[ 0.1 ] \\",
            "    -s [ 20,1 ] -p 0 -c 10 -o $DIR/init.mat --random-seed 1 -v 1 \\", "  > log.txt 2>&1",
            *ants_registration("$DIR/init.mat", ">>"), *ANTS_APPLY,
            "mv time.txt time_antsregistration.txt", SUM + "time_antsai.txt time_antsregistration.txt > time.txt"]


def elastix(run, mask=""):
    """The model-zoo rigid then affine stages, with the parameter files that prepare writes."""
    return ["set -e", ". ~/baselines/tools/env.sh", "IN=$BASELINES/inputs", f"DIR=$BASELINES/affine/{run}", 'cd "$DIR"',
            "echo 12 > threads.txt", "t0=$(date +%s.%N)",
            f"/usr/bin/time -f %e -o time.txt   elastix -f $IN/mri_0.08mm.nii.gz -m $IN/oct_0.08mm.nii.gz{mask}     "
            "-p $DIR/rigid.txt -p $DIR/affine.txt -out $DIR -threads 12 > log.txt 2>&1",
            "t1=$(date +%s.%N)",
            "awk -v a=$t0 -v b=$t1 'BEGIN{printf \"# date arithmetic around the elastix command: %.1f s\\n\", b-a}' "
            ">> time.txt",
            "cp result.1.nii.gz warped_tool.nii.gz"]


def flirt(run, search="", weight="", x10=True):
    """FLIRT rigid then affine, on the x10 copies or on the 0.08 mm inputs."""
    s = "x10" if x10 else "0.08mm"
    rigid = f"flirt -v -in $MOV -ref $REF -dof 6 {search} {weight} $COMMON -omat rigid.mat"
    affine = f"flirt -v -in $MOV -ref $REF -dof 12 -init rigid.mat -nosearch {weight} $COMMON -omat affine.mat"
    out = "warped_tool_x10.nii.gz" if x10 else "warped_tool.nii.gz"
    return ["set -euo pipefail", TOOLS, "export OMP_NUM_THREADS=8", f"DIR=$BASELINES/affine/{run}",
            "IN=$BASELINES/affine/flirt/inputs_x10" if x10 else "IN=$BASELINES/inputs",
            f"REF=$IN/mri_{s}.nii.gz", f"MOV=$IN/oct_{s}.nii.gz", f"FOV=$IN/oct_fov_{s}.nii.gz", 'cd "$DIR"',
            "echo 8 > threads.txt", "cat $FSLDIR/etc/fslversion > version.txt", ": > log.txt",
            'COMMON="-cost normmi -searchcost normmi -bins 256 -interp trilinear"',
            f'echo "== flirt rigid: {rigid}" >> log.txt', "/usr/bin/time -f %e -o time_rigid.txt \\",
            f"  {rigid} >> log.txt 2>&1", f'echo "== flirt affine: {affine}" >> log.txt',
            "/usr/bin/time -f %e -o time_affine.txt \\", f"  {affine} >> log.txt 2>&1",
            SUM + "time_rigid.txt time_affine.txt > time.txt", 'echo "== flirt applyxfm" >> log.txt',
            f"flirt -in $MOV -ref $REF -applyxfm -init affine.mat -out {out} -interp trilinear >> log.txt 2>&1",
            *(REHEAD if x10 else []), "echo DONE >> log.txt"]


def greedy(run, start, mm=""):
    """greedy rigid then affine through its Python package, and its own reslicing of the OCT."""
    i, r = "$BASELINES/inputs", f"$BASELINES/affine/{run}"

    def g(cmd):
        return f'"$PY" -c "import picsl_greedy; picsl_greedy.Greedy3D().execute(\'{cmd}\')" >> log.txt 2>&1'

    head, io = "-d 3 -a -dof {} -m NMI -n 100x50x10 -threads 8 -seed 42", f"-i {i}/mri_0.08mm.nii.gz {i}/oct_0.08mm.nii.gz"
    return ["set -euo pipefail", f'PY="{PY}"', f'DIR="{r}"', 'cd "$DIR"', "echo 8 > threads.txt", "T0=$(date +%s.%N)",
            g(f"{head.format(6)} {start} {io} -o {r}/rigid.mat"), "T1=$(date +%s.%N)",
            g(f"{head.format(12)} -ia {r}/rigid.mat{mm} {io} -o {r}/affine.mat"), "T2=$(date +%s.%N)",
            "awk -v a=$T0 -v b=$T2 'BEGIN{printf \"%.2f\\n\", b-a}' > time.txt",
            g(f"-d 3 -threads 8 -rf {i}/mri_0.08mm.nii.gz -rm {i}/oct_0.08mm.nii.gz {r}/warped_tool.nii.gz -r {r}/affine.mat")]


def fs_head(run, mask=False):
    return ["set -o pipefail", FS, "INPUTS=$BASELINES/inputs", f"RUN=$BASELINES/affine/{run}", "MOV=$INPUTS/oct_0.08mm.nii.gz",
            "DST=$INPUTS/mri_0.08mm.nii.gz", *(["MASK=$INPUTS/oct_fov_0.08mm.nii.gz"] if mask else []), 'cd "$RUN" || exit 1']


def mri_coreg(run, log1, opt1="", log2="", opt2=""):
    """mri_coreg --dof 6, then --dof 12 from it, with VOX2VOX LTAs converted by lta_convert."""
    reg = '/usr/bin/time -f %e -o time_{0}.txt mri_coreg --mov "$MOV" --ref "$DST" --reg {0}.lta --dof {1}{2} --threads 20 '
    return [*fs_head(run, "MASK" in opt1),
            '{ echo "--threads 20"; echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"; echo "nproc=$(nproc)"; } > threads.txt',
            f'echo "### step 1: rigid (--dof 6), {log1}" > log.txt', reg.format("rigid", 6, opt1) + ">> log.txt 2>&1",
            "echo \"### rigid.lta type line: $(grep '^type' rigid.lta)\" >> log.txt",
            "lta_convert --inlta rigid.lta --outlta rigid_ras.lta >> log.txt 2>&1",
            f'echo "### step 2: affine (--dof 12), initialised with rigid.lta{log2}" >> log.txt',
            reg.format("affine", 12, " --init-reg rigid.lta" + opt2) + ">> log.txt 2>&1",
            "echo \"### affine.lta type line: $(grep '^type' affine.lta)\" >> log.txt",
            "lta_convert --inlta affine.lta --outlta affine_ras.lta >> log.txt 2>&1",
            SUM + "time_rigid.txt time_affine.txt > time.txt", VOL2VOL.format("affine_ras.lta"),
            'echo "done $(date)" >> log.txt']


def mri_robust_register(run, log1, opt1="", log2="", opt2=""):
    """mri_robust_register --cost NMI, rigid, then --affine from it."""
    reg = '/usr/bin/time -f %e -o time_{0}.txt mri_robust_register --mov "$MOV" --dst "$DST" --lta {0}.lta --cost NMI{1} '
    return [*fs_head(run, "MASK" in opt1), '{ echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"; echo "nproc=$(nproc)"; } > threads.txt',
            f'echo "### step 1: rigid, cost NMI, {log1}" > log.txt', reg.format("rigid", opt1) + ">> log.txt 2>&1",
            f'echo "### step 2: affine, cost NMI, initialised with rigid.lta{log2}" >> log.txt',
            reg.format("affine", " --affine --ixform rigid.lta" + opt2) + ">> log.txt 2>&1",
            SUM + "time_rigid.txt time_affine.txt > time.txt", VOL2VOL.format("affine.lta"), 'echo "done $(date)" >> log.txt']


def rigid_only(run):
    """The rigid step of mri_robust_register/noinit, resampled by mri_vol2vol."""
    return [*fs_head(run), "cp ../noinit/rigid.lta rigid.lta", "cp ../noinit/threads.txt threads.txt",
            "cp ../noinit/time_rigid.txt time.txt", VOL2VOL.format("rigid.lta"),
            'echo "rigid stage of ../noinit/log.txt (lines 1-226); resample $(date)" > log.txt',
            "sed -n '1,226p' ../noinit/log.txt >> log.txt"]


def niftyreg(run, opt=""):
    """reg_aladin with its defaults, and reg_resample of the same affine with linear interpolation."""
    return ["set -euo pipefail", TOOLS, "export OMP_NUM_THREADS=8", *in_dir(run, 8), "reg_aladin --version > version.txt 2>&1",
            "/usr/bin/time -f %e -o time.txt \\", f"  reg_aladin -ref $IN/mri_0.08mm.nii.gz -flo $IN/oct_0.08mm.nii.gz{opt} \\",
            "    -aff $DIR/aff.txt -res $DIR/warped_tool.nii.gz -omp 8 \\", "  > log.txt 2>&1",
            "reg_resample -ref $IN/mri_0.08mm.nii.gz -flo $IN/oct_0.08mm.nii.gz -trans $DIR/aff.txt -inter 1 -pad 0 \\",
            "  -res $DIR/warped_tool_linear.nii.gz -omp 8 >> log.txt 2>&1"]


def synthmorph(run):
    """mri_synthmorph register -m affine on the CPU."""
    return [*fs_head(run),
            '{ echo "-j 20 (TensorFlow threads), CPU only (no -g)"; echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"; '
            'echo "nproc=$(nproc)"; } > threads.txt',
            'echo "### mri_synthmorph register -m affine, moving=OCT fixed=MRI, -t moving-to-fixed transform, '
            '-o moved image" > log.txt',
            "/usr/bin/time -f %e -o time.txt mri_synthmorph register -m affine -j 20 -t aff.lta -o warped_tool.nii.gz "
            '"$MOV" "$DST" >> log.txt 2>&1',
            'echo "done $(date)" >> log.txt']


CENTRE, COM, GREEDY_FOV = "default centre init", "default (centre-of-mass) init", " -mm $BASELINES/inputs/oct_fov_0.08mm.nii.gz"
RUNS = {
    "ants/com": (ants_syn, {}),
    "ants/com_fov": (ants_syn, {"mask": True}),
    "ants/identity": (ants_identity, {}),
    "ants/antsai": (ants_antsai, {}),
    "elastix/identity": (elastix, {}),
    "elastix/center": (elastix, {}),
    "elastix/cog": (elastix, {}),
    "elastix/cog_fov": (elastix, {"mask": " -mMask " + FOV}),
    "flirt/search_x10": (flirt, {}),
    "flirt/fullsearch_x10": (flirt, {"search": "-searchrx -180 180 -searchry -180 180 -searchrz -180 180"}),
    "flirt/sqform_x10": (flirt, {"search": "-usesqform -nosearch"}),
    "flirt/search_x10_fov": (flirt, {"weight": "-inweight $FOV"}),
    "flirt/search_native": (flirt, {"x10": False}),
    "greedy/identity": (greedy, {"start": "-ia-identity"}),
    "greedy/centers": (greedy, {"start": "-ia-image-centers"}),
    "greedy/search": (greedy, {"start": "-ia-image-centers -search 1000 any 20"}),
    "greedy/centers_fov": (greedy, {"start": "-ia-image-centers" + GREEDY_FOV, "mm": GREEDY_FOV}),
    "mri_coreg/default": (mri_coreg, {"log1": CENTRE + ", brute-force search on"}),
    "mri_coreg/default_movmask": (mri_coreg, {"log1": CENTRE + ", --mov-mask oct_fov, brute-force search on",
                                              "opt1": ' --mov-mask "$MASK"', "log2": ", --mov-mask oct_fov",
                                              "opt2": ' --mov-mask "$MASK"'}),
    "mri_coreg/regheader": (mri_coreg, {"log1": "--regheader (header alignment), brute-force search on",
                                        "opt1": " --regheader"}),
    "mri_robust_register/default_com": (mri_robust_register, {"log1": COM}),
    "mri_robust_register/default_com_maskmov": (mri_robust_register, {"log1": COM + ", --maskmov oct_fov",
                                                                      "opt1": ' --maskmov "$MASK"',
                                                                      "log2": ", --maskmov oct_fov",
                                                                      "opt2": ' --maskmov "$MASK"'}),
    "mri_robust_register/noinit": (mri_robust_register, {"log1": "--noinit (header alignment)", "opt1": " --noinit"}),
    "mri_robust_register/noinit_rigid_only": (rigid_only, {}),
    "niftyreg/center": (niftyreg, {}),
    "niftyreg/center_fov": (niftyreg, {"opt": " -fmask " + FOV}),
    "niftyreg/nac": (niftyreg, {"opt": " -nac"}),
    "synthmorph/affine": (synthmorph, {}),
}
ELASTIX_START = {"center": "GeometricalCenter", "cog": "CenterOfGravity", "cog_fov": "CenterOfGravity", "identity": None}
LINEAR = ["elastix/center", "elastix/cog", "elastix/cog_fov"]  # resample_linear.sh writes their check image
TRANSFORM = {"ants": "ants_0GenericAffine.mat", "elastix": "TransformParameters.1.txt", "flirt": "affine.mat",
             "greedy": "affine.mat", "mri_coreg": "affine_ras.lta", "mri_robust_register": "affine.lta", "niftyreg": "aff.txt",
             "synthmorph": "aff.lta"}


def native(run):
    """greedy runs in this shell with this Python, the other tools in WSL on Windows."""
    return run.startswith("greedy/")


def script(run, lines, header):
    """The set line of the body, the baselines directory in the form of the shell that runs it, then the rest of the body."""
    b = Path(BASELINES).resolve().as_posix() if native(run) else host_path(BASELINES)
    return "\n".join(["#!/bin/bash", header, lines[0], f'BASELINES="{b}"', *lines[1:]]) + "\n"


def launch(path, run):
    if native(run) or not WSL:
        return f'bash "{Path(path).resolve().as_posix()}"'
    return f"MSYS_NO_PATHCONV=1 wsl bash {host_path(path)}"


def elastix_params(d, start):
    """rigid.txt and affine.txt of the model zoo with the start of the run: GeometricalCenter, CenterOfGravity or the header."""
    on = '(AutomaticTransformInitialization "true")\n(AutomaticTransformInitializationMethod "GeometricalCenter")\n'
    new = on.replace("GeometricalCenter", start) if start else '(AutomaticTransformInitialization "false")\n'
    for f in ("rigid.txt", "affine.txt"):
        txt = (ELASTIX / f).read_text(encoding="utf-8")
        if txt.count(on) != 1:
            raise ValueError(f"{ELASTIX / f}: not one automatic initialisation block")
        if not (d / f).is_file():
            (d / f).write_text(txt.replace(on, new), encoding="utf-8", newline="\n")


def prepare(runs):
    a, inputs = BASELINES / "affine", BASELINES / "inputs"
    queue = ["#!/bin/bash", "# every run of bench/baselines/i58/runs.py without a time.txt, one at a time"]
    for run in runs:
        d, (make, kw) = a / run, RUNS[run]
        d.mkdir(parents=True, exist_ok=True)
        header = f"# {run} of the I58 baseline comparison, written by bench/baselines/i58/runs.py"
        (d / "run.sh").write_text(script(run, make(run, **kw), header), encoding="utf-8", newline="\n")
        if run.startswith("elastix/"):
            elastix_params(d, ELASTIX_START[d.name])
        queue.append(f'[ -f "{(d / "time.txt").resolve().as_posix()}" ] || {launch(d / "run.sh", run)}; echo "{run} done"')
        if run in LINEAR:
            queue.append(f'[ -f "{(d / "warped_tool_linear.nii.gz").resolve().as_posix()}" ] || '
                         f'{launch(a / "elastix" / "resample_linear.sh", run)} {d.name}')
    ones = a / "ants" / "com_fov" / "mri_ones_0.08mm.nii.gz"
    if "ants/com_fov" in runs and not ones.is_file():
        m = nib.load(str(inputs / "mri_0.08mm.nii.gz"))
        img = nib.Nifti1Image(np.ones(m.shape, np.uint8), m.affine, m.header)
        img.set_data_dtype(np.uint8)
        img.set_qform(m.affine, code=1)
        img.set_sform(m.affine, code=1)
        nib.save(img, str(ones))
    if any(r.startswith("flirt/") for r in runs):
        (a / "flirt" / "inputs_x10").mkdir(parents=True, exist_ok=True)
        for f in ("oct", "mri", "oct_fov"):
            out = a / "flirt" / "inputs_x10" / f"{f}_x10.nii.gz"
            if not out.is_file():
                nib.save(C.scale_header(nib.load(str(inputs / f"{f}_0.08mm.nii.gz")), 10.0), str(out))
    if any(r in LINEAR for r in runs) and not (a / "elastix" / "resample_linear.sh").is_file():
        header = "# usage: bash resample_linear.sh <elastix variant>, written by bench/baselines/i58/runs.py"
        (a / "elastix" / "resample_linear.sh").write_text(script("elastix", RESAMPLE_LINEAR, header), encoding="utf-8",
                                                          newline="\n")
    (a / "queue.sh").write_text("\n".join(queue) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(runs)} runs prepared, queue: {a / 'queue.sh'}")


def elastix_transform(d):
    """The affine stage composed with the rigid stage it starts from (its InitialTransformParameterFileName)."""
    tp1 = (d / "TransformParameters.1.txt").read_text(encoding="utf-8")
    init = re.compile(r'\(InitialTransformParameterFileName "([^"]*)"\)')
    m = init.search(tp1)
    if not m or Path(m.group(1)).name != "TransformParameters.0.txt":
        raise ValueError(f"{d}: the affine stage does not start from TransformParameters.0.txt")
    with tempfile.TemporaryDirectory() as tmp:
        alone = Path(tmp) / "TransformParameters.1.txt"
        alone.write_text(init.sub('(InitialTransformParameterFileName "NoInitialTransform")', tp1), encoding="utf-8")
        A = C.elastix_tp_fixed2moving_lps(alone) @ C.elastix_tp_fixed2moving_lps(d / "TransformParameters.0.txt")
    return np.linalg.inv(C.lps_to_ras_matrix(A))


def transform_file(run):
    return "rigid.lta" if run == "mri_robust_register/noinit_rigid_only" else TRANSFORM[run.split("/")[0]]


def transform(run, d):
    """The tool's transform as a 4x4 OCT world -> MRI world (NIfTI RAS mm)."""
    tool, f = run.split("/")[0], d / transform_file(run)
    if tool == "elastix":
        return elastix_transform(d)
    if tool == "flirt":
        x10 = RUNS[run][1].get("x10", True)
        src, s = (BASELINES / "affine" / "flirt" / "inputs_x10", "x10") if x10 else (BASELINES / "inputs", "0.08mm")
        T = C.flirt_mat_to_mov2fix_world(f, nib.load(str(src / f"oct_{s}.nii.gz")), nib.load(str(src / f"mri_{s}.nii.gz")))
        return C.unscale_world_transform(T, 10.0) if x10 else T
    read = {"ants": C.ants_mat_to_mov2fix_ras, "greedy": C.greedy_mat_to_mov2fix_ras, "niftyreg": C.niftyreg_txt_to_mov2fix_ras}
    return read.get(tool, C.lta_to_mov2fix_ras)(f)


def convert(runs, template, force=False):
    inputs = BASELINES / "inputs"
    for run in runs:
        d = BASELINES / "affine" / run
        if (d / "check.json").is_file() and not force:
            print(f"{run}: converted, kept")
            continue
        if not (d / transform_file(run)).is_file():
            print(f"{run}: no {transform_file(run)}, no transform")
            continue
        rundir.init(d, transform(run, d), template)
        warped = d / ("warped_tool_linear.nii.gz" if run.split("/")[0] in ("elastix", "niftyreg") else "warped_tool.nii.gz")
        rundir.check(d, warped, inputs / "oct_0.08mm.nii.gz", inputs / "mri_0.08mm.nii.gz")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prepare", "convert"])
    ap.add_argument("runs", nargs="*", help="run directories under <project root>/baselines/affine (default all)")
    ap.add_argument("--template", type=Path, default=None, help="result.json of an octreg run for rundir.py init")
    ap.add_argument("--force", action="store_true", help="convert: also the runs that have a check.json")
    a = ap.parse_args()
    unknown = [r for r in a.runs if r not in RUNS]
    if unknown:
        ap.error(f"not runs of the table: {unknown}")
    if a.step == "prepare":
        prepare(a.runs or list(RUNS))
    else:
        convert(a.runs or list(RUNS), a.template, a.force)


if __name__ == "__main__":
    main()

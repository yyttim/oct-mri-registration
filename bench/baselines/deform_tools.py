#!/usr/bin/env python3
"""The five deformable baselines of the I58 comparison, each started from octreg's affine.

    python bench/baselines/deform_tools.py --run RUN --inputs DIR [-o OUT] [TOOL ...] [--post-only]

RUN is the octreg run the stage starts from (T_oct2mri.txt, result.json) and DIR its inputs from prepare_inputs.py: the fixed
image mri_0.08mm.nii.gz and the moving image oct_affine_0.08mm.nii.gz, the OCT already on the MRI grid through octreg's
affine, so that every tool starts from the identity and writes its field on the MRI grid. Each TOOL given (all five when
none is) gets the run directory OUT/<tool>/<setting> (default OUT: the folder deform next to DIR) with cmd.sh, its command:

    ants/syn_cc         antsRegistrationSyN.sh -t so (SyN, CC radius 2) -n 8 -p f (single precision), ANTS_RANDOM_SEED=1
    elastix/default     the model-zoo Parameters_BSpline.txt (Mattes MI, 16 mm final grid; BSPLINE), -threads 8
    niftyreg/default    reg_f3d with its defaults (NMI 64 bins, control points every 5 voxels), -omp 8
    convexadam/default  convex_adam_pt (MIND-SSC) with the defaults of convex_adam_MIND.py, float16 on the GPU, seed 0
    greedy/wncc         greedy's quick-start deformable command, -m WNCC 2x2x2 -n 100x50x10 -s 1.732vox 0.707vox -sv,
                        -threads 8 -seed 42 (picsl_greedy)

Each TOOL given then runs, one at a time in the order given: ANTs, elastix and NiftyReg in WSL on Windows (in bash
elsewhere) with ~/baselines/tools/env.sh (INSTALL.md), ConvexAdam and greedy in this Python. A run writes log.txt, time.txt
(seconds of the registration alone), threads.txt and the tool's own linear resampling of the moving image through its field
(CHECKED). The field is converted to octreg's convention, oct2mri_warp.nii.gz: the pull-back field u on the MRI grid in mm
along the MRI world axes (octreg.io.save_field), the moving image at x being moving(x + u(x)). The run directory then gets
octreg's affine (rundir.py init), and the conversion is checked against the tool's own resampling (rundir.py check-field).
--post-only does these last steps on the outputs on disk.

The fields of the tools lie on the fixed grid and map fixed points to moving points:
    ANTs ants_1Warp.nii.gz, elastix deformationField.nii.gz (transformix -def all) and greedy warp.nii.gz are ITK displacement
    images with LPS components d, so u = (-d_0, -d_1, d_2). ANTs' ants_0GenericAffine.mat must be the identity.
    NiftyReg def.nii.gz (reg_transform -def) holds the floating-space position p of each reference voxel in world mm: u = p - x.
    ConvexAdam disp.nii.gz holds the displacement d in voxels along the array axes of the common grid A: u = A[:3, :3] d.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE)]
import convert_affines as C  # noqa: E402
import rundir  # noqa: E402
from octreg import io  # noqa: E402
from standard_tools import WIN, wsl  # noqa: E402

FIXED, MOVING = "mri_0.08mm.nii.gz", "oct_affine_0.08mm.nii.gz"
ITK_FIELD = {"ants": "ants_1Warp.nii.gz", "elastix": "deformationField.nii.gz", "greedy": "warp.nii.gz"}
TOOLS = {"ants": "syn_cc", "elastix": "default", "niftyreg": "default", "convexadam": "default", "greedy": "wncc"}
CHECKED = {"ants": "warped_tool.nii.gz", "elastix": "warped_tool_linear.nii.gz", "niftyreg": "warped_tool_linear.nii.gz",
           "convexadam": "warped_tool.nii.gz", "greedy": "warped_tool.nii.gz"}
IN_PYTHON = ("convexadam", "greedy")
CONVEXADAM = dict(mind_r=1, mind_d=2, lambda_weight=1.25, grid_sp=6, disp_hw=4, selected_niter=80, selected_smooth=0,
                  grid_sp_adam=2, ic=True, use_mask=False)
GREEDY = "-m WNCC 2x2x2 -n 100x50x10 -s 1.732vox 0.707vox -sv -threads 8 -seed 42"
BSPLINE = """\
// The parameter lines of the elastix model zoo models/default/Parameters_BSpline.txt, with ResultImagePixelType "float" and
// ResultImageFormat "nii.gz" in place of "short" and "mhd".
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(UseDirectionCosines "true")
(Registration "MultiResolutionRegistration")
(Interpolator "BSplineInterpolator")
(ResampleInterpolator "FinalBSplineInterpolator")
(Resampler "DefaultResampler")
(FixedImagePyramid "FixedRecursiveImagePyramid")
(MovingImagePyramid "MovingRecursiveImagePyramid")
(Optimizer "AdaptiveStochasticGradientDescent")
(Transform "BSplineTransform")
(Metric "AdvancedMattesMutualInformation")
(FinalGridSpacingInPhysicalUnits 16)
(HowToCombineTransforms "Compose")
(NumberOfHistogramBins 32)
(ErodeMask "false")
(NumberOfResolutions 4)
(MaximumNumberOfIterations 500)
(NumberOfSpatialSamples 2048)
(NewSamplesEveryIteration "true")
(ImageSampler "Random")
(BSplineInterpolationOrder 1)
(FinalBSplineInterpolationOrder 3)
(DefaultPixelValue 0)
(WriteResultImage "true")
(ResultImagePixelType "float")
(ResultImageFormat "nii.gz")
"""


def wsl_script(tool, run, inp):
    """The lines of cmd.sh for ANTs, elastix or NiftyReg: the registration, time.txt, the tool's own resampling and field."""
    R, I = wsl(run), wsl(inp)
    f, m, t = f"{I}/{FIXED}", f"{I}/{MOVING}", "/usr/bin/time -f %e -o time.txt"
    head = ["#!/bin/bash", "set -euo pipefail", "source ~/baselines/tools/env.sh", f"cd {R}", "echo 8 > threads.txt"]
    if tool == "ants":          # the script passes ANTS_RANDOM_SEED on as --random-seed
        return head + ["export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8 ANTS_RANDOM_SEED=1",
                       "antsRegistration --version > version.txt",
                       f"{t} antsRegistrationSyN.sh -d 3 -f {f} -m {m} -o {R}/ants_ -t so -n 8 -p f > log.txt 2>&1",
                       "cp ants_Warped.nii.gz warped_tool.nii.gz"]
    if tool == "elastix":       # transformix: the field, and the moving image with linear in place of cubic interpolation
        lin = "transformix_linear/TransformParameters.0.linear.txt"
        return head + ["elastix --version > version.txt",
                       f"{t} elastix -f {f} -m {m} -p {R}/bspline.txt -out {R} -threads 8 > log.txt 2>&1",
                       "mkdir -p transformix_def transformix_linear",
                       f"transformix -def all -tp {R}/TransformParameters.0.txt -out {R}/transformix_def -threads 8 "
                       "> transformix_def/log.txt 2>&1",
                       "mv transformix_def/deformationField.nii.gz deformationField.nii.gz",
                       f"sed 's/^(FinalBSplineInterpolationOrder 3)/(FinalBSplineInterpolationOrder 1)/' "
                       f"TransformParameters.0.txt > {lin}",
                       f"transformix -in {m} -tp {R}/{lin} -out {R}/transformix_linear -threads 8 "
                       "> transformix_linear/log.txt 2>&1",
                       "mv transformix_linear/result.nii.gz warped_tool_linear.nii.gz"]
    return head + ["export OMP_NUM_THREADS=8", "reg_f3d --version > version.txt 2>&1",     # reg_f3d -res: cubic, nan padding
                   f"{t} reg_f3d -ref {f} -flo {m} -cpp {R}/cpp.nii.gz -res {R}/warped_tool.nii.gz -omp 8 > log.txt 2>&1",
                   f"reg_resample -ref {f} -flo {m} -trans {R}/cpp.nii.gz -inter 1 -pad 0 -res {R}/warped_tool_linear.nii.gz "
                   "-omp 8 >> log.txt 2>&1",
                   f"reg_transform -ref {f} -def {R}/cpp.nii.gz {R}/def.nii.gz >> log.txt 2>&1"]


def python_call(tool, run, inp):
    """The Python code that runs ConvexAdam or greedy (the functions below) on a run directory."""
    return (f"import sys; sys.path.insert(0, {HERE.as_posix()!r}); import deform_tools; "
            f"deform_tools.{tool}({Path(run).as_posix()!r}, {Path(inp).as_posix()!r})")


def write(tool, run, inp):
    """The run directory of a tool: cmd.sh, and bspline.txt for elastix."""
    run.mkdir(parents=True, exist_ok=True)
    if tool == "elastix":
        (run / "bspline.txt").write_text(BSPLINE, encoding="utf-8", newline="\n")
    lines = (["#!/bin/bash", "set -euo pipefail", f'cd "{run.as_posix()}"',
              f'"{Path(sys.executable).as_posix()}" -u -c "{python_call(tool, run, inp)}" > log.txt 2>&1']
             if tool in IN_PYTHON else wsl_script(tool, run, inp))
    (run / "cmd.sh").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def launch(tool, run, inp):
    """Run a tool on its run directory, as its cmd.sh does. -> exit code."""
    if tool in IN_PYTHON:
        with open(run / "log.txt", "w") as log:
            return subprocess.run([sys.executable, "-u", "-c", python_call(tool, run, inp)], stdout=log,
                                  stderr=subprocess.STDOUT, cwd=run).returncode
    return subprocess.run(["wsl", "-e", "bash", wsl(run / "cmd.sh")] if WIN else ["bash", str(run / "cmd.sh")]).returncode


def convexadam(run, inputs):
    """ConvexAdam as `python -m convexAdam.convex_adam_MIND -f FIXED -m MOVING` runs it (CONVEXADAM), timed after a warm-up
    call: disp.nii.gz (as that command writes it), warped_tool.nii.gz (its apply_convex), time.txt, threads.txt."""
    import importlib.metadata
    import torch
    from convexAdam.apply_convex import apply_convex
    from convexAdam.convex_adam_MIND import convex_adam_pt
    run, inp = Path(run), Path(inputs)
    torch.manual_seed(0)
    np.random.seed(0)
    dev = torch.device("cuda")
    print(f"convexAdam {importlib.metadata.version('convexAdam')}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    f, m = nib.load(str(inp / FIXED)), nib.load(str(inp / MOVING))
    convex_adam_pt(torch.rand(32, 32, 32), torch.rand(32, 32, 32), device=dev)        # CUDA start-up, not timed
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    d = convex_adam_pt(torch.from_numpy(f.get_fdata()).float(), torch.from_numpy(m.get_fdata()).float(), dtype=torch.float16,
                       device=dev, verbose=True, **CONVEXADAM)
    torch.cuda.synchronize()
    (run / "time.txt").write_text(f"{time.perf_counter() - t0:.2f}\n")
    (run / "threads.txt").write_text("GPU\n")
    d = np.ascontiguousarray(d, np.float32)
    nib.save(nib.Nifti1Image(d, f.affine), str(run / "disp.nii.gz"))
    io.save_nifti(apply_convex(d, np.asarray(m.dataobj, np.float32)).astype(np.float32), f.affine, run / "warped_tool.nii.gz")


def greedy(run, inputs):
    """greedy (GREEDY) from the identity, timed, then its own resampling through warp.nii.gz: warped_tool.nii.gz, time.txt."""
    import importlib.metadata
    import picsl_greedy
    r, f, m = Path(run).as_posix(), (Path(inputs) / FIXED).as_posix(), (Path(inputs) / MOVING).as_posix()
    print(f"picsl_greedy {importlib.metadata.version('picsl_greedy')}", flush=True)
    Path(run, "threads.txt").write_text("8\n")
    for i, cmd in enumerate((f"-d 3 {GREEDY} -i {f} {m} -o {r}/warp.nii.gz",
                             f"-d 3 -threads 8 -rf {f} -rm {m} {r}/warped_tool.nii.gz -r {r}/warp.nii.gz")):
        print("greedy", cmd, flush=True)
        t0 = time.perf_counter()
        picsl_greedy.Greedy3D().execute(cmd)
        if i == 0:
            Path(run, "time.txt").write_text(f"{time.perf_counter() - t0:.2f}\n")


def vectors(path, mri):
    """A vector image on the MRI grid ([X, Y, Z, 3] or [X, Y, Z, 1, 3]) -> float32 [X, Y, Z, 3]."""
    img = nib.load(str(path))
    a = np.asarray(img.dataobj, np.float32)
    a = a[:, :, :, 0] if a.ndim == 5 else a
    if a.shape != tuple(mri.shape[:3]) + (3,) or not np.allclose(img.affine, mri.affine, atol=1e-3):
        raise ValueError(f"{path}: shape {img.shape}, not a vector image on the MRI grid {mri.shape}")
    return a


def field(tool, run, mri):
    """The field of a finished run in octreg's convention (module docstring): u float32 [3, X, Y, Z] on the MRI grid."""
    if tool in ITK_FIELD:
        if tool == "ants" and not np.allclose(C.ants_mat_to_mov2fix_ras(run / "ants_0GenericAffine.mat"), np.eye(4), atol=1e-6):
            raise ValueError(f"{run / 'ants_0GenericAffine.mat'}: not the identity, so the warp alone is not the transform")
        d = vectors(run / ITK_FIELD[tool], mri)
        return np.stack([-d[..., 0], -d[..., 1], d[..., 2]])
    if tool == "niftyreg":      # x = A (i, j, k) in float32, as NiftyReg computes it
        p, A = vectors(run / "def.nii.gz", mri), mri.affine.astype(np.float32)
        ijk = np.meshgrid(*[np.arange(n, dtype=np.float32) for n in mri.shape[:3]], indexing="ij")
        return np.stack([p[..., c] - (A[c, 0] * ijk[0] + A[c, 1] * ijk[1] + A[c, 2] * ijk[2] + A[c, 3]) for c in range(3)])
    return np.einsum("ab,ijkb->aijk", mri.affine[:3, :3], vectors(run / "disp.nii.gz", mri)).astype(np.float32)


def post(tool, run, inp, octreg_run):
    """oct2mri_warp.nii.gz, rundir.py init with octreg's affine, rundir.py check-field. -> True when the check passes."""
    mri = nib.load(str(inp / FIXED))
    io.save_field(field(tool, run, mri), mri.affine, run / "oct2mri_warp.nii.gz")
    rundir.init(run, octreg_run / "T_oct2mri.txt", octreg_run / "result.json")
    return rundir.check_field(run, run / CHECKED[tool], inp / MOVING, inp / FIXED)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tools", nargs="*", metavar="TOOL", help=f"tools to run: {', '.join(TOOLS)}")
    ap.add_argument("--run", required=True, type=Path, help="the octreg run the stage starts from")
    ap.add_argument("--inputs", required=True, type=Path, help="its inputs, written by prepare_inputs.py")
    ap.add_argument("-o", "--out", type=Path, help="the run directories (default: the folder deform next to --inputs)")
    ap.add_argument("--post-only", action="store_true", help="convert and check the outputs on disk, without running the tools")
    a = ap.parse_args()
    if set(a.tools) - set(TOOLS):
        ap.error(f"unknown tools {sorted(set(a.tools) - set(TOOLS))}; the tools are {', '.join(TOOLS)}")
    for f in (a.run / "T_oct2mri.txt", a.run / "result.json", a.inputs / FIXED, a.inputs / MOVING):
        if not f.is_file():
            ap.error(f"{f}: no such file")
    out, inp, octreg_run, failed = (a.out or a.inputs.parent / "deform").resolve(), a.inputs.resolve(), a.run.resolve(), []
    if not a.post_only:
        for tool in a.tools or TOOLS:
            write(tool, out / tool / TOOLS[tool], inp)
        print(f"cmd.sh of {', '.join(a.tools or TOOLS)} in {out}", flush=True)
    for tool in a.tools:
        run = out / tool / TOOLS[tool]
        for f in ("check_field.json", "oct2mri_warp.nii.gz", "oct_in_mri.nii.gz") + (() if a.post_only else ("time.txt",)):
            (run / f).unlink(missing_ok=True)                   # the results of an earlier run
        if not a.post_only:
            print(f"{time.strftime('%F %T')} {tool}: start", flush=True)
            rc = launch(tool, run, inp)
            print(f"{time.strftime('%F %T')} {tool}: exit {rc}", flush=True)
            if rc:
                failed.append(tool)
                continue
        try:
            ok = post(tool, run, inp, octreg_run)
        except Exception as e:                                 # a missing or malformed output of the tool
            print(f"{tool}: {type(e).__name__}: {e}", flush=True)
            ok = False
        if not ok:
            failed.append(tool)
    if failed:
        sys.exit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()

# Installing the baseline tools

The Linux tools run under WSL2 (Ubuntu 24.04) on Windows 11, and the Python tools under Python 3.13 on Windows. Each tool is its
official release, unchanged.

| tool | version | route |
|---|---|---|
| ANTs | 2.6.5 | release archive `ants-2.6.5-ubuntu-24.04-X64-gcc.zip` (github.com/ANTsX/ANTs/releases) |
| elastix | 5.3.1 | release archive `elastix-5.3.1-ubuntu.zip` (github.com/SuperElastix/elastix/releases), with the parameter files of the elastix model zoo, `models/default` |
| NiftyReg | 2.1.2 | source (github.com/KCL-BMEIS/niftyreg), CMake Release build, CPU with OpenMP |
| FSL | 6.0.7.23 | official installer `fslinstaller.py` (FSL licence, non-commercial) |
| FreeSurfer | 8.2.0 | official release, with `mri_robust_register`, `mri_coreg`, `mri_synthmorph`, `mri_vol2vol`, `lta_convert` |
| greedy | 1.4.0 | `pip install picsl_greedy` (1.4.0.3) |
| ConvexAdam | 0.2.0 | `pip install convexAdam`, run on the GPU through PyTorch |

Two behaviours of these versions affect the runs. NiftyReg 2.1.2 ignores the floating mask of `reg_aladin`, so that mask variant
equals the unmasked run. In this FreeSurfer build `mri_coreg --ras2ras` writes a voxel matrix under a RAS label, so `mri_coreg`
runs with its default VOX2VOX output, which `lta_convert` then converts.

The scripts that run ANTs, elastix, NiftyReg, FSL and FreeSurfer, written by `bench/baselines/standard_tools.py` for `bench/i46`
and by `bench/baselines/deform_tools.py` for the deformable stage, source small files in the WSL home that put the tools on the
path. `~/baselines/tools/env.sh` adds ANTs, elastix and NiftyReg (their `bin` and `lib`) and sources FSL's `etc/fslconf/fsl.sh`.
`~/baselines/fs_env.sh` sets `FREESURFER_HOME`, sources `SetUpFreeSurfer.sh` and sets `FS_LICENSE`. Any setup that puts the same
programs on the path serves as well.

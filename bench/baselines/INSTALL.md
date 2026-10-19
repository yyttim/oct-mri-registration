# Installing the baseline tools

The Linux tools run under WSL2 (Ubuntu 24.04) on Windows 11, and greedy under Python 3.13 on Windows. Each tool is its official
release, unchanged.

| tool | version | route |
|---|---|---|
| ANTs | 2.6.5 | release archive `ants-2.6.5-ubuntu-24.04-X64-gcc.zip` (github.com/ANTsX/ANTs/releases) |
| elastix | 5.3.1 | release archive `elastix-5.3.1-ubuntu.zip` (github.com/SuperElastix/elastix/releases), with the parameter files of the elastix model zoo, `models/default` |
| NiftyReg | 2.1.2 | source (github.com/KCL-BMEIS/niftyreg), CMake Release build, CPU with OpenMP |
| FSL | 6.0.7.23 | official installer `fslinstaller.py` (FSL licence, non-commercial) |
| FreeSurfer | 8.2.0 | official release, with `mri_robust_register`, `mri_coreg`, `mri_synthmorph`, `mri_vol2vol`, `lta_convert` |
| greedy | 1.4.0 | `pip install picsl_greedy` (1.4.0.3) |

Two behaviours of these versions affect the runs. NiftyReg 2.1.2 ignores the floating mask of `reg_aladin`, so that mask variant
equals the unmasked run. In this FreeSurfer build `mri_coreg --ras2ras` writes a voxel matrix under a RAS label, so `mri_coreg`
runs with its default VOX2VOX output, which `lta_convert` then converts.

The scripts that run ANTs, elastix, NiftyReg, FSL and FreeSurfer, written by `bench/baselines/i58/runs.py` for the I58
comparison and by `bench/baselines/standard_tools.py` for `bench/i46`, source two small files in the WSL home that put the
tools on the path. Any setup with the same programs and variables serves as well.

```bash
# ~/baselines/tools/env.sh
export BASELINE_TOOLS="$HOME/baselines/tools"
export ANTSPATH="$BASELINE_TOOLS/ants-2.6.5/bin"
export ELASTIX_HOME="$BASELINE_TOOLS/elastix-5.3.1"
export NIFTYREG_HOME="$BASELINE_TOOLS/niftyreg"
export PATH="$ANTSPATH:$ELASTIX_HOME/bin:$NIFTYREG_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$ANTSPATH/../lib:$ELASTIX_HOME/lib:$NIFTYREG_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export BASELINE_PY="$BASELINE_TOOLS/venv/bin/python"   # a Python with numpy and nibabel
export FSLDIR="$HOME/fsl"
. "$FSLDIR/etc/fslconf/fsl.sh"
export PATH="$PATH:$FSLDIR/share/fsl/bin"
```

```bash
# ~/baselines/fs_env.sh
export FREESURFER_HOME=/usr/local/freesurfer/8.2.0
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
export FS_LICENSE="$HOME/freesurfer/license.txt"
export OMP_NUM_THREADS=20
```

# Installing the baseline tools

The Linux tools run under WSL2 (Ubuntu 24.04) on Windows 11, and the Python tools under Python 3.13 on Windows. Each tool is
its official release, unchanged.

| tool | version | route |
|---|---|---|
| ANTs | 2.6.5 | release archive `ants-2.6.5-ubuntu-24.04-X64-gcc.zip` (github.com/ANTsX/ANTs/releases) |
| elastix | 5.3.1 | release archive `elastix-5.3.1-ubuntu.zip` (github.com/SuperElastix/elastix/releases), with the parameter files of the elastix model zoo, `models/default` |
| NiftyReg | 2.1.2 | source (github.com/KCL-BMEIS/niftyreg), CMake Release build, CPU with OpenMP |
| FSL | 6.0.7.23 | official installer `fslinstaller.py` (FSL licence, non-commercial) |
| FreeSurfer | 8.2.0 | official release, with `mri_robust_register`, `mri_coreg`, `mri_synthmorph`, `mri_vol2vol`, `lta_convert` |
| greedy | 1.4.0 | `pip install picsl_greedy` (1.4.0.3) |
| ConvexAdam | 0.2.0 | `pip install convexAdam`, run on the GPU through PyTorch |

Two behaviours of these versions matter for the results. NiftyReg 2.1.2 ignores the floating mask of `reg_aladin`, and that
of `reg_f3d` without `-vel`, so those mask variants equal the unmasked runs. `reg_f3d -vel` honours the mask. In this
FreeSurfer build `mri_coreg --ras2ras` writes a voxel matrix under a RAS label, so `mri_coreg` runs with its default VOX2VOX
output, which `lta_convert` then converts.

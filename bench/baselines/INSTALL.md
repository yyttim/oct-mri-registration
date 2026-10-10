# Installing the baseline tools

The comparison was run on Windows 11 with WSL2 (Ubuntu 24.04) for the Linux tools and Windows Python 3.13 for the Python
ones. Each tool was installed from its official release, without changes.

| tool | version | route |
|---|---|---|
| ANTs | 2.6.5 | release archive `ants-2.6.5-ubuntu-24.04-X64-gcc.zip` (github.com/ANTsX/ANTs/releases) |
| elastix | 5.3.1 | release archive `elastix-5.3.1-ubuntu.zip` (github.com/SuperElastix/elastix/releases), with the parameter files of the elastix model zoo, `models/default` |
| NiftyReg | 2.1.2 | source (github.com/KCL-BMEIS/niftyreg), CMake Release build, CPU with OpenMP |
| FSL | 6.0.7.23 | official installer `fslinstaller.py` (FSL licence, non-commercial) |
| FreeSurfer | 8.2.0 | official release, with `mri_robust_register`, `mri_coreg`, `mri_synthmorph`, `mri_vol2vol`, `lta_convert` |
| greedy | 1.4.0 | `pip install picsl_greedy` (1.4.0.3) |
| ConvexAdam | 0.2.0 | `pip install convexAdam`, run on the GPU through PyTorch |

Two behaviours of these versions matter for the results. NiftyReg 2.1.2 ignores the floating mask of `reg_aladin` and of
`reg_f3d` without `-vel`, so those mask variants equal the unmasked runs (`reg_f3d -vel` honours it).
`mri_coreg --ras2ras` writes a voxel matrix under a RAS label in this FreeSurfer build, so `mri_coreg` was run with its
default VOX2VOX output and converted with `lta_convert`.

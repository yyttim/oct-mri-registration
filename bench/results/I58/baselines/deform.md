# I58, deformable stage

Every tool starts from octreg's affine: the OCT resampled through it onto the 0.08 mm MRI grid, and the identity as initial
transform. No ground truth exists for this pair, and the overlays of BASELINES.md are the result. J is the Jacobian
determinant of x + u(x), u the field, by finite differences on the field grid, over the MRI foreground of octreg's §1 (a
mask of the MRI alone, on the 0.15 mm grid) sampled onto that grid. folded: the share of those voxels with J <= 0. SD log J:
the standard deviation of log J, with J clipped at 1e-6 where it is not positive (Learn2Reg, Hering et al., IEEE Trans. Med.
Imaging 42:697, 2023). Wall times as each run recorded them, on a machine shared with other jobs, and only indicative.
octreg's is its §6 step time in result.json.

## Main: each tool at the setting its documentation gives (FLIRT with both headers scaled x10)

| method | run | setting | start | field grid | Jacobian min | folded (J <= 0) | SD log J | wall time | hardware |
|---|---|---|---|---|---|---|---|---|---|
| no deformation | `none` | octreg's affine, no field | octreg's affine |  |  |  |  |  |  |
| octreg | `octreg` | §6: strain-limited field on a 5 mm lattice, default parameters | octreg's affine | 0.15 mm | 0.94 | 0.00 % | 0.07 | 21 s | GPU |
| ANTs 2.6.5 | `deform_ants_syn_cc` | antsRegistrationSyN.sh -t so: SyN, CC radius 2, script defaults, no mask | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | 0.01 | 0.00 % | 0.46 | 42 min | CPU, 8 threads |
| elastix 5.3.1 | `deform_elastix_default` | model-zoo Parameters_BSpline: Mattes MI, 16 mm final grid, no mask | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | 1.02 | 0.00 % | 0.02 | 21 s | CPU, 8 threads |
| NiftyReg 2.1.2 reg_f3d | `deform_niftyreg_default` | defaults: NMI 64 bins, control grid 5 voxels (0.4 mm), bending energy 0.001, linear elasticity 0.01, no mask | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | -88.30 | 47.80 % | 6.33 | 3 min | CPU, 8 threads |
| ConvexAdam 0.2.0 | `deform_convexadam_default` | MIND-SSC, the package's shipped defaults, no mask | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | -0.86 | 0.46 % | 1.10 | 8 s | GPU |
| greedy 1.4.0 | `deform_greedy_wncc` | the quick-start deformable command: WNCC 2x2x2, -s 1.732vox 0.707vox, -sv, -n 100x50x10, no mask | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | 0.10 | 0.00 % | 0.17 | 4 min | CPU, 8 threads |

## Supplement: every stored run

| method | run | setting | start | field grid | Jacobian min | folded (J <= 0) | SD log J | wall time | hardware |
|---|---|---|---|---|---|---|---|---|---|
| no deformation | `none` | octreg's affine, no field | octreg's affine |  |  |  |  |  |  |
| octreg | `octreg` | §6: strain-limited field on a 5 mm lattice, default parameters | octreg's affine | 0.15 mm | 0.94 | 0.00 % | 0.07 | 21 s | GPU |
| ANTs 2.6.5 | `deform_ants_syn_cc` | antsRegistrationSyN.sh -t so: SyN, CC radius 2, script defaults, no mask | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | 0.01 | 0.00 % | 0.46 | 42 min | CPU, 8 threads |
| ANTs 2.6.5 | `deform_ants_syn_cc_fov` | as deform_ants_syn_cc, with the OCT FOV as moving mask | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | 0.35 | 0.00 % | 0.19 | 33 min | CPU, 8 threads |
| ANTs 2.6.5 | `deform_ants_syn_mi` | the same SyN stage with Mattes MI (32 bins) | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | -1.62 | < 0.01 % | 0.36 | 21 min | CPU, 8 threads |
| ConvexAdam 0.2.0 | `deform_convexadam_default` | MIND-SSC, the package's shipped defaults, no mask | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | -0.86 | 0.46 % | 1.10 | 8 s | GPU |
| ConvexAdam 0.2.0 | `deform_convexadam_default_fov` | as deform_convexadam_default, with the OCT FOV mask | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | -0.70 | 0.07 % | 0.58 | 17 s | GPU |
| elastix 5.3.1 | `deform_elastix_default` | model-zoo Parameters_BSpline: Mattes MI, 16 mm final grid, no mask | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | 1.02 | 0.00 % | 0.02 | 21 s | CPU, 8 threads |
| elastix 5.3.1 | `deform_elastix_default_fov` | as deform_elastix_default, with the OCT FOV as moving mask | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | 0.98 | 0.00 % | 0.01 | 22 s | CPU, 8 threads |
| elastix 5.3.1 | `deform_elastix_grid5mm` | model-zoo B-spline with a 5 mm final grid | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | 0.70 | 0.00 % | 0.16 | 20 s | CPU, 8 threads |
| greedy 1.4.0 | `deform_greedy_nmi` | as deform_greedy_wncc with -m NMI | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | -0.27 | 2.83 % | 2.73 | 4 min | CPU, 8 threads |
| greedy 1.4.0 | `deform_greedy_wncc` | the quick-start deformable command: WNCC 2x2x2, -s 1.732vox 0.707vox, -sv, -n 100x50x10, no mask | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | 0.10 | 0.00 % | 0.17 | 4 min | CPU, 8 threads |
| greedy 1.4.0 | `deform_greedy_wncc_fov` | as deform_greedy_wncc, with the OCT FOV as moving mask | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | 0.18 | 0.00 % | 0.16 | 3 min | CPU, 8 threads |
| greedy 1.4.0 | `deform_greedy_wncc_mm` | as deform_greedy_wncc with the smoothing in mm (-s 2mm 0.7mm) | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | 0.20 | 0.00 % | 0.05 | 3 min | CPU, 8 threads |
| NiftyReg 2.1.2 reg_f3d | `deform_niftyreg_default` | defaults: NMI 64 bins, control grid 5 voxels (0.4 mm), bending energy 0.001, linear elasticity 0.01, no mask | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | -88.30 | 47.80 % | 6.33 | 3 min | CPU, 8 threads |
| NiftyReg 2.1.2 reg_f3d | `deform_niftyreg_sx5mm` | defaults with a 5 mm control grid (-sx 5) | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | -0.48 | 3.51 % | 2.76 | 4 min | CPU, 8 threads |
| NiftyReg 2.1.2 reg_f3d | `deform_niftyreg_vel_fov` | -vel (symmetric, velocity field) with the OCT FOV floating mask | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | -14049.22 | 0.85 % | 2.55 | 2.2 h | CPU, 8 threads |

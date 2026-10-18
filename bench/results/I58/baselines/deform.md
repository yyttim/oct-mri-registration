# I58, deformable stage

Every tool starts from octreg's affine: the OCT resampled through it onto the 0.08 mm MRI grid, and the identity as initial
transform. The pair has no reference, and BASELINES.md shows the overlays. J is the Jacobian determinant of x + u(x), u the
field, by finite differences on the field grid, over the MRI foreground of octreg's §1 (a mask of the MRI alone, computed on
the 0.15 mm base grid) sampled onto the field grid. folded: the share of those voxels with J <= 0. SD log J: the standard
deviation of log J, with J clipped below at 1e-6 (Learn2Reg, Hering et al., IEEE Trans. Med. Imaging 42:697, 2023). Wall
times as each run recorded them, on a machine shared with other jobs. octreg's is its §5 step time in result.json.

## Main: each tool at the setting its documentation gives

| method | run | setting | start | field grid | Jacobian min | folded (J <= 0) | SD log J | wall time | hardware |
|---|---|---|---|---|---|---|---|---|---|
| no deformation | `none` | octreg's affine, no field | octreg's affine |  |  |  |  |  |  |
| octreg | `octreg` | §5: displacement field on a 5 mm lattice under a strain bound, default parameters | octreg's affine | 0.15 mm | 0.84 | 0.00 % | 0.06 | 58 s | GPU |
| ANTs 2.6.5 | `deform_ants_syn_cc` | antsRegistrationSyN.sh -t so: SyN, CC radius 2, script defaults in single precision (-p f), no mask | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | -0.05 | < 0.01 % | 0.47 | 44 min | CPU, 8 threads |
| elastix 5.3.1 | `deform_elastix_default` | model-zoo Parameters_BSpline: Mattes MI, 16 mm final grid, no mask | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | 0.99 | 0.00 % | 0.02 | 24 s | CPU, 8 threads |
| NiftyReg 2.1.2 reg_f3d | `deform_niftyreg_default` | defaults: NMI 64 bins, control grid 5 voxels (0.4 mm), bending energy 0.001, linear elasticity 0.01, no mask | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | -67.00 | 47.23 % | 6.35 | 3 min | CPU, 8 threads |
| ConvexAdam 0.2.0 | `deform_convexadam_default` | MIND-SSC, the package's shipped defaults, no mask | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | -0.80 | 0.49 % | 1.12 | 7 s | GPU |
| greedy 1.4.0 | `deform_greedy_wncc` | the quick-start deformable command: WNCC 2x2x2, -s 1.732vox 0.707vox, -sv, -n 100x50x10, no mask | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | 0.12 | 0.00 % | 0.17 | 3 min | CPU, 8 threads |

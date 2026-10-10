| tool | variant | wall s | interior (mm) | boundary (mm) | within 0.3 mm | F | two-class, core | field median / p95 / max (mm) | Jacobian min | folded voxels | SD log J | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | no deformation | n/a | 0.250 | 0.285 | 51 % | 0.1029 | 0.0581 | 0 | 1 | 0 % | 0 | octreg affine, no field |
| octreg | §6 | 20 | 0.124 | 0.089 | 80 % | 0.1086 | 0.0641 | 0.28 / 0.72 / 1.05 | 0.94 | 0.00 % | 0.072 | octreg §6: strain-limited field on a 5 mm lattice from interior matches and surface-edge offsets |
| ants | syn_cc | 5251 | 0.049 | 0.024 | 93 % | 0.1367 | 0.0921 | 0.68 / 1.76 / 3.41 | 0.03 | 0.00 % | 0.464 | antsRegistrationSyN.sh -t so (SyN, CC radius 4) from octreg's affine, both images on the MRI grid |
| ants | syn_cc_fov | 2908 | 0.063 | 0.027 | 89 % | 0.1206 | 0.0725 | 0.29 / 0.79 / 1.62 | 0.34 | 0.00 % | 0.207 | SyN CC with the OCT field-of-view mask |
| ants | syn_mi | 4397 | 0.289 | 0.351 | 44 % | 0.1141 | 0.0907 | 0.70 / 2.02 / 4.14 | 0.10 | 0.00 % | 0.355 | the same SyN stage with Mattes MI |
| convexadam | default | 11 | 0.218 | 0.051 | 81 % | 0.1065 | 0.0426 | 0.52 / 1.44 / 2.46 | -1.09 | 0.45 % | 1.088 | ConvexAdam MIND-SSC, shipped defaults, GPU |
| convexadam | default_fov | 13 | 0.220 | 0.052 | 81 % | 0.1068 | 0.0531 | 0.48 / 0.97 / 1.92 | -0.54 | 0.08 % | 0.591 | ConvexAdam MIND-SSC with the field-of-view mask |
| elastix | default | 54 | 0.950 | 0.465 | 33 % | 0.0946 | 0.0652 | 1.28 / 2.23 / 2.59 | 1.00 | 0.00 % | 0.052 | model-zoo B-spline (Mattes MI), 16 mm grid |
| elastix | default_fov | 43 | 0.367 | 0.254 | 55 % | 0.1031 | 0.0739 | 0.27 / 0.49 / 0.59 | 1.00 | 0.00 % | 0.011 | model-zoo B-spline, 16 mm grid, with the field-of-view mask |
| elastix | grid5mm | 40 | 0.941 | 0.706 | 24 % | 0.0951 | 0.0860 | 2.11 / 4.15 / 6.37 | 0.75 | 0.00 % | 0.260 | model-zoo B-spline with a 5 mm grid (physical-unit variant) |
| greedy | nmi | 118 | 0.217 | 0.324 | 46 % | 0.1058 | 0.0776 | 0.15 / 1.48 / 5.47 | -0.26 | 2.82 % | 2.727 | greedy deformable with NMI |
| greedy | wncc | 245 | 0.187 | 0.039 | 86 % | 0.1068 | 0.0220 | 0.13 / 0.64 / 1.89 | 0.12 | 0.00 % | 0.163 | greedy deformable, WNCC 2x2x2, quick-start smoothing in voxels |
| greedy | wncc_fov | 221 | 0.181 | 0.033 | 89 % | 0.1076 | 0.0038 | 0.14 / 0.66 / 1.53 | 0.14 | 0.00 % | 0.171 | greedy deformable, WNCC, with the field-of-view mask |
| greedy | wncc_mm | 251 | 0.221 | 0.111 | 66 % | 0.1050 | 0.0580 | 0.19 / 0.52 / 0.78 | 0.30 | 0.00 % | 0.046 | greedy deformable, WNCC, smoothing in mm |
| niftyreg | default | 535 | 0.158 | 1.018 | 3 % | 0.0760 | -0.0717 | 3.07 / 7.17 / 13.00 | -71.00 | 47.12 % | 6.345 | reg_f3d defaults (NMI, control grid -5 voxels = 0.4 mm): the image collapses |
| niftyreg | sx5mm | 500 | 1.232 | 0.615 | 25 % | 0.0925 | 0.0903 | 4.82 / 9.41 / 12.70 | -0.70 | 4.72 % | 3.147 | reg_f3d with a 5 mm control grid |
| niftyreg | vel_fov | 3793 | 0.376 | 0.962 | 6 % | 0.0674 | -0.1532 | 2.52 / 5.03 / 8.85 | -2245.33 | 0.89 % | 2.595 | reg_f3d -vel (symmetric F3D2) with the field-of-view mask, the one form in which the mask is honoured |

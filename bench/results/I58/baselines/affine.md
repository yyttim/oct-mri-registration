# I58, affine stage

The pair has no reference, so the table has no accuracy column, and BASELINES.md shows the overlays. Scale per OCT axis: the
length of the transform applied to a unit step along each array axis of the OCT header (1 = voxel size kept). det: the
determinant of its 3x3 (1 = volume kept). Wall times as each run recorded them, on a 24-core machine shared with other jobs.
octreg's is the sum of its §1 to §4 step times in result.json. FLIRT's main run has both headers scaled x10, because its
schedule resamples the images to fixed voxel sizes in mm, and its run on the unscaled headers is flirt_search_native.

## Main: each tool at the setting its documentation gives (FLIRT with both headers scaled x10)

| method | run | setting | start | scale per OCT axis | det | wall time | hardware |
|---|---|---|---|---|---|---|---|
| image centres aligned (the start) | `reference_centres` | not a registration: the OCT translated so that the centres of the two image boxes coincide, header orientation kept | image centres aligned | 1.00 / 1.00 / 1.00 | 1.00 |  |  |
| octreg | `octreg` | §1 to §4 with default parameters | own orientation search (8,000 rotations) inside the MRI crop | 0.96 / 0.95 / 0.97 | 0.88 | 11 min | GPU |
| NiftyReg 2.1.2 reg_aladin | `niftyreg_center` | defaults: symmetric block matching, rigid then affine | image centres (default) | 0.97 / 0.96 / 0.98 | 0.91 | 5 min | CPU, 8 threads |
| greedy 1.4.0 | `greedy_centers` | NMI, -dof 6 then -dof 12, -n 100x50x10, no mask | image centres (-ia-image-centers) | 1.19 / 0.93 / 0.90 | 0.91 | 63 s | CPU, 8 threads |
| elastix 5.3.1 | `elastix_center` | model-zoo Parameters_Rigid then Parameters_Affine (AdvancedMattesMutualInformation), no mask | geometric centres (GeometricalCenter) | 0.92 / 0.84 / 0.97 | 0.74 | 26 s | CPU, 12 threads |
| FreeSurfer 8.2.0 mri_robust_register | `mri_robust_register_default_com` | --cost NMI, rigid then --affine --ixform, tool defaults | centres of mass (default) | 0.99 / 1.00 / 1.07 | 1.06 | 14 min | CPU, 20 threads |
| ANTs 2.6.5 | `ants_com` | antsRegistrationSyN.sh -t a: rigid then affine, Mattes MI, the script's large-image schedule, no mask | centres of mass (script default) | 0.86 / 0.62 / 0.92 | 0.49 | 45 s | CPU, 12 threads |
| FreeSurfer 8.2.0 mri_coreg | `mri_coreg_default` | NMI, --dof 6 then --dof 12 --init-reg, tool defaults | volume centres + the default brute-force search | 0.92 / 0.81 / 1.00 | 0.74 | 22 min | CPU, 20 threads |
| FSL FLIRT 6.0.7.23 | `flirt_search_x10` | normmi, -dof 6 then -dof 12 -nosearch, both headers scaled x10 because FLIRT's schedule resamples the images to fixed voxel sizes in mm (an adaptation FSL does not document) | FLIRT default rotation search (±90°, coarse 60°, fine 18°) | 0.76 / 0.76 / 0.67 | 0.39 | 53 min | CPU (FLIRT runs single-threaded) |
| FreeSurfer 8.2.0 mri_synthmorph | `synthmorph_affine` | register -m affine as documented. A model of 1 mm whole-brain images, run on a 0.08 mm block | none (the network predicts the affine) | 0.82 / 0.62 / 1.20 | 0.49 | 38 s | CPU, 20 threads |

## Supplement: every stored run

| method | run | setting | start | scale per OCT axis | det | wall time | hardware |
|---|---|---|---|---|---|---|---|
| image centres aligned (the start) | `reference_centres` | not a registration: the OCT translated so that the centres of the two image boxes coincide, header orientation kept | image centres aligned | 1.00 / 1.00 / 1.00 | 1.00 |  |  |
| octreg | `octreg` | §1 to §4 with default parameters | own orientation search (8,000 rotations) inside the MRI crop | 0.96 / 0.95 / 0.97 | 0.88 | 11 min | GPU |
| ANTs 2.6.5 | `ants_antsai` | antsAI rigid rotation search (Mattes MI, 0.15 mm copies), then the stages of ants_com | antsAI global rotation search | 1.03 / 1.00 / 1.03 | 1.07 | 4.9 h | CPU, 12 threads |
| ANTs 2.6.5 | `ants_com` | antsRegistrationSyN.sh -t a: rigid then affine, Mattes MI, the script's large-image schedule, no mask | centres of mass (script default) | 0.86 / 0.62 / 0.92 | 0.49 | 45 s | CPU, 12 threads |
| ANTs 2.6.5 | `ants_com_fov` | as ants_com, with the OCT field-of-view (FOV) mask | centres of mass | 1.28 / 1.01 / 1.10 | 1.39 | 68 s | CPU, 12 threads |
| ANTs 2.6.5 | `ants_identity` | as ants_com | header alignment (identity) | 0.91 / 1.03 / 0.99 | 0.93 | 25 s | CPU, 12 threads |
| elastix 5.3.1 | `elastix_center` | model-zoo Parameters_Rigid then Parameters_Affine (AdvancedMattesMutualInformation), no mask | geometric centres (GeometricalCenter) | 0.92 / 0.84 / 0.97 | 0.74 | 26 s | CPU, 12 threads |
| elastix 5.3.1 | `elastix_cog` | as elastix_center | centres of gravity (CenterOfGravity) | 0.90 / 0.82 / 0.97 | 0.71 | 32 s | CPU, 12 threads |
| elastix 5.3.1 | `elastix_cog_fov` | as elastix_center, with the OCT FOV mask | centres of gravity | 1.03 / 0.99 / 1.01 | 1.03 | 29 s | CPU, 12 threads |
| FSL FLIRT 6.0.7.23 | `flirt_fullsearch_x10` | as flirt_search_x10 | rotation search over ±180° | 0.73 / 0.73 / 0.73 | 0.39 | 2.4 h | CPU (FLIRT runs single-threaded) |
| FSL FLIRT 6.0.7.23 | `flirt_search_native` | normmi, -dof 6 then -dof 12, as documented, on the unscaled 0.08 mm headers | FLIRT default rotation search | 0.76 / 0.78 / 0.61 | 0.36 | 63 min | CPU (FLIRT runs single-threaded) |
| FSL FLIRT 6.0.7.23 | `flirt_search_x10` | normmi, -dof 6 then -dof 12 -nosearch, both headers scaled x10 because FLIRT's schedule resamples the images to fixed voxel sizes in mm (an adaptation FSL does not document) | FLIRT default rotation search (±90°, coarse 60°, fine 18°) | 0.76 / 0.76 / 0.67 | 0.39 | 53 min | CPU (FLIRT runs single-threaded) |
| FSL FLIRT 6.0.7.23 | `flirt_search_x10_fov` | as flirt_search_x10, with -inweight (OCT FOV) | FLIRT default rotation search | 0.86 / 0.81 / 0.78 | 0.54 | 74 min | CPU (FLIRT runs single-threaded) |
| FSL FLIRT 6.0.7.23 | `flirt_sqform_x10` | as flirt_search_x10 with -usesqform -nosearch | header alignment | 1.17 / 1.18 / 1.17 | 1.62 | 23 min | CPU (FLIRT runs single-threaded) |
| greedy 1.4.0 | `greedy_centers` | NMI, -dof 6 then -dof 12, -n 100x50x10, no mask | image centres (-ia-image-centers) | 1.19 / 0.93 / 0.90 | 0.91 | 63 s | CPU, 8 threads |
| greedy 1.4.0 | `greedy_centers_fov` | as greedy_centers, with the OCT FOV as moving mask | image centres | 1.00 / 1.00 / 1.00 | 1.00 | 73 s | CPU, 8 threads |
| greedy 1.4.0 | `greedy_identity` | as greedy_centers | header alignment (-ia-identity) | 1.28 / 1.97 / 1.02 | 1.83 | 55 s | CPU, 8 threads |
| greedy 1.4.0 | `greedy_search` | as greedy_centers | image centres + -search 1000 random rotations | 1.00 / 1.00 / 1.00 | 1.00 | 53 s | CPU, 8 threads |
| FreeSurfer 8.2.0 mri_coreg | `mri_coreg_default` | NMI, --dof 6 then --dof 12 --init-reg, tool defaults | volume centres + the default brute-force search | 0.92 / 0.81 / 1.00 | 0.74 | 22 min | CPU, 20 threads |
| FreeSurfer 8.2.0 mri_coreg | `mri_coreg_default_movmask` | as mri_coreg_default, with --mov-mask (OCT FOV) | volume centres + brute-force search | 0.92 / 0.81 / 1.00 | 0.74 | 18 min | CPU, 20 threads |
| FreeSurfer 8.2.0 mri_coreg | `mri_coreg_regheader` | as mri_coreg_default | header alignment (--regheader) | 1.05 / 1.06 / 1.02 | 1.14 | 19 min | CPU, 20 threads |
| FreeSurfer 8.2.0 mri_robust_register | `mri_robust_register_default_com` | --cost NMI, rigid then --affine --ixform, tool defaults | centres of mass (default) | 0.99 / 1.00 / 1.07 | 1.06 | 14 min | CPU, 20 threads |
| FreeSurfer 8.2.0 mri_robust_register | `mri_robust_register_default_com_maskmov` | as mri_robust_register_default_com, with --maskmov (OCT FOV) | centres of mass | 0.99 / 1.00 / 1.07 | 1.06 | 18 min | CPU, 20 threads |
| FreeSurfer 8.2.0 mri_robust_register | `mri_robust_register_noinit_rigid_only` | --cost NMI, the rigid step only: the affine step found no overlap of the two images and made no iteration | header alignment (--noinit) | 1.00 / 1.00 / 1.00 | 1.00 | 5 min | CPU, 20 threads |
| NiftyReg 2.1.2 reg_aladin | `niftyreg_center` | defaults: symmetric block matching, rigid then affine | image centres (default) | 0.97 / 0.96 / 0.98 | 0.91 | 5 min | CPU, 8 threads |
| NiftyReg 2.1.2 reg_aladin | `niftyreg_center_fov` | as niftyreg_center with -fmask (OCT FOV). This version ignores the mask, and the output equals niftyreg_center | image centres | 0.97 / 0.96 / 0.98 | 0.91 | 4 min | CPU, 8 threads |
| NiftyReg 2.1.2 reg_aladin | `niftyreg_nac` | defaults | header origin (-nac) | 1.07 / 0.95 / 1.04 | 1.02 | 3 min | CPU, 8 threads |
| FreeSurfer 8.2.0 mri_synthmorph | `synthmorph_affine` | register -m affine as documented. A model of 1 mm whole-brain images, run on a 0.08 mm block | none (the network predicts the affine) | 0.82 / 0.62 / 1.20 | 0.49 | 38 s | CPU, 20 threads |

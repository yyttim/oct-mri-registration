# Baselines on the I58 brainstem pair

Standard registration tools run on the I58 pair. The fixed image is the MRI crop as given (0.08 mm). The moving image is the
20 µm OCT box-averaged to 0.08 mm isotropic, the voxel size of the MRI. The affine stage runs eight tools, each from its
standard starts (the header alignment, the image centres or centres of mass, and the tool's own search where it has one)
and, as separate runs, with a field-of-view (FOV) mask of the OCT. The deformable stage runs five tools, each started from
octreg's affine, the shared start of the deformable comparison of Klein et al. (NeuroImage 46:786, 2009). No setting is
tuned on this pair. Every transform is checked against the image the tool itself resampled, and every overlay below is the
20 µm OCT resampled through the tool's transform by octreg's resampler, so the panels differ only by registration. Versions,
settings and scripts are in [README.md](README.md) and [INSTALL.md](INSTALL.md).

The main figures and the main rows of the tables show each tool at the setting its documentation gives for this kind of
registration, except that FLIRT's main run has both headers scaled x10 because its schedule works in mm (its run on the
unscaled headers is `flirt_search_native` in the supplement). This rule was fixed after the runs had been made, and it is
applied to every tool. Every other run is in the
supplement tables. I58 has no ground truth, so the result is the panels. The tables hold only properties that need no
reference: the scale per OCT axis and the determinant of an affine, the minimum of the Jacobian determinant, the share of
folded voxels and the SD of log J of a field, the setting, the start, the wall time and the hardware.

Each panel is one freeview screenshot of the whole MRI crop at one screen pixel per 0.08 mm voxel, on the sagittal, coronal and
axial planes through the voxel centre nearest the centroid of the MRI tissue, at (1.75, 0.38, 22.18) mm RAS. The red line is the
boundary of the MRI foreground that the rule of octreg's §1 gives on the 0.08 mm grid of the crop, computed from the MRI alone.
The OCT is shown at the default grey window freeview gives the moving image (0.00122 to 0.00553), and the MRI at 0.212 to 1.662,
both linear ([docs/figures/figures.json](../../docs/figures/figures.json)). Sagittal planes have anterior on the right, coronal
and axial planes show the subject's right on the left, and superior is up on sagittal and coronal planes, anterior up on axial
planes. The OCT sections show as fine stripes, horizontal on the sagittal and coronal planes. The bright lobulated tissue at the
right of the coronal MRI plane and the lower right of the axial one is cerebellar folia. `bench/figures.py` makes the panels.
The descriptions under each figure say what the panels show to the eye. They are not measurements.

## Affine

| method | run | setting | start | scale per OCT axis | det | wall time | hardware |
|---|---|---|---|---|---|---|---|
| image centres aligned (the start) | `reference_centres` | not a registration: the OCT translated so that the centres of the two image boxes coincide, header orientation kept | image centres aligned | 1.00 / 1.00 / 1.00 | 1.00 |  |  |
| octreg | `octreg` | §1 to §5 with default parameters | own orientation search (8000 orientations) inside the MRI crop | 0.99 / 0.96 / 0.98 | 0.94 | 10 min | GPU |
| NiftyReg 2.1.2 reg_aladin | `niftyreg_center` | defaults: symmetric block matching, rigid then affine | image centres (default) | 0.97 / 0.96 / 0.98 | 0.91 | 5 min | CPU, 8 threads |
| greedy 1.4.0 | `greedy_centers` | NMI, -dof 6 then -dof 12, -n 100x50x10, no mask | image centres (-ia-image-centers) | 1.19 / 0.93 / 0.90 | 0.91 | 63 s | CPU, 8 threads |
| elastix 5.3.1 | `elastix_center` | model-zoo Parameters_Rigid then Parameters_Affine (AdvancedMattesMutualInformation), no mask | geometric centres (GeometricalCenter) | 0.92 / 0.84 / 0.97 | 0.74 | 26 s | CPU, 12 threads |
| FreeSurfer 8.2.0 mri_robust_register | `mri_robust_register_default_com` | --cost NMI, rigid then --affine --ixform, tool defaults | centres of mass (default) | 0.99 / 1.00 / 1.07 | 1.06 | 14 min | CPU, 20 threads |
| ANTs 2.6.5 | `ants_com` | antsRegistrationSyN.sh -t a: rigid then affine, Mattes MI, the script's large-image schedule, no mask | centres of mass (script default) | 0.86 / 0.62 / 0.92 | 0.49 | 45 s | CPU, 12 threads |
| FreeSurfer 8.2.0 mri_coreg | `mri_coreg_default` | NMI, --dof 6 then --dof 12 --init-reg, tool defaults | volume centres + the default brute-force search | 0.92 / 0.81 / 1.00 | 0.74 | 22 min | CPU, 20 threads |
| FSL FLIRT 6.0.7.23 | `flirt_search_x10` | normmi, -dof 6 then -dof 12 -nosearch, both headers scaled x10 because FLIRT's schedule works in mm (an adaptation FSL does not document) | FLIRT default rotation search (±90°, coarse 60°, fine 18°) | 0.76 / 0.76 / 0.67 | 0.39 | 53 min | CPU (FLIRT runs single-threaded) |
| FreeSurfer 8.2.0 mri_synthmorph | `synthmorph_affine` | register -m affine as documented. A 1 mm whole-brain model, outside its domain on a 0.08 mm block | none (the network predicts the affine) | 0.82 / 0.62 / 1.20 | 0.49 | 38 s | CPU, 20 threads |

| | sagittal | coronal | axial |
|---|---|---|---|
| **MRI** | <img src="../../docs/figures/affine/mri_sagittal.png" width="240" alt="MRI, sagittal"> | <img src="../../docs/figures/affine/mri_coronal.png" width="168" alt="MRI, coronal"> | <img src="../../docs/figures/affine/mri_axial.png" width="168" alt="MRI, axial"> |
| **image centres aligned (start)** | <img src="../../docs/figures/affine/start_sagittal.png" width="240" alt="image centres aligned (start), sagittal"> | <img src="../../docs/figures/affine/start_coronal.png" width="168" alt="image centres aligned (start), coronal"> | <img src="../../docs/figures/affine/start_axial.png" width="168" alt="image centres aligned (start), axial"> |
| **octreg** | <img src="../../docs/figures/affine/octreg_sagittal.png" width="240" alt="octreg, sagittal"> | <img src="../../docs/figures/affine/octreg_coronal.png" width="168" alt="octreg, coronal"> | <img src="../../docs/figures/affine/octreg_axial.png" width="168" alt="octreg, axial"> |
| **NiftyReg reg_aladin** | <img src="../../docs/figures/affine/reg_aladin_sagittal.png" width="240" alt="NiftyReg reg_aladin, sagittal"> | <img src="../../docs/figures/affine/reg_aladin_coronal.png" width="168" alt="NiftyReg reg_aladin, coronal"> | <img src="../../docs/figures/affine/reg_aladin_axial.png" width="168" alt="NiftyReg reg_aladin, axial"> |
| **greedy** | <img src="../../docs/figures/affine/greedy_sagittal.png" width="240" alt="greedy, sagittal"> | <img src="../../docs/figures/affine/greedy_coronal.png" width="168" alt="greedy, coronal"> | <img src="../../docs/figures/affine/greedy_axial.png" width="168" alt="greedy, axial"> |
| **elastix** | <img src="../../docs/figures/affine/elastix_sagittal.png" width="240" alt="elastix, sagittal"> | <img src="../../docs/figures/affine/elastix_coronal.png" width="168" alt="elastix, coronal"> | <img src="../../docs/figures/affine/elastix_axial.png" width="168" alt="elastix, axial"> |
| **mri_robust_register** | <img src="../../docs/figures/affine/mri_robust_register_sagittal.png" width="240" alt="mri_robust_register, sagittal"> | <img src="../../docs/figures/affine/mri_robust_register_coronal.png" width="168" alt="mri_robust_register, coronal"> | <img src="../../docs/figures/affine/mri_robust_register_axial.png" width="168" alt="mri_robust_register, axial"> |
| **ANTs** | <img src="../../docs/figures/affine/ants_sagittal.png" width="240" alt="ANTs, sagittal"> | <img src="../../docs/figures/affine/ants_coronal.png" width="168" alt="ANTs, coronal"> | <img src="../../docs/figures/affine/ants_axial.png" width="168" alt="ANTs, axial"> |
| **mri_coreg** | <img src="../../docs/figures/affine/mri_coreg_sagittal.png" width="240" alt="mri_coreg, sagittal"> | <img src="../../docs/figures/affine/mri_coreg_coronal.png" width="168" alt="mri_coreg, coronal"> | <img src="../../docs/figures/affine/mri_coreg_axial.png" width="168" alt="mri_coreg, axial"> |
| **FLIRT** | <img src="../../docs/figures/affine/flirt_sagittal.png" width="240" alt="FLIRT, sagittal"> | <img src="../../docs/figures/affine/flirt_coronal.png" width="168" alt="FLIRT, coronal"> | <img src="../../docs/figures/affine/flirt_axial.png" width="168" alt="FLIRT, axial"> |
| **SynthMorph** | <img src="../../docs/figures/affine/synthmorph_sagittal.png" width="240" alt="SynthMorph, sagittal"> | <img src="../../docs/figures/affine/synthmorph_coronal.png" width="168" alt="SynthMorph, coronal"> | <img src="../../docs/figures/affine/synthmorph_axial.png" width="168" alt="SynthMorph, axial"> |

| method | what the panels show |
|---|---|
| image centres aligned (start) | The block keeps the header orientation, with its edges parallel to the image axes. The specimen crosses the line on all three planes. On the sagittal plane tissue lies outside the line at the anterior surface, and on the axial plane above the line at the top. The folia are offset from the folia lobe of the outline on the coronal and axial planes. |
| octreg | The block is turned relative to the image axes. The specimen surface runs along the line on all three planes. On the sagittal plane the bright rim near the top of the anterior surface lies just outside the line, and on the coronal plane the folia lie outside the line, above the folia lobe of the outline. The sections are straight. |
| NiftyReg reg_aladin | The block is turned relative to the image axes. The specimen surface runs along the line on all three planes. On the sagittal plane the bright rim near the top of the anterior surface lies just outside the line, and on the coronal plane the folia lie outside the line, above the folia lobe of the outline. The sections are straight. |
| greedy | The block is sheared: its edges are slanted on all three planes, and the sections run obliquely on the coronal and axial planes. On the sagittal plane the posterior surface lies inside the line and the lower part of the outline is empty. On the axial plane tissue lies beyond the line on the left, and the posterior part of the outline (the bottom) is empty. |
| elastix | The block keeps the header orientation and is smaller than the outline. The specimen lies inside the line, with a gap posteriorly in the lower half of the sagittal plane and at the top and the left of the axial plane. The lower left of the axial outline is empty. |
| mri_robust_register | The block keeps the header orientation and stays close to the start: on the sagittal plane tissue lies outside the line at the anterior surface, and on the axial plane above the line at the top. On the sagittal plane the block reaches further inferiorly than at the start. |
| ANTs | The block is shrunk, with its edges nearly parallel to the image axes. On the sagittal plane it is narrow from front to back and leaves the posterior part of the outline empty. On the axial plane it ends above the posterior part of the outline. On every plane the specimen fills only part of the outline. |
| mri_coreg | The block is compressed and slightly tilted. On the axial plane the folia sit near the middle of the outline and its posterior part (the bottom) is empty. On the sagittal plane tissue fragments lie outside the line posteriorly. |
| FLIRT | The block is shrunk and turned, and the sections run obliquely on the sagittal plane. The specimen outline does not follow the line on any plane: the folia appear in the right half of the sagittal outline and in the upper middle of the axial outline, and the lower part of the outline is empty on the sagittal and coronal planes. |
| SynthMorph | No OCT reaches the sagittal and coronal planes. Only one corner of the block appears, at the lower right of the axial plane, outside the line. |

The supplement holds every run whose transform passed the check against the tool's own resampling. reg_aladin's FOV-mask
variant writes the same transform as its run without the mask, since this version of reg_aladin ignores the mask. Three runs
have no row. elastix started from the header alignment stops before its first iteration with too many samples outside the
moving image and writes no transform. The affine step of mri_robust_register from the header alignment finds no overlap of
the two images, and its rigid step is the row `mri_robust_register_noinit_rigid_only`. mri_synthmorph run on headers scaled
by 10, a use its documentation does not give, returns a mirrored transform, which the run directory refuses.

| method | run | setting | start | scale per OCT axis | det | wall time | hardware |
|---|---|---|---|---|---|---|---|
| image centres aligned (the start) | `reference_centres` | not a registration: the OCT translated so that the centres of the two image boxes coincide, header orientation kept | image centres aligned | 1.00 / 1.00 / 1.00 | 1.00 |  |  |
| octreg | `octreg` | §1 to §5 with default parameters | own orientation search (8000 orientations) inside the MRI crop | 0.99 / 0.96 / 0.98 | 0.94 | 10 min | GPU |
| ANTs 2.6.5 | `ants_antsai` | antsAI rigid rotation search (Mattes MI, 0.15 mm copies), then the stages of ants_com | antsAI global rotation search | 1.03 / 1.00 / 1.03 | 1.07 | 4.9 h | CPU, 12 threads |
| ANTs 2.6.5 | `ants_com` | antsRegistrationSyN.sh -t a: rigid then affine, Mattes MI, the script's large-image schedule, no mask | centres of mass (script default) | 0.86 / 0.62 / 0.92 | 0.49 | 45 s | CPU, 12 threads |
| ANTs 2.6.5 | `ants_com_fov` | as ants_com, with the OCT field-of-view (FOV) mask | centres of mass | 1.28 / 1.01 / 1.10 | 1.39 | 68 s | CPU, 12 threads |
| ANTs 2.6.5 | `ants_identity` | as ants_com | header alignment (identity) | 0.91 / 1.03 / 0.99 | 0.93 | 25 s | CPU, 12 threads |
| elastix 5.3.1 | `elastix_center` | model-zoo Parameters_Rigid then Parameters_Affine (AdvancedMattesMutualInformation), no mask | geometric centres (GeometricalCenter) | 0.92 / 0.84 / 0.97 | 0.74 | 26 s | CPU, 12 threads |
| elastix 5.3.1 | `elastix_cog` | as elastix_center | centres of gravity (CenterOfGravity) | 0.90 / 0.82 / 0.97 | 0.71 | 32 s | CPU, 12 threads |
| elastix 5.3.1 | `elastix_cog_fov` | as elastix_center, with the OCT FOV mask | centres of gravity | 1.03 / 0.99 / 1.01 | 1.03 | 29 s | CPU, 12 threads |
| FSL FLIRT 6.0.7.23 | `flirt_fullsearch_x10` | as flirt_search_x10 | rotation search over ±180° | 0.73 / 0.73 / 0.73 | 0.39 | 2.4 h | CPU (FLIRT runs single-threaded) |
| FSL FLIRT 6.0.7.23 | `flirt_search_native` | normmi, -dof 6 then -dof 12, as documented, on the unscaled 0.08 mm headers | FLIRT default rotation search | 0.76 / 0.78 / 0.61 | 0.36 | 63 min | CPU (FLIRT runs single-threaded) |
| FSL FLIRT 6.0.7.23 | `flirt_search_x10` | normmi, -dof 6 then -dof 12 -nosearch, both headers scaled x10 because FLIRT's schedule works in mm (an adaptation FSL does not document) | FLIRT default rotation search (±90°, coarse 60°, fine 18°) | 0.76 / 0.76 / 0.67 | 0.39 | 53 min | CPU (FLIRT runs single-threaded) |
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
| FreeSurfer 8.2.0 mri_robust_register | `mri_robust_register_default_com_maskmov` | as default_com, with --maskmov (OCT FOV) | centres of mass | 0.99 / 1.00 / 1.07 | 1.06 | 18 min | CPU, 20 threads |
| FreeSurfer 8.2.0 mri_robust_register | `mri_robust_register_noinit_rigid_only` | --cost NMI, the rigid step only: the affine step found no overlap of the two images and made no iteration | header alignment (--noinit) | 1.00 / 1.00 / 1.00 | 1.00 | 5 min | CPU, 20 threads |
| NiftyReg 2.1.2 reg_aladin | `niftyreg_center` | defaults: symmetric block matching, rigid then affine | image centres (default) | 0.97 / 0.96 / 0.98 | 0.91 | 5 min | CPU, 8 threads |
| NiftyReg 2.1.2 reg_aladin | `niftyreg_center_fov` | as niftyreg_center with -fmask (OCT FOV). This version ignores the mask, and the output equals niftyreg_center | image centres | 0.97 / 0.96 / 0.98 | 0.91 | 4 min | CPU, 8 threads |
| NiftyReg 2.1.2 reg_aladin | `niftyreg_nac` | defaults | header origin (-nac) | 1.07 / 0.95 / 1.04 | 1.02 | 3 min | CPU, 8 threads |
| FreeSurfer 8.2.0 mri_synthmorph | `synthmorph_affine` | register -m affine as documented. A 1 mm whole-brain model, outside its domain on a 0.08 mm block | none (the network predicts the affine) | 0.82 / 0.62 / 1.20 | 0.49 | 38 s | CPU, 20 threads |

## Deformable

Every tool starts from octreg's affine: the OCT is resampled through it onto the 0.08 mm MRI grid, and the tool starts from the
identity on that grid. The field is the only difference between the panels. Its Jacobian determinant J is computed by finite
differences on the field's own grid, over the MRI foreground of octreg's §1 (a mask of the MRI alone) sampled onto that grid.
Folded voxels are those with J <= 0, and the SD of log J, with J clipped at 1e-6 where it is not positive, is the smoothness
measure of Learn2Reg (Hering et al., IEEE Trans. Med. Imaging 42:697, 2023). The size of the displacements is not reported:
every field starts from octreg's affine, so it would be a distance from octreg's result.

| method | run | setting | start | field grid | Jacobian min | folded (J <= 0) | SD log J | wall time | hardware |
|---|---|---|---|---|---|---|---|---|---|
| no deformation | `none` | octreg's affine, no field | octreg's affine |  |  |  |  |  |  |
| octreg | `octreg` | §6: strain-limited field on a 5 mm lattice, default parameters | octreg's affine | 0.15 mm | 0.94 | 0.00 % | 0.07 | 21 s | GPU |
| ANTs 2.6.5 | `deform_ants_syn_cc` | antsRegistrationSyN.sh -t so: SyN, CC radius 2, script defaults, no mask | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | 0.01 | 0.00 % | 0.46 | 42 min | CPU, 8 threads |
| elastix 5.3.1 | `deform_elastix_default` | model-zoo Parameters_BSpline: Mattes MI, 16 mm final grid, no mask | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | 1.02 | 0.00 % | 0.02 | 21 s | CPU, 8 threads |
| NiftyReg 2.1.2 reg_f3d | `deform_niftyreg_default` | defaults: NMI 64 bins, control grid 5 voxels (0.4 mm), bending energy 0.001, linear elasticity 0.01, no mask | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | -88.30 | 47.80 % | 6.33 | 3 min | CPU, 8 threads |
| ConvexAdam 0.2.0 | `deform_convexadam_default` | MIND-SSC, the package's shipped defaults, no mask | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | -0.86 | 0.46 % | 1.10 | 8 s | GPU |
| greedy 1.4.0 | `deform_greedy_wncc` | the quick-start deformable command: WNCC 2x2x2, -s 1.732vox 0.707vox, -sv, -n 100x50x10, no mask | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | 0.10 | 0.00 % | 0.17 | 4 min | CPU, 8 threads |

| | sagittal | coronal | axial |
|---|---|---|---|
| **MRI** | <img src="../../docs/figures/deformable/mri_sagittal.png" width="240" alt="MRI, sagittal"> | <img src="../../docs/figures/deformable/mri_coronal.png" width="168" alt="MRI, coronal"> | <img src="../../docs/figures/deformable/mri_axial.png" width="168" alt="MRI, axial"> |
| **no deformation (octreg's affine)** | <img src="../../docs/figures/deformable/affine_sagittal.png" width="240" alt="no deformation (octreg's affine), sagittal"> | <img src="../../docs/figures/deformable/affine_coronal.png" width="168" alt="no deformation (octreg's affine), coronal"> | <img src="../../docs/figures/deformable/affine_axial.png" width="168" alt="no deformation (octreg's affine), axial"> |
| **octreg, §6** | <img src="../../docs/figures/deformable/octreg_sagittal.png" width="240" alt="octreg, §6, sagittal"> | <img src="../../docs/figures/deformable/octreg_coronal.png" width="168" alt="octreg, §6, coronal"> | <img src="../../docs/figures/deformable/octreg_axial.png" width="168" alt="octreg, §6, axial"> |
| **ANTs SyN** | <img src="../../docs/figures/deformable/ants_syn_sagittal.png" width="240" alt="ANTs SyN, sagittal"> | <img src="../../docs/figures/deformable/ants_syn_coronal.png" width="168" alt="ANTs SyN, coronal"> | <img src="../../docs/figures/deformable/ants_syn_axial.png" width="168" alt="ANTs SyN, axial"> |
| **elastix B-spline** | <img src="../../docs/figures/deformable/elastix_sagittal.png" width="240" alt="elastix B-spline, sagittal"> | <img src="../../docs/figures/deformable/elastix_coronal.png" width="168" alt="elastix B-spline, coronal"> | <img src="../../docs/figures/deformable/elastix_axial.png" width="168" alt="elastix B-spline, axial"> |
| **NiftyReg reg_f3d** | <img src="../../docs/figures/deformable/reg_f3d_sagittal.png" width="240" alt="NiftyReg reg_f3d, sagittal"> | <img src="../../docs/figures/deformable/reg_f3d_coronal.png" width="168" alt="NiftyReg reg_f3d, coronal"> | <img src="../../docs/figures/deformable/reg_f3d_axial.png" width="168" alt="NiftyReg reg_f3d, axial"> |
| **ConvexAdam** | <img src="../../docs/figures/deformable/convexadam_sagittal.png" width="240" alt="ConvexAdam, sagittal"> | <img src="../../docs/figures/deformable/convexadam_coronal.png" width="168" alt="ConvexAdam, coronal"> | <img src="../../docs/figures/deformable/convexadam_axial.png" width="168" alt="ConvexAdam, axial"> |
| **greedy** | <img src="../../docs/figures/deformable/greedy_sagittal.png" width="240" alt="greedy, sagittal"> | <img src="../../docs/figures/deformable/greedy_coronal.png" width="168" alt="greedy, coronal"> | <img src="../../docs/figures/deformable/greedy_axial.png" width="168" alt="greedy, axial"> |

| method | what the panels show |
|---|---|
| no deformation (octreg's affine) | The shared start. The sections are straight and the specimen surface runs along the line over most of its length. On the sagittal plane the bright rim near the top of the anterior surface lies just outside the line, and on the coronal plane the folia lie outside the line, above the folia lobe of the outline. The flat superior cut face of the block lies below the top of the MRI tissue on the coronal plane. |
| octreg, §6 | The sections stay straight and the block is only gently bent. On the sagittal plane the bright rim near the top of the anterior surface lies on the line. The superior cut face stays flat. |
| ANTs SyN | The flat superior cut face is pulled up into the MRI tissue above it: the OCT fills the outline up to its top on the coronal plane and the upper left of the outline on the sagittal plane, with stretched texture there. The sections are wavy throughout and the block edges are distorted. The specimen surface lies on the line nearly everywhere. |
| elastix B-spline | The whole block is bent gently and moved inferiorly: on the sagittal plane the dark gap of the inferior notch lies below the notch of the outline. The sections stay straight. On the sagittal plane the bright rim near the top of the anterior surface lies on the line. |
| NiftyReg reg_f3d | The OCT cannot be recognised. Inside the outline its texture is replaced by smooth swirl and cell-like patterns, and bright smeared material lies outside the line on every plane. |
| ConvexAdam | The block outline is strongly warped: the corners bulge outward and the edges are wavy. The agarose around the specimen is smeared into large smooth patches, at the lower corners and the upper right of the sagittal plane and at the bottom of the axial plane. The superior cut face is lifted to the top of the outline on the coronal plane, and the OCT fills the upper left of the outline on the sagittal plane. The specimen surface lies on the line, and inside the tissue the sections show local waves. |
| greedy | The sections stay mostly straight, with mild waviness near the block edges. The specimen surface lies on the line on all three planes, including the bright rim near the top of the anterior surface on the sagittal plane. The superior cut face is raised slightly towards the sloping top of the outline on the coronal plane. |

The supplement holds every run:

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

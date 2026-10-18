# Baselines on the I58 brainstem pair

Standard registration tools run on the I58 pair. The fixed image is the MRI crop as given (0.08 mm). The moving image is the 20
µm OCT box-averaged to 0.08 mm isotropic, the voxel size of the MRI. The affine stage runs eight tools. Seven of them run from
their standard starts (the header alignment, the image centres or centres of mass, and the tool's own search where it has one)
and, as separate runs, with a field-of-view (FOV) mask of the OCT. SynthMorph, which predicts the affine and has no mask option,
runs once. The deformable stage runs five tools, each started from octreg's affine. Versions, settings and scripts are in
[README.md](README.md) and [INSTALL.md](INSTALL.md).

The figures and tables below show each tool at the setting its documentation gives for this kind of registration, with two
exceptions: FLIRT's two headers are scaled x10, because its schedule resamples the images to fixed voxel sizes in mm, and ANTs
SyN runs in single precision. The other starts and settings of the affine stage are in the stored table
[affine.md](../results/I58/baselines/affine.md), and the deformable table is also stored as
[deform.md](../results/I58/baselines/deform.md). I58 has no reference, so the tools are shown, not scored. The tables give, for
an affine, the scale per OCT axis (the length of a unit step along each OCT array axis after the transform, 1 = voxel size kept)
and the determinant, for a field the minimum of its Jacobian determinant J, the share of folded voxels (J <= 0) and the SD of
log J, and for each run the setting, start, wall time and hardware.

Each panel is one freeview screenshot of the whole MRI crop at one screen pixel per 0.08 mm voxel, showing the MRI or the 20 µm
OCT resampled through the tool's transform by octreg's resampler. The planes are the sagittal, coronal and axial planes through
the voxel centre nearest the centroid of the MRI tissue, at (1.75, 0.30, 22.18) mm RAS. The red line is the MRI tissue boundary,
the boundary of the MRI foreground that the rule of octreg's [§1](../../docs/METHOD.md#1-masks) gives on the 0.08 mm grid of the
crop. On the sagittal and coronal planes it also encloses MRI tissue above and below the OCT specimen. On the axial plane it
also encloses MRI tissue along the right side of the image, where the OCT at octreg's affine shows a dark band with a few bright
fragments. The captions call this the right axial band. The OCT is shown at the default grey window freeview gives the moving
image (0.00122 to 0.00553), and the MRI at 0.212 to 1.662, both linear
([docs/figures/figures.json](../../docs/figures/figures.json)). Sagittal planes have anterior on the right, coronal and axial
planes show the subject's right on the left, and superior is up on sagittal and coronal planes, anterior up on axial planes.
Left, right, top and bottom in the text and the captions refer to the image. On the sagittal and coronal planes the OCT sections
show as fine stripes parallel to the top and bottom edges of the block, horizontal at the start. `bench/figures.py` makes the
panels.

## Affine

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

| | sagittal | coronal | axial |
|---|---|---|---|
| **MRI** | <img src="../../docs/figures/affine/mri_sagittal.png" width="240" alt="MRI, sagittal"> | <img src="../../docs/figures/affine/mri_coronal.png" width="168" alt="MRI, coronal"> | <img src="../../docs/figures/affine/mri_axial.png" width="168" alt="MRI, axial"> |
| **image centres aligned (the start)** | <img src="../../docs/figures/affine/start_sagittal.png" width="240" alt="image centres aligned (the start), sagittal"> | <img src="../../docs/figures/affine/start_coronal.png" width="168" alt="image centres aligned (the start), coronal"> | <img src="../../docs/figures/affine/start_axial.png" width="168" alt="image centres aligned (the start), axial"> |
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
| image centres aligned (the start) | The block keeps the header orientation, with its edges parallel to the image axes. Tissue lies outside the line along the right side and at the left of the sagittal plane, along the left side and at the upper right of the coronal plane, and along the left side, at the top and at the lower left of the axial plane. |
| octreg | The block is turned relative to the image axes and the sections are straight. The specimen surface runs along the line except at these places. On the sagittal plane the bright rim at the top, the rounded tissue edge at the upper left and a bright band just below the step in the line at the lower left lie outside the line. On the coronal plane the striped upper right corner of the specimen and a separate piece of tissue below it lie partly outside the line, and below that piece the specimen ends short of the line. On the axial plane the specimen ends short of the line at the right axial band. |
| NiftyReg reg_aladin | As for octreg: the block is turned, the sections are straight, and the specimen surface runs along the line except at the same places. |
| greedy | The block is sheared: its edges are slanted, the sections run obliquely on the coronal plane and show as nearly vertical stripes on the axial plane. Parts of the area inside the line hold no tissue on all three planes, most at the lower left of the coronal plane, and on the axial plane tissue lies outside the line along the left side. |
| elastix | The block's edges stay nearly parallel to the image axes. The specimen is smaller than the area inside the line and lies mostly inside the line with gaps, and pieces of tissue cross the line at the upper right of the coronal plane, at the left of the sagittal plane and at the lower left of the axial plane. |
| mri_robust_register | The block keeps the header orientation and stays close to the start. As at the start, tissue lies outside the line on all three planes. |
| ANTs | The block is shrunk, with its edges nearly parallel to the image axes. On every plane the specimen fills only part of the area inside the line. |
| mri_coreg | The block is slightly tilted on the sagittal and coronal planes. The specimen ends short of the posterior end of the line on the axial plane, and the upper left of the area inside the line on the sagittal plane holds no tissue. |
| FLIRT | The block is shrunk and turned. The lower part of the area inside the line on the sagittal and coronal planes holds no tissue, and on the axial plane the specimen is narrower than the area inside the line and reaches past its posterior end. |
| SynthMorph | No OCT reaches the sagittal and coronal planes. One corner of the block appears at the lower right of the axial plane, outside the line. |

## Deformable

As in the comparison of Klein et al. (NeuroImage 46:786, 2009), every tool starts from one shared affine, here octreg's: the OCT
is resampled through it onto the 0.08 mm MRI grid, and the tool starts from the identity on that grid. The Jacobian determinant
J of the field is computed by finite differences on the field's own grid, over the MRI foreground of octreg's §1, computed on
its 0.15 mm base grid and sampled onto the field's grid. Folded voxels are those with J <= 0, and the SD of log J, with J
clipped below at 1e-6, is the smoothness measure of Learn2Reg (Hering et al., IEEE Trans. Med. Imaging 42:697, 2023).

| method | run | setting | start | field grid | Jacobian min | folded (J <= 0) | SD log J | wall time | hardware |
|---|---|---|---|---|---|---|---|---|---|
| no deformation | `none` | octreg's affine, no field | octreg's affine |  |  |  |  |  |  |
| octreg | `octreg` | §5: displacement field on a 5 mm lattice under a strain bound, default parameters | octreg's affine | 0.15 mm | 0.84 | 0.00 % | 0.06 | 58 s | GPU |
| ANTs 2.6.5 | `deform_ants_syn_cc` | antsRegistrationSyN.sh -t so: SyN, CC radius 2, script defaults in single precision (-p f), no mask | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | -0.05 | < 0.01 % | 0.47 | 44 min | CPU, 8 threads |
| elastix 5.3.1 | `deform_elastix_default` | model-zoo Parameters_BSpline: Mattes MI, 16 mm final grid, no mask | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | 0.99 | 0.00 % | 0.02 | 24 s | CPU, 8 threads |
| NiftyReg 2.1.2 reg_f3d | `deform_niftyreg_default` | defaults: NMI 64 bins, control grid 5 voxels (0.4 mm), bending energy 0.001, linear elasticity 0.01, no mask | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | -67.00 | 47.23 % | 6.35 | 3 min | CPU, 8 threads |
| ConvexAdam 0.2.0 | `deform_convexadam_default` | MIND-SSC, the package's shipped defaults, no mask | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | -0.80 | 0.49 % | 1.12 | 7 s | GPU |
| greedy 1.4.0 | `deform_greedy_wncc` | the quick-start deformable command: WNCC 2x2x2, -s 1.732vox 0.707vox, -sv, -n 100x50x10, no mask | octreg's affine (both images on the MRI grid, identity start) | 0.08 mm | 0.12 | 0.00 % | 0.17 | 3 min | CPU, 8 threads |

| | sagittal | coronal | axial |
|---|---|---|---|
| **MRI** | <img src="../../docs/figures/deformable/mri_sagittal.png" width="240" alt="MRI, sagittal"> | <img src="../../docs/figures/deformable/mri_coronal.png" width="168" alt="MRI, coronal"> | <img src="../../docs/figures/deformable/mri_axial.png" width="168" alt="MRI, axial"> |
| **no deformation (octreg's affine)** | <img src="../../docs/figures/deformable/affine_sagittal.png" width="240" alt="no deformation (octreg's affine), sagittal"> | <img src="../../docs/figures/deformable/affine_coronal.png" width="168" alt="no deformation (octreg's affine), coronal"> | <img src="../../docs/figures/deformable/affine_axial.png" width="168" alt="no deformation (octreg's affine), axial"> |
| **octreg, §5** | <img src="../../docs/figures/deformable/octreg_sagittal.png" width="240" alt="octreg, §5, sagittal"> | <img src="../../docs/figures/deformable/octreg_coronal.png" width="168" alt="octreg, §5, coronal"> | <img src="../../docs/figures/deformable/octreg_axial.png" width="168" alt="octreg, §5, axial"> |
| **ANTs SyN** | <img src="../../docs/figures/deformable/ants_syn_sagittal.png" width="240" alt="ANTs SyN, sagittal"> | <img src="../../docs/figures/deformable/ants_syn_coronal.png" width="168" alt="ANTs SyN, coronal"> | <img src="../../docs/figures/deformable/ants_syn_axial.png" width="168" alt="ANTs SyN, axial"> |
| **elastix B-spline** | <img src="../../docs/figures/deformable/elastix_sagittal.png" width="240" alt="elastix B-spline, sagittal"> | <img src="../../docs/figures/deformable/elastix_coronal.png" width="168" alt="elastix B-spline, coronal"> | <img src="../../docs/figures/deformable/elastix_axial.png" width="168" alt="elastix B-spline, axial"> |
| **NiftyReg reg_f3d** | <img src="../../docs/figures/deformable/reg_f3d_sagittal.png" width="240" alt="NiftyReg reg_f3d, sagittal"> | <img src="../../docs/figures/deformable/reg_f3d_coronal.png" width="168" alt="NiftyReg reg_f3d, coronal"> | <img src="../../docs/figures/deformable/reg_f3d_axial.png" width="168" alt="NiftyReg reg_f3d, axial"> |
| **ConvexAdam** | <img src="../../docs/figures/deformable/convexadam_sagittal.png" width="240" alt="ConvexAdam, sagittal"> | <img src="../../docs/figures/deformable/convexadam_coronal.png" width="168" alt="ConvexAdam, coronal"> | <img src="../../docs/figures/deformable/convexadam_axial.png" width="168" alt="ConvexAdam, axial"> |
| **greedy** | <img src="../../docs/figures/deformable/greedy_sagittal.png" width="240" alt="greedy, sagittal"> | <img src="../../docs/figures/deformable/greedy_coronal.png" width="168" alt="greedy, coronal"> | <img src="../../docs/figures/deformable/greedy_axial.png" width="168" alt="greedy, axial"> |

| method | what the panels show |
|---|---|
| no deformation (octreg's affine) | The shared start. The sections are straight. On the sagittal plane the bright rim at the top, the rounded tissue edge at the upper left and a bright band just below the step in the line at the lower left lie outside the line. On the coronal plane the flat superior cut face lies below the top of the MRI tissue, the striped upper right corner of the specimen and a separate piece of tissue below it lie partly outside the line, and below that piece the specimen ends short of the line. On the axial plane the specimen ends short of the line at the right axial band. |
| octreg, §5 | The sections stay straight. On the sagittal plane the bright rim at the top lies just outside the line, the rounded tissue edge at the upper left and the bright band below the step at the lower left still lie outside it, the rest of the surface at the left lies on the line, and along the lower right the surface lies inside the line with a dark gap. On the coronal plane the superior cut face stays flat, the striped upper right corner and the separate piece below it still lie partly outside the line, and below that piece the specimen still ends short of it. On the axial plane the specimen still ends short of the line at the right axial band. |
| ANTs SyN | The flat superior cut face is pulled up into the MRI tissue above it on the sagittal and coronal planes, with stretched texture there. The sections are wavy and the block edges distorted. The specimen surface lies on the line except at the right axial band and at the upper right of the coronal plane, where the separate piece lies partly outside it and the specimen below that piece ends short of it. |
| elastix B-spline | The block is moved inferiorly on the sagittal and coronal planes, and the sections stay straight. On both planes the specimen lies inside the line along parts of the left and right sides, with gaps. On the sagittal plane the bright rim at the top lies on the line, the rounded tissue edge at the upper left lies just outside it and the inferior edge of the specimen lies below it. On the coronal plane the striped upper right corner and the separate piece below it lie partly outside the line. On the axial plane the specimen lies inside the line with gaps on all sides, a piece of folded tissue fills the lower part of the right axial band, and the upper part of the separate piece at the lower left lies outside the line. |
| NiftyReg reg_f3d | The OCT cannot be recognised: inside the line its texture is replaced by smooth swirl and cell-like patterns, and bright smeared material lies outside the line on every plane. |
| ConvexAdam | The block edges are strongly warped and the agarose is smeared into large smooth patches. The superior cut face is lifted to the top of the line on the coronal plane, and on the sagittal plane the top of the block is lifted part of the way into the MRI tissue at the upper left and rises above the line near the top centre. The specimen surface lies on the line except at the right axial band, at the separate piece at the lower left of the axial plane, which lies partly outside it, and at the upper right of the coronal plane, where the striped corner and the separate piece below it lie partly outside the line and the specimen below them ends short of it. The sections show local waves. |
| greedy | The sections stay mostly straight, with mild waviness near the block edges. On the coronal plane the superior cut face is pulled up into the MRI tissue above it. The specimen surface, including the bright rim, lies on the line except at the right axial band, at the rounded tissue edge at the upper left of the sagittal plane, which lies partly outside it, and at the upper right of the coronal plane, where the striped corner and the separate piece below it lie partly outside the line and the specimen below them ends short of it. |

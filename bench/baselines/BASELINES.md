# Baselines on the I58 brainstem pair

[README.md](README.md) gives the inputs, settings and starts, and the stored table
[affine.md](../results/I58/baselines/affine.md) the scale per OCT axis and the determinant of each run's affine.

Each panel is one freeview screenshot of the whole MRI crop at one screen pixel per 0.08 mm voxel, showing the MRI or the 20 µm
OCT resampled through the tool's transform by octreg's resampler. The planes are the sagittal, coronal and axial planes through
the voxel centre nearest the centroid of the MRI tissue, at (1.75, 0.38, 22.26) mm RAS. The red line is the boundary of the MRI
foreground octreg uses ([§1](../../docs/METHOD.md#1-masks)) resampled from its 0.15 mm base grid to the crop. On the sagittal
and coronal planes it also encloses MRI tissue above and below the block. On the axial plane it also encloses MRI tissue along
the right side of the image, where the OCT at octreg's affine shows a dark band with a few bright fragments. The captions call
this the right axial band. The OCT is shown at the default grey window freeview gives the moving image (0.00122 to 0.00553), and
the MRI at 0.212 to 1.662, both linear ([docs/figures/figures.json](../../docs/figures/figures.json)). Sagittal planes have
anterior on the right, coronal and axial planes show the subject's right on the left, and superior is up on sagittal and coronal
planes, anterior up on axial planes. Left, right, top and bottom in the text and the captions refer to the image. On the
sagittal and coronal planes the OCT sections show as fine stripes parallel to the top and bottom edges of the block, horizontal
at the start. `bench/figures.py` makes the panels.

| | sagittal | coronal | axial |
|---|---|---|---|
| **MRI** | <img src="../../docs/figures/affine/mri_sagittal.png" width="240" alt="MRI, sagittal"> | <img src="../../docs/figures/affine/mri_coronal.png" width="166" alt="MRI, coronal"> | <img src="../../docs/figures/affine/mri_axial.png" width="166" alt="MRI, axial"> |
| **image box centres aligned (the start)** | <img src="../../docs/figures/affine/start_sagittal.png" width="240" alt="image box centres aligned (the start), sagittal"> | <img src="../../docs/figures/affine/start_coronal.png" width="166" alt="image box centres aligned (the start), coronal"> | <img src="../../docs/figures/affine/start_axial.png" width="166" alt="image box centres aligned (the start), axial"> |
| **octreg** | <img src="../../docs/figures/affine/octreg_sagittal.png" width="240" alt="octreg, sagittal"> | <img src="../../docs/figures/affine/octreg_coronal.png" width="166" alt="octreg, coronal"> | <img src="../../docs/figures/affine/octreg_axial.png" width="166" alt="octreg, axial"> |
| **NiftyReg reg_aladin** | <img src="../../docs/figures/affine/reg_aladin_sagittal.png" width="240" alt="NiftyReg reg_aladin, sagittal"> | <img src="../../docs/figures/affine/reg_aladin_coronal.png" width="166" alt="NiftyReg reg_aladin, coronal"> | <img src="../../docs/figures/affine/reg_aladin_axial.png" width="166" alt="NiftyReg reg_aladin, axial"> |
| **greedy** | <img src="../../docs/figures/affine/greedy_sagittal.png" width="240" alt="greedy, sagittal"> | <img src="../../docs/figures/affine/greedy_coronal.png" width="166" alt="greedy, coronal"> | <img src="../../docs/figures/affine/greedy_axial.png" width="166" alt="greedy, axial"> |
| **elastix** | <img src="../../docs/figures/affine/elastix_sagittal.png" width="240" alt="elastix, sagittal"> | <img src="../../docs/figures/affine/elastix_coronal.png" width="166" alt="elastix, coronal"> | <img src="../../docs/figures/affine/elastix_axial.png" width="166" alt="elastix, axial"> |
| **mri_robust_register** | <img src="../../docs/figures/affine/mri_robust_register_sagittal.png" width="240" alt="mri_robust_register, sagittal"> | <img src="../../docs/figures/affine/mri_robust_register_coronal.png" width="166" alt="mri_robust_register, coronal"> | <img src="../../docs/figures/affine/mri_robust_register_axial.png" width="166" alt="mri_robust_register, axial"> |
| **ANTs** | <img src="../../docs/figures/affine/ants_sagittal.png" width="240" alt="ANTs, sagittal"> | <img src="../../docs/figures/affine/ants_coronal.png" width="166" alt="ANTs, coronal"> | <img src="../../docs/figures/affine/ants_axial.png" width="166" alt="ANTs, axial"> |
| **mri_coreg** | <img src="../../docs/figures/affine/mri_coreg_sagittal.png" width="240" alt="mri_coreg, sagittal"> | <img src="../../docs/figures/affine/mri_coreg_coronal.png" width="166" alt="mri_coreg, coronal"> | <img src="../../docs/figures/affine/mri_coreg_axial.png" width="166" alt="mri_coreg, axial"> |
| **FLIRT** | <img src="../../docs/figures/affine/flirt_sagittal.png" width="240" alt="FLIRT, sagittal"> | <img src="../../docs/figures/affine/flirt_coronal.png" width="166" alt="FLIRT, coronal"> | <img src="../../docs/figures/affine/flirt_axial.png" width="166" alt="FLIRT, axial"> |
| **SynthMorph** | <img src="../../docs/figures/affine/synthmorph_sagittal.png" width="240" alt="SynthMorph, sagittal"> | <img src="../../docs/figures/affine/synthmorph_coronal.png" width="166" alt="SynthMorph, coronal"> | <img src="../../docs/figures/affine/synthmorph_axial.png" width="166" alt="SynthMorph, axial"> |

| method | what the panels show |
|---|---|
| image box centres aligned (the start) | The block keeps the header orientation, with its edges parallel to the image axes. Tissue lies outside the line along the right side and at the left of the sagittal plane, along the left side and at the upper right of the coronal plane, and along the left side, at the top and at the lower left of the axial plane. |
| octreg | The block is turned relative to the image axes and the sections are straight. The specimen surface runs along the line except at these places. On the sagittal plane the bright rim at the top, the rounded tissue edge at the upper left and a bright band just below the step in the line at the lower left lie outside the line, and on the right, below the rim, small loops of the line reach past the specimen edge. On the coronal plane the striped upper right corner of the specimen and a separate piece of tissue below it lie partly outside the line, below that piece the specimen ends short of the line, and at the top the line dips into the specimen. On the axial plane the specimen ends short of the line at the right axial band. |
| NiftyReg reg_aladin | As for octreg: the block is turned, the sections are straight, and the specimen surface runs along the line except at the same places. |
| greedy | The block is sheared: its edges are slanted, the sections run obliquely on the coronal plane and show as nearly vertical stripes on the axial plane. Parts of the area inside the line hold no tissue on all three planes, most at the lower left of the coronal plane, and on the axial plane tissue lies outside the line along the left side. |
| elastix | The block's edges stay nearly parallel to the image axes. The specimen is smaller than the area inside the line and lies mostly inside the line with gaps, and pieces of tissue cross the line at the upper right of the coronal plane, at the left of the sagittal plane and at the lower left of the axial plane. |
| mri_robust_register | The block keeps the header orientation and stays close to the start. As at the start, tissue lies outside the line on all three planes. |
| ANTs | The block is shrunk, with its edges nearly parallel to the image axes. On every plane the specimen fills only part of the area inside the line. |
| mri_coreg | The block is slightly tilted on the sagittal and coronal planes. The specimen ends short of the posterior end of the line on the axial plane, and the upper left of the area inside the line on the sagittal plane holds no tissue. |
| FLIRT | The block is shrunk and turned. The lower part of the area inside the line on the sagittal and coronal planes holds no tissue, and on the axial plane the specimen is narrower than the area inside the line and reaches past its posterior end. |
| SynthMorph | No OCT reaches the sagittal and coronal planes. One corner of the block appears at the lower right of the axial plane, outside the line. |

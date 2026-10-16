# octreg

octreg registers a serial-sectioning OCT volume of a tissue block to an ex-vivo MRI cropped around the block, with an affine
transform and a small smooth deformation on top. It uses no labels and needs no initial alignment, since it searches every
orientation of the block inside the MRI crop.

## How it works

```mermaid
flowchart LR
  OCT[OCT block] --> S1
  MRI[MRI crop] --> S1
  S1["§1 specimen mask"] --> S2["§2 two-class maps and outline"]
  S2 --> S3["§3 orientation search"]
  S3 --> S4["§4 affine refinement"]
  S4 --> S5["§5 fine-structure refinement"]
  S5 --> AFF[affine]
  S5 --> S6["§6 smooth deformation"]
  S6 --> FIELD[deformation field]
```

**§1 Specimen mask.** Doped agarose scatters about as strongly as tissue, but its section stripes and tile seams vary along one
array axis, while tissue texture varies along all three. The OCT specimen mask is the region of isotropic texture, and the MRI
foreground is a histogram threshold.

**§2 Two-class maps and outline.** Both scans become soft maps of a brighter and a darker tissue class, scored together with the
outline when both scans show what lies outside the tissue. The sign of the two-class score gives the contrast polarity.

**§3 Orientation search.** An FFT search scores 8,000 rotations with all translations inside the MRI crop and keeps 24 poses.

**§4 Affine refinement.** Each pose is refined to a 12-parameter affine under a prior on scale and shear.

**§5 Fine-structure refinement.** The best pose is refined on the correlation of the gradient orientation tensors of both scans.
The gradients follow fibre bundles and vessels, and their orientation needs no intensity mapping.

**§6 Smooth deformation.** A displacement field on a 5 mm control lattice, held to a strain limit of 0.15, is fitted to interior
matches and surface-edge offsets. It is applied only when it predicts held-out evidence at least 10 % better than no
deformation, and the affine stays the primary result.

[docs/METHOD.md](docs/METHOD.md) gives every step in full.

## Installation

```
git clone https://github.com/yyttim/oct-mri-registration
cd oct-mri-registration
pip install -e .
```

octreg requires Python 3.9 or newer and PyTorch. A CUDA GPU is recommended for full-size volumes. `pip install -e ".[test]"`
adds the test dependencies, and `pytest` runs the tests on synthetic data on the CPU.

## Usage

```
octreg register OCT MRI -o OUT [--oct-spacing-um Z,Y,X] [--oct-mask F] [--mri-mask F] [--params F] [--device cuda|cpu]
octreg qc --run OUT --oct OCT --mri MRI [--T F] [-o PREFIX] [--oct-spacing-um Z,Y,X] [--oct-mask F] [--mri-mask F]
octreg apply --run OUT --moving X --reference Y -o Z [--inverse] [--affine-only] [--oct-spacing-um Z,Y,X]
```

`register` runs §1 to §6 and writes the outputs below into `OUT`. The OCT can be NIfTI, TIFF, OME-TIFF or NPY, and the MRI is
NIfTI. `--oct-spacing-um` gives the OCT voxel spacing in µm along the numpy axes z, y, x, for the OCT and its mask file. A TIFF
or NPY without spacing needs it, and it overrides the spacing a file has. `--oct-mask` and `--mri-mask` replace the automatic
masks with files in the world frame of their scan, whose positive voxels are the mask. `--params` reads parameter overrides from
JSON (the fields of `octreg.params.Params`, listed in [docs/METHOD.md](docs/METHOD.md#parameters)), and `--device cpu` runs
without a GPU.

`qc` renders the QC images of the affine again, for the run's transform or for another one given with `--T`. `-o` sets the
output prefix (default `OUT/qc`, or `OUT/qc_<T stem>` for another transform), and `--oct-spacing-um`, `--oct-mask` and
`--mri-mask` default to those of the run.

`apply` resamples an OCT-frame volume onto the grid of an MRI-frame `--reference`, through the affine and, when the run wrote
one, the deformation field. `--affine-only` leaves the field out. With `--inverse` it resamples an MRI-frame volume onto an
OCT-frame reference through the affine alone, since the field is not inverted. `--oct-spacing-um` gives the spacing of an
OCT-frame TIFF or NPY. It defaults to that of the run for the run's own OCT file at its original path.

The same functions are available in Python: `from octreg.register import register, apply, qc`.

### Outputs

| file | content |
|---|---|
| `T_oct2mri.txt`, `T_mri2oct.txt` | 4×4 affine in mm between the world frames of the two input files |
| `oct2mri.lta`, `oct2mri_itk.txt` | the same affine for FreeSurfer and ITK |
| `oct2mri_warp.nii.gz` | the deformation of §6, only when it is applied: displacement u in mm along the MRI world axes on the MRI base grid (0.15 mm), as a NIfTI vector image (ITK and ANTs expect the first two components negated). The registered OCT at the MRI point x is OCT(T⁻¹(x + u(x))) |
| `oct_in_mri.nii.gz` | OCT on the MRI grid through the affine and the deformation |
| `oct_in_mri_affine.nii.gz` | OCT on the MRI grid through the affine alone (the same image when no deformation is applied) |
| `mri_in_oct.nii.gz` | MRI on the OCT base grid (0.15 mm), through the affine |
| `qc.png`, `qc_montage.png` | QC of the affine in four columns: OCT, MRI through the transform, a checkerboard of the two, and the OCT with the outlines of the MRI foreground (red) and the specimen mask (cyan). `qc.png` shows one plane per OCT array axis, `qc_montage.png` four |
| `qc_deform.png` | only with a deformation: MRI, OCT through the affine, OCT through the affine and the deformation, and the field magnitude, on three planes with the MRI outline |
| `result.json` | inputs, parameters and their hash, the transform, scores, contrast polarity, scales, the read-outs of §6, flags, time per step and peak memory |

A registration is judged by looking at it. In every plane of the QC images the MRI boundary should follow the edge of the OCT
specimen, and the OCT should lie on the same MRI anatomy.

## Inputs and assumptions

- The OCT shows a tissue block embedded in scatterer-doped agarose, which the texture mask of §1 separates from the tissue. A
  block that is not embedded needs its mask given with `--oct-mask`, and the OCT file itself serves when it is exactly zero
  outside the tissue. Exact zeros in the OCT are read as missing data.
- The OCT contrast may be inverted relative to the MRI.
- The MRI is an ex-vivo scan roughly cropped around the block. The crop holds the whole block, may hold tissue beyond it and is
  the only position prior.
- Inside the tissue the MRI shows a brighter and a darker class, as grey and white matter do in a FLASH image at a suitable flip
  angle. The medium around the MRI tissue is darker than half the brightest tissue and the intensities have no offset, or else
  the MRI foreground is given with `--mri-mask`.
- Both voxel sizes are correct, so the true scales are close to 1.
- A NIfTI header (sform or qform) may give any position and orientation, but its handedness must be that of the specimen. A TIFF
  or NPY stack (x, y, z = numpy axes 2, 1, 0), or a NIfTI file without sform and qform, is taken in its array frame, which must
  then have the handedness of the specimen. §3 searches proper rotations only, so a mirrored frame is not corrected.
- In both scans the tissue at the surface is brighter than what surrounds it, since §6 takes the surface edge as the steepest
  fall of intensity going outwards.

## Results on the I58 brainstem block

octreg was developed on this block, which has no reference registration, so its result is shown and not scored. On the I58 pair
(OCT 1457×2013×1595 at 20 µm, MRI crop 343×489×495 at 0.08 mm) the run takes 10 min 46 s, of which §6 is 21 s, with 5.2 GiB of
RAM and 1.6 GiB of GPU memory allocated by torch (RTX 5090, Windows). The outline term and §6 fit the OCT to the MRI tissue
boundary drawn in red below, which the standard tools do not use, so the internal structures are the independent check.

| sagittal | coronal | axial |
|---|---|---|
| <img src="docs/figures/result/sagittal_mri.png" width="300" alt="MRI, sagittal"> | <img src="docs/figures/result/coronal_mri.png" width="208" alt="MRI, coronal"> | <img src="docs/figures/result/axial_mri.png" width="208" alt="MRI, axial"> |
| <img src="docs/figures/result/sagittal_oct.png" width="300" alt="OCT, sagittal"> | <img src="docs/figures/result/coronal_oct.png" width="208" alt="OCT, coronal"> | <img src="docs/figures/result/axial_oct.png" width="208" alt="OCT, axial"> |
| <img src="docs/figures/result/sagittal_oct_boundary.png" width="300" alt="OCT and MRI tissue boundary, sagittal"> | <img src="docs/figures/result/coronal_oct_boundary.png" width="208" alt="OCT and MRI tissue boundary, coronal"> | <img src="docs/figures/result/axial_oct_boundary.png" width="208" alt="OCT and MRI tissue boundary, axial"> |

Three planes through the centroid of the MRI tissue in freeview. From top to bottom: the MRI crop, the OCT registered onto it
(`oct_in_mri.nii.gz`, affine and §6), and the same OCT with the boundary of the MRI foreground of §1 in red. Each panel is the
whole crop at one pixel per MRI voxel of 0.08 mm, at freeview's default grey window
([docs/figures/figures.json](docs/figures/figures.json)). Where the boundary runs beyond the OCT, at both ends of the block, the
MRI holds tissue outside its cut faces. The grey around the specimen is the agarose, and the horizontal stripes in the sagittal
and coronal planes are the sections. Sagittal planes have anterior on the right, and coronal and axial planes show the subject's
right on the left.

### Comparison with standard tools

Eight standard registration tools run on the same pair in the affine stage, each at the setting its documentation gives
([bench/baselines](bench/baselines/README.md)), with the OCT box-averaged to the 0.08 mm voxels of the MRI as the moving image.
The table gives properties that need no reference: the scale per OCT axis and the determinant of each affine.

| method | setting | start | scale per OCT axis | determinant |
|---|---|---|---|---|
| image centres aligned (the start) | not a registration | header orientation | 1.00 / 1.00 / 1.00 | 1.00 |
| octreg 2.1.0 | §1 to §5, default parameters | its orientation search | 0.99 / 0.96 / 0.98 | 0.94 |
| NiftyReg reg_aladin 2.1.2 | defaults, rigid then affine | image centres | 0.97 / 0.96 / 0.98 | 0.91 |
| greedy 1.4.0 | NMI, dof 6 then 12 | image centres | 1.19 / 0.93 / 0.90 | 0.91 |
| elastix 5.3.1 | model-zoo rigid then affine, Mattes MI | geometric centres | 0.92 / 0.84 / 0.97 | 0.74 |
| FreeSurfer mri_robust_register 8.2.0 | NMI, rigid then affine | centres of mass | 0.99 / 1.00 / 1.07 | 1.06 |
| ANTs 2.6.5 | antsRegistrationSyN.sh -t a | centres of mass | 0.86 / 0.62 / 0.92 | 0.49 |
| FreeSurfer mri_coreg 8.2.0 | NMI, dof 6 then 12 | its brute-force search | 0.92 / 0.81 / 1.00 | 0.74 |
| FSL FLIRT 6.0.7.23 | normmi, dof 6 then 12, headers scaled x10 | its rotation search (±90°) | 0.76 / 0.76 / 0.67 | 0.39 |
| SynthMorph (FreeSurfer 8.2.0) | -m affine | none | 0.82 / 0.62 / 1.20 | 0.49 |

| MRI | image centres aligned (the start) | octreg | NiftyReg reg_aladin |
|---|---|---|---|
| <img src="docs/figures/affine/mri_axial.png" width="175" alt="MRI"> | <img src="docs/figures/affine/start_axial.png" width="175" alt="image centres aligned"> | <img src="docs/figures/affine/octreg_axial.png" width="175" alt="octreg"> | <img src="docs/figures/affine/reg_aladin_axial.png" width="175" alt="NiftyReg reg_aladin"> |
| **greedy** | **elastix** | **mri_robust_register** | **ANTs** |
| <img src="docs/figures/affine/greedy_axial.png" width="175" alt="greedy"> | <img src="docs/figures/affine/elastix_axial.png" width="175" alt="elastix"> | <img src="docs/figures/affine/mri_robust_register_axial.png" width="175" alt="mri_robust_register"> | <img src="docs/figures/affine/ants_axial.png" width="175" alt="ANTs"> |
| **mri_coreg** | **FLIRT** | **SynthMorph** | |
| <img src="docs/figures/affine/mri_coreg_axial.png" width="175" alt="mri_coreg"> | <img src="docs/figures/affine/flirt_axial.png" width="175" alt="FLIRT"> | <img src="docs/figures/affine/synthmorph_axial.png" width="175" alt="SynthMorph"> | |

The axial plane of the result figure through each affine, with the MRI tissue boundary in red.

Five deformable tools then start from octreg's affine with the OCT resampled through it, so their panels compare the
deformation alone. The table gives the Jacobian of each field over the MRI foreground on the field's own grid (0.15 mm for
octreg, 0.08 mm for the tools), with J clipped at 10⁻⁶ where it is not positive. Times are on a machine shared with other jobs.

| method | setting | Jacobian min | folded (J ≤ 0) | SD of log J | time |
|---|---|---|---|---|---|
| no deformation (octreg's affine) | | 1 | 0 % | 0 | |
| octreg 2.1.0, §6 | strain-limited field on a 5 mm lattice | 0.94 | 0 % | 0.07 | 21 s |
| ANTs 2.6.5 SyN | antsRegistrationSyN.sh -t so, CC, no mask | 0.01 | 0 % | 0.46 | 42 min |
| elastix 5.3.1 | model-zoo B-spline, Mattes MI | 1.02 | 0 % | 0.02 | 21 s |
| NiftyReg reg_f3d 2.1.2 | defaults | −88.3 | 47.8 % | 6.33 | 3 min |
| ConvexAdam 0.2.0 | MIND-SSC, shipped defaults | −0.86 | 0.46 % | 1.10 | 8 s |
| greedy 1.4.0 | quick-start deformable command, WNCC | 0.10 | 0 % | 0.17 | 4 min |

| MRI | no deformation (octreg's affine) | octreg, §6 | ANTs SyN |
|---|---|---|---|
| <img src="docs/figures/deformable/mri_sagittal.png" width="175" alt="MRI"> | <img src="docs/figures/deformable/affine_sagittal.png" width="175" alt="no deformation"> | <img src="docs/figures/deformable/octreg_sagittal.png" width="175" alt="octreg, §6"> | <img src="docs/figures/deformable/ants_syn_sagittal.png" width="175" alt="ANTs SyN"> |
| **elastix** | **NiftyReg reg_f3d** | **ConvexAdam** | **greedy** |
| <img src="docs/figures/deformable/elastix_sagittal.png" width="175" alt="elastix"> | <img src="docs/figures/deformable/reg_f3d_sagittal.png" width="175" alt="NiftyReg reg_f3d"> | <img src="docs/figures/deformable/convexadam_sagittal.png" width="175" alt="ConvexAdam"> | <img src="docs/figures/deformable/greedy_sagittal.png" width="175" alt="greedy"> |

The sagittal plane of the result figure through octreg's affine and each field, with the MRI tissue boundary in red.
[BASELINES.md](bench/baselines/BASELINES.md) shows every tool on all three planes.

### Ablations

Each variant removes or replaces one element of the method and keeps the rest (`bench/ablate.py`).

- With an intensity threshold in place of the texture mask, or without the outline term, the section stripes, nearly level in
  the result, slant and the specimen leaves the boundary.
- Without the scale and shear penalty the stripes stand upright in a smaller block that leaves the boundary.
- With the outline alone, §4 ends at another pose and §5 moves the block to one whose panel looks like the result.
- Without §5 the posterior surface lies beyond the boundary.
- Without the orientation search, §4 and §5 end at the result from the header orientation of this pair, but from the OCT turned
  by 90° about the sectioning axis the block stays turned. With the search octreg returns to the result.

| intensity-threshold mask | no outline term | no scale and shear penalty | no fine-structure refinement |
|---|---|---|---|
| <img src="docs/figures/ablation/intensity_mask.png" width="175" alt="intensity-threshold mask"> | <img src="docs/figures/ablation/no_outline.png" width="175" alt="no outline term"> | <img src="docs/figures/ablation/no_penalty.png" width="175" alt="no scale and shear penalty"> | <img src="docs/figures/ablation/no_refinement.png" width="175" alt="no fine-structure refinement"> |
| **the outline alone** | **turned 90°, the start** | **turned 90°, no orientation search** | **turned 90°, octreg** |
| <img src="docs/figures/ablation/outline_only.png" width="175" alt="the outline alone"> | <img src="docs/figures/ablation/turned_start.png" width="175" alt="turned start"> | <img src="docs/figures/ablation/turned_no_search.png" width="175" alt="turned 90°, no orientation search"> | <img src="docs/figures/ablation/turned_octreg.png" width="175" alt="turned 90°, octreg"> |

The sagittal plane of the result figure through each variant's own affine. [bench/BENCHMARK.md](bench/BENCHMARK.md) gives every
variant of §1 to §5 with its values and the start test, and
[bench/results/I58/deform_ablation.md](bench/results/I58/deform_ablation.md) the variants of §6.

## Test on the data of Costantini et al. (2023)

Costantini et al. imaged a block of Broca's area from one fixed human hemisphere with serial-sectioning OCT, scanned the
hemisphere with 7 T MRI at 0.15 mm and registered the MRI to the OCT with an affine transform ([Science Advances 9:eadg3844,
2023](https://doi.org/10.1126/sciadv.adg3844)). The data are public in
[DANDI:000026](https://dandiarchive.org/dandiset/000026/draft) (sample I46, CC BY 4.0), and their affine serves as the
reference R. R is their own registration, not ground truth: a fit of the OCT vessels to the MRI vessel labels alone lies
0.16 mm from it on average, so distances below about 0.3 mm are not read. Parts of the method were designed with runs on these
data, so this is not a held-out test.

The MRI was cropped around the block as R places it, ten times per condition with a seeded random shift: with a 3 mm margin on
the FLASH images at 10°, 20° and 30°, and with 6 and 12 mm margins at 10°. Every method got the same OCT, masked to the tissue
since the block is not embedded, and octreg also got that mask. The distance is the mean displacement from R over the OCT
tissue.

| condition | within 1 mm of R | distance from R (mm) |
|---|---|---|
| 10° FLASH, 3 mm margin | 10 of 10 | 0.37 to 0.42 |
| 20° FLASH, 3 mm margin | 10 of 10 | 0.44 to 0.60 |
| 30° FLASH, 3 mm margin | 10 of 10 | 0.31 to 0.35 |
| 10° FLASH, 6 mm margin | 10 of 10 | 0.37 to 0.41 |
| 10° FLASH, 12 mm margin | 10 of 10 | 0.36 to 0.41 |
| 10° FLASH, 3 mm margin, octreg's own specimen mask | 10 of 10 | 0.52 to 0.67 |

On the crop with the median distance, from the OCT turned by 90° and 180° about each array axis and by 20 rotations drawn at
random, octreg ends 0.38 to 0.42 mm from R every time. From the header orientation, 7 mm and 122° from R, the eight standard
tools come within 1 mm in none of their 96 runs, and none does with its own rotation search (FLIRT, ANTs and greedy) or with
the OCT tissue as a mask of the moving image.

| OCT plane | OCT | OCT with the line through R | OCT with the line through octreg | MRI through R | MRI through octreg |
|---|---|---|---|---|---|
| 102 | <img src="docs/figures/i46/enface/flip1_C3_k0_p102_oct.png" width="150" alt="OCT, plane 102"> | <img src="docs/figures/i46/enface/flip1_C3_k0_p102_oct_line_reference.png" width="150" alt="OCT with the line through R, plane 102"> | <img src="docs/figures/i46/enface/flip1_C3_k0_p102_oct_line_octreg.png" width="150" alt="OCT with the line through octreg, plane 102"> | <img src="docs/figures/i46/enface/flip1_C3_k0_p102_mri_reference.png" width="150" alt="MRI through R, plane 102"> | <img src="docs/figures/i46/enface/flip1_C3_k0_p102_mri_octreg.png" width="150" alt="MRI through octreg, plane 102"> |
| 300 | <img src="docs/figures/i46/enface/flip1_C3_k0_p300_oct.png" width="150" alt="OCT, plane 300"> | <img src="docs/figures/i46/enface/flip1_C3_k0_p300_oct_line_reference.png" width="150" alt="OCT with the line through R, plane 300"> | <img src="docs/figures/i46/enface/flip1_C3_k0_p300_oct_line_octreg.png" width="150" alt="OCT with the line through octreg, plane 300"> | <img src="docs/figures/i46/enface/flip1_C3_k0_p300_mri_reference.png" width="150" alt="MRI through R, plane 300"> | <img src="docs/figures/i46/enface/flip1_C3_k0_p300_mri_octreg.png" width="150" alt="MRI through octreg, plane 300"> |
| 498 | <img src="docs/figures/i46/enface/flip1_C3_k0_p498_oct.png" width="150" alt="OCT, plane 498"> | <img src="docs/figures/i46/enface/flip1_C3_k0_p498_oct_line_reference.png" width="150" alt="OCT with the line through R, plane 498"> | <img src="docs/figures/i46/enface/flip1_C3_k0_p498_oct_line_octreg.png" width="150" alt="OCT with the line through octreg, plane 498"> | <img src="docs/figures/i46/enface/flip1_C3_k0_p498_mri_reference.png" width="150" alt="MRI through R, plane 498"> | <img src="docs/figures/i46/enface/flip1_C3_k0_p498_mri_octreg.png" width="150" alt="MRI through octreg, plane 498"> |

The crop with the median distance under octreg (10° FLASH, 3 mm margin, k = 0) in planes 102, 300 and 498 of the 12 µm masked
OCT, the way Costantini et al. show their OCT. From left: the OCT, the OCT with the white-matter boundary of the authors' manual
MRI labels placed through R and through octreg (red), and the MRI crop resampled through R and through octreg, all in freeview.
Outside the OCT tissue the line continues through the MRI, and where the white matter reaches the edge of the crop it runs along
that edge. The grey levels are freeview's default window ([docs/figures/i46/figures.json](docs/figures/i46/figures.json)).

| R | octreg | NiftyReg reg_aladin | greedy | ANTs |
|---|---|---|---|---|
| <img src="docs/figures/i46/enface/flip1_C3_k0_p300_oct_line_reference.png" width="120" alt="OCT plane 300 with the line through R"> | <img src="docs/figures/i46/enface/flip1_C3_k0_p300_oct_line_octreg.png" width="120" alt="OCT plane 300 with the line through octreg"> | <img src="docs/figures/i46/enface/flip1_C3_k0_p300_oct_line_reg_aladin.png" width="120" alt="OCT plane 300 with the line through NiftyReg reg_aladin"> | <img src="docs/figures/i46/enface/flip1_C3_k0_p300_oct_line_greedy.png" width="120" alt="OCT plane 300 with the line through greedy"> | <img src="docs/figures/i46/enface/flip1_C3_k0_p300_oct_line_ants.png" width="120" alt="OCT plane 300 with the line through ANTs"> |
| **mri_robust_register** | **mri_coreg** | **FLIRT** | **SynthMorph** | **elastix** |
| <img src="docs/figures/i46/enface/flip1_C3_k0_p300_oct_line_mri_robust_register.png" width="120" alt="OCT plane 300 with the line through mri_robust_register"> | <img src="docs/figures/i46/enface/flip1_C3_k0_p300_oct_line_mri_coreg.png" width="120" alt="OCT plane 300 with the line through mri_coreg"> | <img src="docs/figures/i46/enface/flip1_C3_k0_p300_oct_line_flirt.png" width="120" alt="OCT plane 300 with the line through FLIRT"> | <img src="docs/figures/i46/enface/flip1_C3_k0_p300_oct_line_synthmorph.png" width="120" alt="OCT plane 300 with the line through SynthMorph"> | no transform (tool error) |
| **FLIRT, its search over ±180°** | **ANTs, antsAI start** | **greedy, its search** |  |  |
| <img src="docs/figures/i46/enface/flip1_C3_k0_p300_oct_line_flirt_search.png" width="120" alt="OCT plane 300 with the line through FLIRT, its search over ±180°"> | <img src="docs/figures/i46/enface/flip1_C3_k0_p300_oct_line_ants_antsai.png" width="120" alt="OCT plane 300 with the line through ANTs, antsAI start"> | <img src="docs/figures/i46/enface/flip1_C3_k0_p300_oct_line_greedy_search.png" width="120" alt="OCT plane 300 with the line through greedy, its search"> |  |  |

OCT plane 300 of the same crop with the white-matter boundary placed through R, through octreg and through each standard tool
at its documented setting and with its own search. For FLIRT, SynthMorph and the searches of FLIRT and greedy the boundary does
not cross this plane inside the OCT field of view, so their panels show no line. elastix ends without a transform on this
crop. [bench/i46/README.md](bench/i46/README.md) gives the protocol and
[bench/results/I46/tables.md](bench/results/I46/tables.md) every number.

## Reproducing the benchmark

[bench/README.md](bench/README.md) describes how to run the I58 benchmark, whose data are not distributed with the repository,
and [bench/i46/README.md](bench/i46/README.md) runs the test on the data of Costantini et al. (2023) from the download to the
tables and figures. The stored results are in `bench/results/`. `python bench/evaluate.py --selftest` checks the evaluation on
synthetic arrays without any of the data.

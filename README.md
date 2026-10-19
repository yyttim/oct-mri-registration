# octreg

octreg registers a serial-sectioning OCT volume of a tissue block to an ex-vivo MRI cropped around the block. It finds an affine
transform from any starting orientation, with no labels and no manual start.

| sagittal | coronal | axial |
|---|---|---|
| <img src="docs/figures/result/sagittal_oct.png" width="300" alt="registered OCT, sagittal"> | <img src="docs/figures/result/coronal_oct.png" width="208" alt="registered OCT, coronal"> | <img src="docs/figures/result/axial_oct.png" width="208" alt="registered OCT, axial"> |
| <img src="docs/figures/result/sagittal_oct_boundary.png" width="300" alt="registered OCT with the MRI tissue boundary, sagittal"> | <img src="docs/figures/result/coronal_oct_boundary.png" width="208" alt="registered OCT with the MRI tissue boundary, coronal"> | <img src="docs/figures/result/axial_oct_boundary.png" width="208" alt="registered OCT with the MRI tissue boundary, axial"> |
| <img src="docs/figures/result/sagittal_mri.png" width="300" alt="MRI, sagittal"> | <img src="docs/figures/result/coronal_mri.png" width="208" alt="MRI, coronal"> | <img src="docs/figures/result/axial_mri.png" width="208" alt="MRI, axial"> |

The OCT of the I58 brainstem block registered by octreg onto its MRI crop (top), the same with the boundary of the MRI tissue in
red (middle), and the MRI (bottom), on three planes in freeview. On the sagittal plane the bright rim at the top lies just
outside the line, the rounded tissue edge at the upper left and a bright band just below the step in the line at the lower left
lie outside it, and along the lower right the surface lies inside the line with a dark gap. On the coronal plane the right part
of the striped upper right corner of the specimen lies outside the line, and so does a separate piece of tissue below it except
for a thin strip along its lower left edge. Below that piece the specimen ends short of the line. On the axial plane the
specimen ends short of the line along the right side, where the OCT shows a dark band with bright fragments and then plain grey.
On the sagittal and coronal planes the line also runs past the top and bottom of the OCT image.

## Method

The search and the refinement maximise one similarity, the correlation of the gradient-orientation tensors of the two volumes
(§2 of [docs/METHOD.md](docs/METHOD.md)),

$$F = \operatorname{corr}\big(P(n_\text{OCT}),\, P(n_\text{MRI})\big), \qquad P(n) = n n^\top,$$

where n is the normalised image gradient and ⟨P(n), P(n′)⟩ = (n·n′)², the squared product of normalised gradient fields (Haber
and Modersitzki, 2007). P(n) = P(−n), so F needs neither an intensity mapping between OCT and MRI nor their contrast polarity.
As an inner product of six-channel fields, F can be computed at every translation by FFT, as in Cross-Sim-NGF (Öfverstedt et
al., 2022). F is normalised over the OCT block, with the MRI tensors zero outside the MRI foreground.

- **§1 Masks.** The OCT specimen is the voxels with signal and isotropic texture, both decided by one threshold rule. Isotropic
  texture separates tissue from doped agarose, which can be as bright as tissue but whose section stripes and tile seams vary
  along one axis only. The MRI foreground lies above a histogram valley whose darker class lies mostly outside the convex hull
  of the brighter voxels, as the embedding medium does, and not inside it, as white matter under cortex does.
- **§3 Orientation search.** For each of 8,000 rotations, F is scored at every translation inside the crop. Poses are proposed
  by F and, where both images show medium around the tissue, also by (F + O) / 2. The outline term O, the correlation of the OCT
  specimen mask with the MRI foreground, places a block by its shape. 24 poses are kept.
- **§4 Affine refinement.** Every pose is refined on F to a 12-parameter affine under a prior on scale and shear, on a 0.15 mm
  grid with gradient scales from 0.6 down to 0.3 mm, and the pose with the lowest final loss, 1 − F plus the prior, wins.

[docs/METHOD.md](docs/METHOD.md) gives the equations and the parameters.

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
octreg apply --run OUT --moving X --reference Y -o Z [--inverse] [--oct-spacing-um Z,Y,X]
```

`register` writes the outputs below into `OUT`. The OCT can be NIfTI, TIFF, OME-TIFF or NPY, and the MRI is NIfTI.
`--oct-spacing-um` gives the OCT voxel spacing in µm along the numpy axes z, y, x, which a TIFF or NPY without spacing needs. It
overrides the spacing in the file, also that of an `--oct-mask` of the OCT's array shape. `--oct-mask` and `--mri-mask` replace
the masks of §1 with files whose positive voxels are the mask, and the OCT file itself serves when it is exactly zero outside
the tissue. `--params` reads overrides of `octreg.params.Params` from JSON. `apply` resamples an OCT-frame volume onto an
MRI-frame reference through the affine, or an MRI-frame volume onto an OCT-frame reference through the inverse affine with
`--inverse`. The same functions are available in Python: `from octreg.register import register, apply`.

| file | content |
|---|---|
| `T_oct2mri.txt` | 4×4 affine in mm from the world frame of the OCT file to that of the MRI file |
| `oct2mri.lta`, `oct2mri_itk.txt` | the same affine for FreeSurfer and ITK |
| `oct_in_mri.nii.gz` | the OCT on the MRI grid, through the affine |
| `mri_in_oct.nii.gz` | the MRI on the OCT base grid (0.15 mm), through the affine |
| `qc.png`, `qc_montage.png` | visual QC of the affine |
| `result.json` | inputs, parameters, the transform, the scores of every step and the time per step |

A registration is judged by looking at it: the OCT should lie on the same MRI anatomy, and where both images show medium around
the tissue, the MRI tissue boundary should follow the edge of the OCT specimen.

## Results

### I58 brainstem block

The pair is an OCT of 1457 × 2013 × 1595 voxels at 20 µm and an MRI crop of 343 × 489 × 495 voxels at 0.08 mm. The block has no
reference registration, so its result is shown and not scored. Eight standard registration tools ran on the same pair, with the
OCT box-averaged to the 0.08 mm voxels of the MRI as the moving image, each at the setting its documentation gives, except that
FLIRT ran with both headers scaled x10, because its schedule resamples the images to fixed voxel sizes in mm
([bench/baselines](bench/baselines/README.md)).

| MRI | image centres aligned (the start) | octreg | NiftyReg reg_aladin |
|---|---|---|---|
| <img src="docs/figures/affine/mri_axial.png" width="175" alt="MRI"> | <img src="docs/figures/affine/start_axial.png" width="175" alt="image centres aligned"> | <img src="docs/figures/affine/octreg_axial.png" width="175" alt="octreg"> | <img src="docs/figures/affine/reg_aladin_axial.png" width="175" alt="NiftyReg reg_aladin"> |
| **greedy** | **elastix** | **mri_robust_register** | **ANTs** |
| <img src="docs/figures/affine/greedy_axial.png" width="175" alt="greedy"> | <img src="docs/figures/affine/elastix_axial.png" width="175" alt="elastix"> | <img src="docs/figures/affine/mri_robust_register_axial.png" width="175" alt="mri_robust_register"> | <img src="docs/figures/affine/ants_axial.png" width="175" alt="ANTs"> |
| **mri_coreg** | **FLIRT** | **SynthMorph** | |
| <img src="docs/figures/affine/mri_coreg_axial.png" width="175" alt="mri_coreg"> | <img src="docs/figures/affine/flirt_axial.png" width="175" alt="FLIRT"> | <img src="docs/figures/affine/synthmorph_axial.png" width="175" alt="SynthMorph"> | |

The axial plane of the I58 figure at the top, the MRI and the OCT through each affine, with the MRI tissue boundary in red.
Through the SynthMorph affine only a corner of the OCT image, at the lower right, reaches this plane.
[BASELINES.md](bench/baselines/BASELINES.md) shows every tool on all three planes, with the scale and determinant of each
affine.

### Test on the data of Costantini et al. (2023)

Costantini et al. imaged a block of Broca's area from one fixed human hemisphere with serial-sectioning OCT, scanned the
hemisphere with 7 T MRI at 0.15 mm and registered the MRI to the OCT with an affine transform ([Science Advances 9:eadg3844,
2023](https://doi.org/10.1126/sciadv.adg3844)). The data are public in
[DANDI:000026](https://dandiarchive.org/dandiset/000026/draft) (sample I46, CC BY 4.0), and their affine serves as the reference
R. R is their own registration, not ground truth: a fit of the OCT vessels to the MRI vessel labels alone lies 0.16 mm from it
on average and 0.31 mm at the 95th percentile, so distances below about 0.3 mm are not read. octreg was developed on I58 and
checked on these data during its development, so this is not a held-out test.

The MRI was cropped around the block as R places it, ten times per condition (crops k = 0 to 9) with a seeded random shift: with
a 3 mm margin on the three 7 T FLASH images, of flip angles 10°, 20° and 30°, and with 6 and 12 mm margins on the 10° FLASH.
Every method got the OCT as released, in which most of the agarose around the block is zero, and octreg made its own masks. The
distance is the mean displacement from R over the test's specimen mask (`bench/i46/octmask_24um.nii.gz`, made from the OCT by a
threshold and morphology).

| condition | within 1 mm of R | distance from R (mm) |
|---|---|---|
| 10° FLASH, 3 mm margin | 10 of 10 | 0.35 to 0.36 |
| 20° FLASH, 3 mm margin | 10 of 10 | 0.36 to 0.37 |
| 30° FLASH, 3 mm margin | 10 of 10 | 0.32 |
| 10° FLASH, 6 mm margin | 10 of 10 | 0.35 to 0.36 |
| 10° FLASH, 12 mm margin | 10 of 10 | 0.35 to 0.37 |
| 10° FLASH, 3 mm margin, the test's specimen mask given | 10 of 10 | 0.37 |

The last row gives octreg that mask as `--oct-mask` in place of its own. From the OCT turned by 90° and 180° about each array
axis and by 10 random rotations, on the k = 0 crop of the 10° FLASH with a 3 mm margin, octreg ends 0.35 to 0.37 mm from R every
time. From the header orientation with the centres of the two image boxes aligned, 7.0 mm (median over the ten crops of the 10°
FLASH with a 3 mm margin) and 122° from R, the eight standard tools come within 1 mm in none of their 96 runs, the three with a
global rotation search (FLIRT, ANTs and greedy) in none with that search turned on, and the seven that take a mask of the moving
image in none with the test's specimen mask as that mask.

| OCT | OCT with the line through R | OCT with the line through octreg | MRI through R | MRI through octreg |
|---|---|---|---|---|
| <img src="docs/figures/i46/enface/flip1_C3_k8_p300_oct.png" width="150" alt="OCT, plane 300"> | <img src="docs/figures/i46/enface/flip1_C3_k8_p300_oct_line_reference.png" width="150" alt="OCT with the line through R"> | <img src="docs/figures/i46/enface/flip1_C3_k8_p300_oct_line_octreg.png" width="150" alt="OCT with the line through octreg"> | <img src="docs/figures/i46/enface/flip1_C3_k8_p300_mri_reference.png" width="150" alt="MRI through R"> | <img src="docs/figures/i46/enface/flip1_C3_k8_p300_mri_octreg.png" width="150" alt="MRI through octreg"> |

Plane 300 of the OCT (12 µm in the plane), shown in the OCT's own planes as Costantini et al. show their OCT, for the crop at
the lower median of the ten distances under octreg (10° FLASH, 3 mm margin, k = 8). The red line is the white-matter boundary of
the authors' manual MRI labels, placed through R and through octreg's affine, and the last two panels show the MRI crop
resampled through each, all in freeview at its default window. The line through octreg follows the line through R, a little
lower where it crosses the left half of the tissue, and cuts off a smaller part of the upper left corner of the tissue. The
white matter is dark in both MRI panels. [bench/i46/README.md](bench/i46/README.md) gives the protocol, the other planes and
every standard tool, and [bench/results/I46/tables.md](bench/results/I46/tables.md) every number.

### Ablations

Each variant changes one thing in the method (`bench/ablate.py`). On the ten crops of the 10° FLASH with a 3 mm margin:

| variant | within 1 mm of R | distance from R, median (mm) |
|---|---|---|
| octreg | 10 of 10 | 0.36 |
| no orientation search: the refinement from the header orientation with the image box centres aligned | 0 of 10 | 6.53 |
| no refinement: the first search pose | 7 of 10 | 0.87 |
| only the first search pose refined | 10 of 10 | 0.36 |
| signed normalised gradients instead of their tensor | 0 of 10 | 1.28 |
| the same, with the OCT intensities negated | 0 of 10 | 8.82 |
| no scale and shear prior | 10 of 10 | 0.47 |
| the trace-free tensor n nᵀ − \|n\|² I / 3 instead of n nᵀ | 10 of 10 | 0.38 |
| in the search, OCT gradients over the whole specimen mask instead of only its part more than 0.8 mm inside | 8 of 10 | 0.36 |

The outline is never used on these crops, whose MRI is all tissue. On I58:

| no outline term | signed gradients | signed gradients, OCT negated | no refinement |
|---|---|---|---|
| <img src="docs/figures/ablation/no_outline.png" width="175" alt="no outline term"> | <img src="docs/figures/ablation/signed.png" width="175" alt="signed gradients"> | <img src="docs/figures/ablation/signed_neg.png" width="175" alt="signed gradients, OCT negated"> | <img src="docs/figures/ablation/no_refine.png" width="175" alt="no refinement"> |
| **turned 90°, the start** | **turned 90°, no orientation search** | **turned 90°, octreg** | |
| <img src="docs/figures/ablation/turned_start.png" width="175" alt="turned 90°, the start"> | <img src="docs/figures/ablation/turned_no_search.png" width="175" alt="turned 90°, no orientation search"> | <img src="docs/figures/ablation/turned_octreg.png" width="175" alt="turned 90°, octreg"> | |

The sagittal plane of the I58 figure through each variant, with the MRI tissue boundary in red. The last three panels start from
the OCT turned by 90° about its sectioning axis. [bench/README.md](bench/README.md) shows every variant of `bench/ablate.py`.

## Reproducing the results

[bench/README.md](bench/README.md) gives the commands of the I58 benchmark, whose data are not distributed with the repository,
and [bench/i46/README.md](bench/i46/README.md) those of the test on the data of Costantini et al. (2023), from the download to
the tables and figures. The stored results are in `bench/results/`.

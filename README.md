# octreg

Label-free affine registration of serial-section OCT blocks to ex-vivo MRI, with a small smooth deformation on top.

octreg aligns an OCT volume of a tissue block embedded in scatterer-doped agarose to an MRI roughly cropped around the block.
It writes the affine transform between the two input files, a smooth displacement field on top of it where the data support
one, each volume resampled into the other's frame, and QC images.

## Installation

```
pip install -e .
```

Requires Python 3.9 or newer and PyTorch. A CUDA GPU is recommended for full-size volumes.

## Usage

```
octreg register OCT MRI -o OUT
octreg qc --run OUT --oct OCT --mri MRI [--T F] [-o PREFIX]
octreg apply --run OUT --moving X --reference Y -o Z [--inverse] [--affine-only] [--oct-spacing-um Z,Y,X]
```

`register` runs the registration. The OCT can be NIfTI, TIFF, OME-TIFF or NPY (`--oct-spacing-um Z,Y,X` when the file has no
spacing) and the MRI is NIfTI. The file frames fix the handedness: a NIfTI orientation header (sform or qform) must be correct,
and a TIFF or NPY stack (x, y, z = numpy axes 2, 1, 0) or a NIfTI file without one is taken in its array frame, which must then
have the handedness of the specimen.
`--oct-mask` and `--mri-mask` replace the automatic masks, `--params` reads parameter overrides from JSON, and `--device cpu`
runs without a GPU. `qc` renders the QC images of the affine again, for the run's transform or another one given with `--T`. `-o` sets the
output prefix (default `OUT/qc`, or `OUT/qc_<T stem>` for another transform), and `--oct-spacing-um`, `--oct-mask` and
`--mri-mask` default to the run's.
`apply` resamples an OCT-frame volume onto the grid of an MRI-frame `--reference`, through the affine and, when the run wrote
one, the deformation field (`--affine-only` leaves the field out). With `--inverse` it resamples an MRI-frame volume onto an
OCT-frame reference through the affine alone, since the field is not inverted. `--oct-spacing-um` gives the spacing of the
OCT-frame file when it is a TIFF or NPY that no longer sits where the run read it.

## Outputs

| file | content |
|---|---|
| `T_oct2mri.txt`, `T_mri2oct.txt` | 4×4 affine in mm between the worlds of the input files |
| `oct2mri.lta`, `oct2mri_itk.txt` | the same transform for FreeSurfer and ITK |
| `oct2mri_warp.nii.gz` | the smooth deformation on top of the affine, only when one is applied: displacement u (mm, MRI world axes) on the 0.15 mm MRI grid as a NIfTI vector image. The registered OCT at the MRI point x is OCT(T⁻¹(x + u(x))) |
| `oct_in_mri.nii.gz` | OCT on the MRI grid through the affine and the deformation |
| `oct_in_mri_affine.nii.gz` | OCT on the MRI grid through the affine alone (the same image when no deformation is applied) |
| `mri_in_oct.nii.gz` | MRI on a 0.15 mm grid in the OCT frame, through the affine |
| `qc.png`, `qc_montage.png` | visual QC of the affine, four columns: OCT, MRI through the transform, a checkerboard of the two, and the OCT with the MRI foreground (red) and specimen mask (cyan) outlines. In every plane the MRI boundary should follow the edge of the OCT specimen and the OCT should lie on the same MRI anatomy |
| `qc_deform.png` | only with a deformation: MRI, OCT through the affine, OCT through affine and deformation, and the field magnitude, on three planes with the MRI outline |
| `result.json` | inputs, parameters and their hash, the transform, scores, fine-structure agreement, contrast polarity, scales, deformation read-outs, flags, time per step and peak memory |

## Method

1. **Specimen mask from isotropic texture.** Agarose artefacts vary along one array axis, tissue texture along all three.
2. **One score for structure and outline.** Both scans become bright/dark tissue maps, compared together with the specimen
   outline (OCT tissue must lie on MRI tissue, while the MRI may hold tissue beyond the block). The sign of the structure term gives
   the contrast polarity.
3. **Orientation search in the crop.** FFT search over 8,000 rotations and all translations inside the MRI crop.
4. **Prior-bounded affine refinement.** A 12-parameter affine fit of the best poses under a scale and shear prior.
5. **Fine-structure refinement.** The best pose is refined on the gradient orientations inside both scans (normalised gradient
   fields), which follow fibre bundles and vessels and need no intensity mapping.
6. **Smooth deformation.** A small displacement field on top of the affine, fitted to interior matches of the same gradient orientations and to the offset between the surface edges of the two scans along the MRI boundary normals, as flexible as a
   strain limit allows. Without a held-out gain over the affine no field is applied. The affine stays the primary result.

The method assumes an OCT block embedded in scatterer-doped agarose and an MRI cropped around it. It was developed and
benchmarked on the I58 brainstem pair, which is unpublished, so the figures of that pair are drawn locally by the bench scripts
and are not in the repository. Results on the cortex blocks of DANDI:000026 (public, CC-BY 4.0, Costantini et al. 2023), which
are not embedded, are withdrawn until a re-run: `bench/dandi.py` built six of the eight MRI crops through label headers that
do not match the MRI.

Details, evaluation and parameters: [docs/METHOD.md](docs/METHOD.md). A comparison with standard registration tools on the
I58 pair: [bench/baselines/BASELINES.md](bench/baselines/BASELINES.md).

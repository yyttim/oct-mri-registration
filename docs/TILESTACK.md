# tilestack: internal registration of serial-sectioning OCT slabs

tilestack registers the slabs of a serial-sectioning OCT block to each other and writes them as one volume. Each slab is
a block-face OCT volume of the scattering coefficient mu_s, stitched from a mosaic of tiles and imaged before the next cut.
The imaging window reaches deeper than the cut step, so neighbouring slabs image the same tissue. tilestack matches that
shared tissue, places every slab and every mosaic tile from the matches in one solve, and composes the registered volume
from the raw mu_s of the slabs at their registered positions, with no other change to the values.

## Method

**Cut surface.** In every column of a slab the cut is the plane where log10 mu_s rises most sharply along depth: the
three planes from it down lie above the three planes over it, by as much as possible. A rise and not a level, because the
water above the cut reads brighter the less signal lies under it. A column more than 0.06 mm from the median of its
3 × 3 neighbourhood takes that median. The planes from 0.06 mm under the cut are used, below the surface reflection.

**Correspondences.** Each slab is matched to the next two. Blocks of the newer slab, 3 planes of log10 mu_s at 0.08 to
0.40 mm under its cut, are found in the older slab by phase correlation: a coarse pass with 2.56 mm blocks gives the
pair's depth offset as a plane and its in-plane shift, a dense pass with 1.28 mm blocks searches ±0.10 mm in depth around
that plane. The in-plane peak is refined on an upsampled DFT (Guizar-Sicairos et al., 2008) and the depth to a quarter
plane. Matches with a clear correlation peak inside the depth search are kept. Two slabs share no usable tissue when
fewer than 8 coarse blocks agree on their depth offset within 0.04 mm.

**Placement.** Slab s maps its raw voxel (x, y, z) in mosaic tile k to the common frame by

$$(X, Y, Z) = (x,\ y,\ z) + a_s + f_s(x, y) + e_s[k],$$

with three parts, each acting in-plane and in depth: a translation of the slab a_s (stage step, motion of the block), a
smooth deformation f_s, which also carries the slab's tilt (a cubic B-spline with 2.56 mm control spacing, Rueckert et
al., 1999), and a translation e_s[k] of each mosaic tile (acquisition). Every match says that its two ends land on the
same point. All parts of all slabs are solved at once by robust least squares (Huber loss), with a group-L1
penalty on each tile translation and a fused-L1 penalty between neighbouring tiles of a slab (Tibshirani et al., 2005),
since tile errors come in rows and columns, and a weak ridge on the B-spline. The matches fix the slabs only relative to
each other, so the common frame is the mean slab.

**Volume.** Every voxel is taken from the newest slab whose cut lies at least 0.06 mm above it and whose window reaches
it (down to `deep_a_mm`), as that slab's raw mu_s resampled at the registered position (trilinear). Each slab is
resampled tile by tile. There is no brightness correction, no blending at the seams and no filling: a voxel no slab holds
stays 0. A source map gives the slab of every voxel, −1 for none, plus 1000 below the slab's reliable planes, which end
at `deep_mm`. The NIfTI files place the common-frame point (X, Y, Z) at world (X, Y, −Z) mm, the third axis pointing up
out of the block face ([Scope](#scope)).

**MRI reference.** A transform that all slabs share changes no match, so the matches leave it open. With `--mri`, octreg
([METHOD.md](METHOD.md)) finds its affine part, one affine of the registered volume to the block's MRI, as Wang et al.
(NeuroImage 100:395, 2014) and Costantini et al. (2023) register serial-sectioning OCT to MRI. octreg reads a tissue
mask and a levelled copy of the 40 µm volume, both on the volume's grid and world, so the affine holds for the volume.
The mask replaces octreg's specimen mask, which takes the larger part of the measured volume to be the specimen, and in
a mosaic that can be the gel. log10 mu_s is smoothed over the reliable voxels alone (normalised Gaussian, σ 0.16 mm),
since below the reliable planes gel and tissue read alike, and the tissue is the measured voxels above Otsu's threshold
of the smoothed map, with holes filled in every array plane. Tissue mu_s changes with the depth under a cut and slabs
differ in level, so the volume, which changes slab under every cut, has an edge there. In the copy every voxel is
divided by the median of the tissue of its slab at its depth under that slab's cut, counted in planes from where the
slab begins in its column. The matches place the slabs more finely than an MRI voxel, so octreg works at the MRI's
resolution. Its base grid is the MRI voxel in place of 0.15 mm. The σ of its last pass, 0.3 mm, is two voxels of the
0.15 mm grid, so further passes divide σ by 1.5, the largest ratio between octreg's own passes, and the last is at σ of
two base voxels.

## Usage

```
python -m tilestack BLOCK_JSON OUT_DIR [--full] [--layout LATTICE_JSON] [--mri MRI]
```

`BLOCK_JSON` describes the block:

```json
{
 "slab_file": "/data/block/Mus_slice{:03d}.mat",
 "work": "/data/block/tilestack_work",
 "slabs": [5, 6, 7, 8],
 "voxel_mm": 0.02,
 "neighbours": 2,
 "margin_mm": 0.06,
 "deep_mm": 0.86,
 "deep_a_mm": 1.12
}
```

A slab is a MATLAB v7.3 file with one array `M` of mu_s per µm, read by h5py as (z, y, x), z growing with depth from
above the cut. `work` holds the caches of a run (the slabs at 40 µm, cut surfaces, matches), valid only for the slab
files and settings they were made with. `neighbours` is the number of slabs ahead that still share tissue, the imaging
window's depth against the cut step. `deep_mm` ends the reliable planes, where the estimate of a uniform material such
as the gel starts to rise toward the bottom of the window. `deep_a_mm` is the deepest plane at which the older slab of a
pair still matches. The tile lattice is read from the staircase outline of the coverage masks when the mosaic is rotated
by about 0.4° or more, and is otherwise given with `--layout` (X0, Y0, PX, PY, SX, SY, W, H in pixels, rows, cols).
`--full` also writes the volume at the raw voxel. `--mri` registers the volume to an MRI NIfTI cropped around the block,
the input octreg takes. The matching, the composition and octreg run on a CUDA GPU.

| file | content |
|---|---|
| `registered_40um.nii.gz`, `registered_40um_source.nii.gz` | the registered volume on a 40 µm grid in-plane and the raw planes in depth, and its source map |
| `registered_20um.nii`, `registered_20um_source.nii` | the same at the raw voxel (`--full`) |
| `transforms.json` | the placement of every slab and tile in mm, with the formulas in its description |
| `solution.pkl` | the solver's output |
| `run.json` | the block, the settings, the pairs and the residuals of the matches |
| `mri/` | with `--mri`: octreg's outputs for the levelled copy, whose `T_oct2mri.txt`, `oct2mri.lta` and `oct2mri_itk.txt` hold for the registered volumes, which share its world. `levelled.nii.gz` and `tissue.nii.gz` are what octreg read, and `registered_40um_in_mri.nii.gz` is the volume's mu_s on the MRI grid |

## Scope

octreg takes the handedness from the header and searches proper rotations only. The world of the NIfTI files has the
specimen's handedness when a plane of a slab, shown with x (the first MATLAB dimension) down and y across, is the block
face seen from above, and slabs that show it mirrored need one in-plane axis reversed in their files. octreg's prior
holds the scales near 1, so the slabs should have the voxel `voxel_mm` along all three axes in the tissue, the depth
axis already converted from optical path. The tissue mask takes the gel to be darker than the tissue and a large part of
the volume, as for a block in plain agarose imaged with a margin, and its σ is half a cut step of about 0.3 mm, which
damps the change with depth under a cut. The levelling takes tissue mu_s to change with depth under the cut by the same
factor across a slab. With `--mri` the base grid is the largest side of the MRI voxel, kept between octreg's own fine
and base grids, 0.04 and 0.15 mm, so an MRI of 0.15 mm or coarser is registered with octreg's own parameters.
# tilestack: internal registration of serial-sectioning OCT slabs

tilestack registers the slabs of a serial-sectioning OCT block to each other and writes them as one volume. Each slab is
a block-face OCT volume of the scattering coefficient mu_s, stitched from a mosaic of tiles and imaged before the next cut.
The imaging window reaches deeper than the cut step, so neighbouring slabs image the same tissue. tilestack matches that
shared tissue, places every slab and every mosaic tile from the matches in one solve, and composes the registered volume
from the raw mu_s of the slabs at their registered positions, with no other change to the values.

## Method

**Cut surface.** In every column of a slab the cut is the plane where log10 mu_s rises most sharply along depth: the
smallest of the three planes from it down exceeds the largest of the three planes over it (the one or two there are at
the top of the window) by as much as possible. A rise and not a level, because the water above the cut reads brighter
the less signal lies under it. A column more than 0.06 mm from the median of its 3 × 3 neighbourhood takes that median.
Columns are covered where the middle plane of the window holds data. The planes from `margin_mm` (0.06 mm) under the cut
are used, below the surface reflection.

**Correspondences.** Each slab is matched to the next `neighbours` slabs. Blocks of the newer slab, 3 planes of log10
mu_s at five depths from 0.08 to 0.40 mm under its cut, are found in the older slab by phase correlation: a coarse pass
with 2.56 mm blocks at the first two depths gives the pair's depth offset as a plane and its in-plane shift, a dense
pass with 1.28 mm blocks searches ±0.10 mm in depth around that plane. The in-plane peak is refined on an upsampled DFT
(Guizar-Sicairos et al., 2008) and the depth by a Gaussian through the correlation peaks of stacks of the older slab
interpolated at quarter planes. A match is kept when its correlation peak, the phase correlation averaged over the three
planes, exceeds 0.12 and lies inside the depth search. A pair is left out when fewer than 8 coarse blocks agree on its
depth offset within 0.04 mm, or when that offset does not lie between 0 and `deep_a_mm`.

**Placement.** Slab s maps its raw voxel (x, y, z) in mosaic tile k to the common frame by

$$(X, Y, Z) = (x,\ y,\ z) + a_s + f_s(x, y) + e_s[k],$$

with three parts, each acting in-plane and in depth: a translation of the slab a_s (stage step, motion of the block), a
smooth deformation f_s, which also carries the slab's tilt (a cubic B-spline with 2.56 mm control spacing, Rueckert et
al., 1999), and a translation e_s[k] of each mosaic tile (acquisition). Every match says that its two ends land on the
same point. All parts of all slabs are solved at once, in-plane and in depth as two problems with one design, by robust
least squares (Huber loss, 5 µm), with a group-L1 penalty on each tile translation (weight 0.0006 mm) and a fused-L1
penalty between neighbouring tiles of a slab (weight 0.015 mm, Tibshirani et al., 2005), since a stitching error moves
neighbouring tiles together, and a weak ridge on the B-spline, which holds the field at zero where no match reaches. The
matches fix the slabs only relative to each other, so the mean translation of the slabs is held at zero.

**Volume.** Every voxel is taken from the newest slab that covers it, whose cut lies at least `margin_mm` above it and
whose window reaches it (down to `deep_a_mm`), as that slab's raw mu_s resampled at the registered position (trilinear),
with the slab's cut and coverage read there bilinearly. Each slab is resampled tile by tile: a voxel takes the tile
whose moved cell holds it, the one it lies farthest inside where moved cells overlap, and no tile of that slab where
they leave a gap. The 40 µm volume is composed in the same way from the slabs' in-plane means over the raw voxels of
each 40 µm column, with the deepest cut of its raw columns and covered where all of them are, so that the margin and the
coverage hold for every raw column in the mean. There is no brightness correction, no blending at the seams and no
filling: a voxel no slab holds stays 0. A source map gives the slab of every voxel, −1 for none, plus 1000 below the
slab's reliable planes, which end at `deep_mm`, so slab numbers run from 0 to 999. The NIfTI files place the
common-frame point (X, Y, Z) at world (X, Y, −Z) mm, the third axis pointing up out of the block face ([Scope](#scope)).

**MRI reference.** A transform that all slabs share changes no match, so the matches leave it open. With `--mri`, octreg
([METHOD.md](METHOD.md)) finds its affine part, one affine of the registered volume to the block's MRI, as Wang et al.
(NeuroImage 100:395, 2014) and Costantini et al. (2023) register serial-sectioning OCT to MRI, and the rest of that
transform, a deformation all slabs share, stays in the volume. octreg reads a tissue mask and a levelled copy of the 40
µm volume, both on the volume's grid and world, so the affine holds for the volume. The mask replaces octreg's specimen
mask, which takes the larger part of the measured volume to be the specimen, and in a mosaic that can be the gel. log10
mu_s is smoothed over the reliable voxels alone (normalised Gaussian, σ 0.16 mm), since below the reliable planes gel
and tissue read alike, and the tissue is the measured voxels above Otsu's threshold of the smoothed map, with holes
filled in every array plane. Tissue mu_s changes with the depth under a cut and slabs differ in level, so the volume,
which changes slab under every cut, has an edge there. In the copy every voxel is divided by the median of the tissue of
its slab at its depth under that slab's cut, counted in planes from where the slab begins in its column. The matches
place the slabs more finely than an MRI voxel, so octreg works at the MRI's resolution. Its base grid is the largest
side of the MRI voxel, so that the last pass spans at least two voxels along every MRI axis. It is kept between octreg's
own fine and base grids, 0.04 and 0.15 mm, so an MRI of 0.15 mm or coarser is registered with octreg's own parameters.
The σ of its last pass, 0.3 mm, is two voxels of the 0.15 mm grid, so further passes divide σ by 1.5, the largest ratio
between octreg's own passes, and the last is at σ of two base voxels.

## Usage

```
python -m tilestack BLOCK_JSON OUT_DIR [--full] [--layout LATTICE_JSON] [--mri MRI]
```

`BLOCK_JSON` describes the block:

```json
{
 "slab_file": "/data/block/slab{:03d}.mat",
 "work": "/data/block/tilestack_work",
 "slabs": [1, 2, 3, 4],
 "voxel_mm": 0.02,
 "neighbours": 2,
 "margin_mm": 0.06,
 "deep_mm": 0.86,
 "deep_a_mm": 1.12
}
```

A slab is a MATLAB v7.3 file with one array `M` of mu_s per µm, read by h5py as (z, y, x), z growing with depth from
above the cut. `slab_file` is the path of a slab with a Python format field such as `{:03d}` for its number. `slabs`
lists the slabs in cut order, all with the same imaging window. `work` holds the caches of a run (the slabs at 40 µm,
cut surfaces, the tile lattice, matches), valid only for the slab files and settings they were made with. `voxel_mm` is
the side of the raw voxel, 0.04 mm divided by a whole number. `neighbours` is the number of slabs ahead that still share
tissue, the imaging window's depth against the cut step. `deep_mm` ends the reliable planes, where the estimate of a
uniform material such as the gel starts to rise towards the bottom of the window. `deep_a_mm` is the deepest plane at
which the older slab of a pair still matches. Both count from the first plane of the window. The tile lattice is read
from the staircase outline of the coverage masks when the mosaic has three or more tile rows and columns and is turned
enough that the outline steps by more than a pixel per tile. It is otherwise given with `--layout`, a JSON object with
X0, Y0, PX, PY, SX, SY, the tile width W and height H in pixels, rows and cols (the tile in row i and column j has its
top-left corner at x = X0 + j PX + i SX, y = Y0 + i PY + j SY). `--full` also writes the volume at the raw voxel.
`--mri` registers the volume to an MRI NIfTI cropped around the block, the input octreg takes. The matching, the
composition and octreg run on a CUDA GPU.

| file | content |
|---|---|
| `registered_40um.nii.gz`, `registered_40um_source.nii.gz` | the registered volume on a 40 µm grid in-plane and the raw planes in depth, and its source map |
| `registered_20um.nii`, `registered_20um_source.nii` | the same at the raw voxel (`--full`), named with `voxel_mm` in µm |
| `transforms.json` | the placement of every slab and tile in mm, with the formulas in its description |
| `solution.pkl` | the solver's output |
| `run.json` | the block, the settings, the pairs and the residuals of the matches |
| `mri/` | with `--mri`: octreg's outputs for the levelled copy, whose `T_oct2mri.txt`, `oct2mri.lta` and `oct2mri_itk.txt` hold for the registered volumes, which share its world. `levelled.nii.gz` and `tissue.nii.gz` are what octreg read, and `registered_40um_in_mri.nii.gz` is the volume's mu_s on the MRI grid |

## Scope

octreg takes the handedness from the header and searches proper rotations only. The world of the NIfTI files has the
specimen's handedness when a plane of a slab, shown with x (the first MATLAB dimension) down and y across, is the block
face seen from above, and slabs that show it mirrored need one in-plane axis reversed in their files. octreg's prior
holds the scales near 1, so the slabs should have the voxel `voxel_mm` along all three axes in the tissue, the depth
axis already converted from optical path with one refractive index for the block. The tissue mask takes the gel to be
darker than the tissue and a large part of the volume, as for a block in plain agarose imaged with a margin, and its σ
is half a cut step of about 0.3 mm, which damps the change with depth under a cut. The levelling takes tissue mu_s to
change with depth under the cut by the same factor across a slab.
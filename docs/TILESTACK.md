# tilestack: internal registration of serial-sectioning OCT slabs

tilestack registers the slabs of a serial-sectioning OCT block to each other and writes them as one volume. Each slab is
a block-face OCT volume of the scattering coefficient mu_s, stitched from a mosaic of tiles and imaged before the next cut.
The imaging window reaches deeper than the cut step, so neighbouring slabs image the same tissue. tilestack matches that
shared tissue, places every slab and every mosaic tile from the matches in one solve, and composes the registered volume
from the raw mu_s of the slabs at their registered positions, with no other change to the values.

## Method

**Grid.** The voxel of the slabs is read from their files, and x and y must have the same voxel. The cut surfaces, the
matches and the solve work on the working grid: the slabs averaged over the whole number of in-plane voxels nearest
20 µm, and over as many planes as give the same spacing in depth, which must be a whole number (20 µm for files at 20, 10
or 5 µm, or at 5 × 5 × 2.5 µm). The volume is written at the voxel of the slab files.

**Cut surface.** In every column of a slab the cut is the plane where log10 mu_s rises most sharply along depth: the
three planes from it down lie above the three planes over it, by as much as possible. A rise and not a level, because the
water above the cut reads brighter the less signal lies under it. A column more than 0.06 mm from the median of its
3 × 3 neighbourhood takes that median. The planes from `margin_mm` under the cut are used, below the surface reflection.

**Correspondences.** Each slab is matched to the next `neighbours` slabs. Blocks of the newer slab, 3 planes of log10
mu_s at 0.08 to 0.40 mm under its cut, are found in the older slab by phase correlation: a coarse pass with 2.56 mm blocks
gives the pair's depth offset as a plane and its in-plane shift, a dense pass with 1.28 mm blocks searches ±0.10 mm in
depth around that plane. The in-plane peak is refined on an upsampled DFT (Guizar-Sicairos et al., 2008) and the depth on
the older slab interpolated at quarter planes. Matches with a clear correlation peak inside the depth search are kept.
Two slabs share no usable tissue when fewer than 8 coarse blocks agree on their depth offset within 0.04 mm.

**Placement.** Slab s maps its raw voxel (x, y, z) in mosaic tile k to the common frame by

$$(X, Y, Z) = (x,\ y,\ z) + a_s + f_s(x, y) + e_s[k],$$

with three parts, each acting in-plane and in depth: a translation of the slab a_s (stage step, motion of the block), a
smooth deformation f_s, which also carries the slab's tilt (a cubic B-spline with 2.56 mm control spacing, Rueckert et
al., 1999), and a translation e_s[k] of each mosaic tile (acquisition). Every match says that its two ends land on the
same point. All parts of all slabs are solved at once by robust least squares (Huber loss), with a group-L1
penalty on each tile translation and a fused-L1 penalty between neighbouring tiles of a slab (Tibshirani et al., 2005),
since tile errors come in rows and columns, and a weak ridge on the B-spline. The matches fix the slabs only relative to
each other, so the common frame is the mean slab.

**Volume.** Every voxel takes the newest slab whose cut lies at least `margin_mm` above it and that holds it in its
window, down to `deep_a_mm`, and holds that slab's raw mu_s resampled at the registered position (trilinear). Each slab
is resampled tile by tile. There is no brightness correction, no blending at the seams and no filling: a voxel no slab
holds stays 0. A source map gives the slab of every voxel, plus 1000 where it lies below the reliable planes
(`deep_mm`), -1 where no slab holds it. The NIfTI files place the common-frame point (X, Y, Z) at world (X, Y, −Z) mm,
the third axis pointing up out of the block face, so the world has the specimen's handedness.

## Usage

```
python -m tilestack BLOCK_JSON OUT_DIR [--layout LATTICE_JSON]
```

`BLOCK_JSON` describes the block:

```json
{
 "slab_file": "/data/block/Mus_slice{:03d}.mat",
 "work": "/data/block/tilestack_work",
 "slabs": [5, 6, 7, 8],
 "neighbours": 2,
 "margin_mm": 0.06,
 "deep_mm": 0.86,
 "deep_a_mm": 1.12
}
```

A slab is a MATLAB v7.3 file with one array `M` of mu_s per µm, read by h5py as (z, y, x), z growing with depth from
above the cut, and `res_um`, its voxel in µm (x, y, depth). A plane of a slab shown with x (the first MATLAB dimension)
down and y across is the block face seen from above. `slabs` lists the slab numbers in cut order, all with the same
imaging window. `work` holds the caches of a run (the slabs on the working grid, cut surfaces, matches), which hold only
for the settings they were made with. `neighbours` is the number of slabs ahead that still share tissue, the imaging
window's depth against the cut step. `margin_mm` is the depth skipped under the cut, the surface reflection. `deep_mm`
ends the reliable planes, where the estimate of a uniform material such as the gel starts to rise towards the bottom of
the window, and `deep_a_mm` is the deepest plane at which the older slab of a pair still matches, both measured from the
first plane of `M`. The tile lattice is read from the staircase outline of the coverage masks when the outline steps by
a working pixel or more at every tile row and column, and must otherwise be given with `--layout`, a JSON with rows,
cols and, in pixels of the slab files counted from 0, the tile size W × H and the lattice x = X0 + j·PX + i·SX,
y = Y0 + i·PY + j·SY of the top-left corner of tile (i, j), row i along y and column j along x. The matching and the
composition run on a CUDA GPU. The composition holds one slab file on the GPU and the source map of the volume in memory.

| file | content |
|---|---|
| `registered_20um.nii.gz`, `registered_20um_source.nii.gz` | the registered volume at the voxel of the slab files, named by it (here 20 µm), and its source map |
| `warps/slab_NNN.nii.gz` | the placement of slab NNN as an ITK displacement field on the in-plane grid of the registered volume, in two planes since it does not change with depth: at each point the displacement to the point of the slab placed there, and between moved tiles that of the nearest tile |
| `run.json` | a hash of the code, the block, the number of pairs used, the pairs skipped and the residuals of the matches |

Voxel (i, j, k) of a slab file, i along x, lies at world (i·vx, j·vy, −k·vz) mm before the warp, (vx, vy, vz) being
`res_um` in mm. With that world in a NIfTI of the slab,
`antsApplyTransforms -i slab.nii -r registered_20um.nii.gz -t warps/slab_NNN.nii.gz -o slab_NNN.nii.gz` resamples the slab
onto the registered volume, and so does SimpleITK with the warp read as a 64-bit vector image.

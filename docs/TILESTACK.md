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

$$(X, Y, Z) = (x,\ y,\ z - \beta\, C_s(x, y, z)) + a_s + f_s(x, y) + e_s[k],$$

with three parts, each acting in-plane and in depth: a translation of the slab a_s (stage step, motion of the block), a
smooth deformation f_s, which also carries the slab's tilt (a cubic B-spline with 2.56 mm control spacing, Rueckert et
al., 1999), and a translation e_s[k] of each mosaic tile (acquisition). Every match says that its two ends land on the
same point. All parts of all slabs and β are solved at once by robust least squares (Huber loss), with a group-L1
penalty on each tile translation and a fused-L1 penalty between neighbouring tiles of a slab (Tibshirani et al., 2005),
since tile errors come in rows and columns, and a weak ridge on the B-spline. The matches fix the slabs only relative to
each other, so the common frame is the mean slab.

**Optical path.** OCT measures depth as optical path, and light is slower in denser, brighter tissue, so a voxel under
bright tissue lies deeper in the image than in the block. Its covariate C is the level of the material above it, log10
mu_s above the embedding gel's level L0, summed down its column from the cut, and β converts it to planes, one number for
the stack. The levels come from a first registration without the optical term, whose volume holds at every position the
slab that imaged it nearest its cut. L0 is the lowest mode of those levels, the gel's.

**Volume.** Every voxel takes the newest slab whose cut lies at least 0.06 mm above it and that holds it in its window,
down to `deep_a_mm`, and holds that slab's raw mu_s resampled at the registered position (trilinear). Each slab is
resampled tile by tile. There is no brightness correction, no blending at the seams and no filling: a voxel no slab holds
stays 0. A source map gives the slab of every voxel, plus 1000 where it lies below the reliable planes (`deep_mm`).

## Usage

```
python -m tilestack BLOCK_JSON OUT_DIR [--full] [--layout LATTICE_JSON] [--zero LOG10]
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

A slab is a MATLAB v7.3 file with one array `M` of mu_s per µm, read by h5py as (z, y, x), z growing with depth from above
the cut. `work` holds the caches of a run (cut surfaces, matches), which hold only for the settings they were made with.
`neighbours` is the number of slabs ahead that still share tissue, the imaging window's depth against the cut step.
`deep_mm` ends the reliable planes, where the estimate of a uniform material such as the gel starts to rise toward the
bottom of the window. `deep_a_mm` is the deepest plane at which the older slab of a pair still matches. The tile lattice is
read from the staircase outline of the coverage masks when the mosaic is rotated by about 0.4° or more, and is otherwise
given with `--layout` (X0, Y0, PX, PY, SX, SY, W, H in pixels, rows, cols). `--full` also writes the volume at the raw voxel,
`--zero` gives L0 instead of the lowest mode of the levels. The matching and the composition run on a CUDA GPU.

| file | content |
|---|---|
| `registered_40um.nii.gz`, `registered_40um_source.nii.gz` | the registered volume on a 40 µm grid in-plane and the raw planes in depth, and its source map |
| `registered_20um.nii`, `registered_20um_source.nii` | the same at the raw voxel (`--full`) |
| `transforms.json` | the placement of every slab and tile in mm, with the formulas in its description |
| `transforms_pass1.json`, `level_pass1_40um.nii.gz`, `cut_surface_40um.nii.gz` | the first registration, its levels and the cut surfaces, from which the covariate C is read |
| `solution.pkl`, `solution_pass1.pkl` | the solver's output |
| `run.json` | the block, the settings, the pairs, the residuals of the matches, β and L0 |

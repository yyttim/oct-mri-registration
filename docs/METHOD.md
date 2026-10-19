# octreg: method

octreg registers an OCT volume of an embedded tissue block to an ex-vivo MRI cropped around it. It finds an affine transform
from any starting orientation. The search (§3) and the refinement (§4) maximise one similarity F, the correlation of
gradient-orientation tensors of the two volumes (§2), inside the masks of §1. An outline term O proposes further poses in the
search. The tensor ignores the sign of an edge, so F needs neither an intensity mapping between OCT and MRI nor their contrast
polarity.

## Inputs and frames

The OCT is a scattering or intensity map of a block embedded in agarose, plain or doped. The medium is what surrounds the tissue
in either image, agarose in the OCT. The OCT has section stripes along one array axis, tile seams in the section plane and exact
zeros where data are missing. The MRI is an ex-vivo scan cropped around the block. The crop must hold the whole block, and it
may hold tissue that is not in the block, for example beyond a face where the block was cut out of a larger specimen. Both voxel
sizes are taken as correct, so the true scales are close to 1.

Transforms map millimetres in the world frame of the OCT file to that of the MRI file. The world frame is the NIfTI header frame
or, for TIFF and NPY, diag(spacing) on the axes (x, y, z) = numpy axes (2, 1, 0). The two volumes show the same specimen, so
they differ by a proper rotation and never by a mirror image. The file frames fix the handedness, and §3 searches proper
rotations only. The OCT is streamed plane by plane into a 0.04 mm fine grid, where the specimen mask is computed. Both volumes
then have a 0.15 mm base grid, for the OCT a box average of the fine grid (masks kept above 0.5), and the MRI foreground is
computed on it. The search of §3 runs on a 0.6 mm box average of the base grid.

## §1 Masks

**OCT specimen mask.** A voxel is specimen when it has signal and isotropic texture. Both are decided by one threshold rule,
each on its own histogram of 256 bins over the range of its values: the lower of Otsu's threshold and the foot of the tallest
peak on its low side (the triangle threshold of Zack et al., J. Histochem. Cytochem. 25:741, 1977, taken on the low side). With
two modes Otsu's threshold falls between them. With one mode it would split that mode, and the foot of that mode lies below
Otsu's threshold. Signal is a log intensity above the threshold of the measured voxels, the positive ones. Texture separates
doped agarose, which can be as bright as tissue, from tissue: section stripes vary only along the sectioning axis and tile seams
only across the seam, while tissue varies along every direction. On the signal voxels the OCT is smoothed (normalised Gaussian,
σ 0.08 mm), and the running coefficient of variation c_a is computed along each array axis over 0.36 mm where at least half of
that window is signal. C = min_a c_a is averaged onto a 0.16 mm texture grid, whose blocks count where at least half of their
voxels have a C, and log C is smoothed over those blocks (normalised Gaussian, σ 1.2 mm). The specimen is the signal voxels
where log C lies above its threshold, its components holding at least 1 % of it, with holes filled in every array plane: uniform
tissue such as a white-matter bundle leaves holes, and a hole that reaches a cut face is not enclosed in 3-D. Voxels at or below
zero are missing data and never specimen. A mask can be given as a file instead (`--oct-mask`).

**MRI foreground.** The candidate foreground of a histogram valley is the voxels above it, with holes filled and components
below 1 % of it dropped. The threshold is the first valley, walking down from the brightest peak, whose count is less than half
that of the smaller of its two peaks and whose darker class lies mostly outside the convex hull of the candidate foreground:
less than half of the voxels with values between the next lower valley (or zero) and this one lie inside that hull. Medium lies
outside the specimen, while a darker tissue class, such as white matter under cortex, lies inside the hull of the brighter one
even where the valley between them is deep. The histogram has 256 bins over the positive values up to their 99.5th percentile,
so that a few hot voxels cannot squeeze the tissue into a few bins, and its counts are averaged over 5 bins, so that an
integer-valued image gives no comb of empty bins. Its peaks are found on the square root of the counts, so that a narrow
background peak cannot hide the tissue peak. An empty bin is added at each end, so that a peak in an end bin counts. A peak
needs a prominence of at least 5 % of the largest square-root count and lies at least 5 bins from the next peak. Without such a
valley the foreground is every positive voxel, holes filled and small components dropped in the same way. A mask can be given as
a file instead (`--mri-mask`).

## §2 Similarity

**Orientation tensors.** A volume I with mask m is differentiated inside the mask by normalised convolution (Knutsson and
Westin, CVPR 1993), g = ∇[G_σ ∗ (I m) / G_σ ∗ m], so the mask edge adds no gradient, and g is taken to world coordinates. The
normalised gradient is n = g / √(|g|² + ε²), with ε the median |g| over the mask, and its tensor P(n) = n nᵀ is stored as six
channels, (xx, yy, zz) and √2 (xy, xz, yz), whose dot product is ⟨P(n), P(n′)⟩ = (n·n′)², the squared product of normalised
gradient fields (Haber and Modersitzki, Methods Inf. Med. 46:292, 2007). P(n) = P(−n), so an edge counts the same whichever side
is brighter.

**Structure similarity F.** For a pose T with linear part A, F runs over OCT points x_i, every second base voxel along each axis
that lies farther than 0.8 mm inside the specimen mask. The OCT tensor a_i is taken of the OCT gradient carried through the
pose, A⁻ᵀg, and the MRI tensor b_i = P(n_MRI)(T x_i) is sampled trilinearly, zero outside the MRI foreground and grid:

$$F(T) = \frac{\sum_i \langle a_i - \bar a,\; b_i - \bar b\rangle}{\sqrt{\sum_i |a_i - \bar a|^2 \,\sum_i |b_i - \bar b|^2}} .$$

F is taken over the block, not over the overlap of the two volumes. OCT tissue that lands on MRI background therefore lowers F,
while MRI tissue outside the block does not enter it.

**Outline O.** O is the correlation of the OCT specimen mask w with the MRI foreground M(T x) over the measured OCT voxels,
weighted by q (1 − (1 − w) E), with q the measured fraction and E = M inside the MRI crop and 1 outside. The variance of M under
that weight counts as at least 2 % of the variance of M over the crop, so O stays near 0 where M hardly varies under the block.
OCT tissue has to lie on MRI tissue, but medium over MRI tissue is no mismatch, since the crop can hold tissue beyond a face
where the block was cut out. The weight leaves that case out. O places a block by its shape where its internal structure is
weak, as for a whole piece of brainstem surrounded by medium. For a block cut out of a larger specimen O is about the same
wherever the block lies inside the MRI tissue, so O only proposes poses and F decides between them (§3, §4). O is computed when
both images show medium around the tissue: the measured OCT voxels outside the specimen mask, eroded by one base voxel, are at
least 1 % of the measured OCT voxels, and the MRI voxels outside the foreground, eroded by one base voxel, are at least 1 % of
the MRI grid.

## §3 Orientation search

The crop bounds the position, so the search is over all rotations and every translation inside the crop. The rotations are 8,000
points of the Super-Fibonacci spiral, a deterministic near-uniform set on SO(3) (Alexa, CVPR 2022). For each rotation R, the OCT
on the search grid (intensities, specimen mask, its erosion by 0.8 mm and measured fraction) is turned about its box centre and
resampled onto a cube of MRI voxels that holds its bounding sphere. Its tensors are computed after the turn, in the MRI axes,
with σ 1.2 mm, and F takes the form of §2 over the voxels of the turned erosion, except that the OCT gradients are taken inside
that erosion too: at this scale a margin of medium inside the specimen mask would add an edge that reaches the points. F and O
at every translation are masked correlations, and each of their sums is an FFT correlation of a template channel with a channel
of the MRI zero-padded beyond the crop, or with E (Padfield, IEEE Trans. Image Process. 21:2706, 2012). The six tensor channels
enter as one sum, as in Cross-Sim-NGF (Öfverstedt et al., WBIR 2022). Each rotation keeps its best translation by F and, when O
is computed, its best translation by S = (F + O) / 2. Greedy non-maximum suppression drops a pose when a pose already kept
places the OCT box centre within 3 mm of where it does and differs from it by less than 20° in rotation. The best 12 poses by S
and then the best by F up to 24 in all go to §4, or the best 24 by F when O is not computed.

## §4 Affine refinement

The pose model is

$$x_\text{MRI} = R(r)\, \mathrm{Sh}(h)\, \mathrm{diag}(e^{\ell})\, (x_\text{OCT} - c) + t ,$$

with rotation vector r, upper-triangular shears h, log-scales ℓ, translation t and c the OCT box centre. Every one of the 24
search poses is refined on the base grid by Adam on

$$L = 1 - F + \lambda \Big(\sum_i \ell_i^2 + \sum_i h_i^2\Big), \qquad \lambda = 2,$$

in one pass of 150 iterations at each of σ = 0.6, 0.4 and 0.3 mm, with a cosine schedule. The step is 0.06 mm of motion of the
OCT box corners: 0.06 mm for t and 0.06 mm / ρ for r, ℓ and h, ρ the radius of the box. The penalty is the prior that the voxel
sizes are right, since F alone can reward distorting the block. Of the poses whose final L lies within 10⁻⁵ of the lowest, the
one proposed first wins.

## Parameters

The parameters are fields of `octreg.params.Params` (override with `--params`).

| parameter | default | role |
|---|---|---|
| `search_mm`, `base_mm`, `fine_mm` | 0.6, 0.15, 0.04 mm | search grid, base grid, OCT grid of the specimen mask |
| `texture_bandpass_mm`, `texture_window_mm` | 0.08, 0.36 mm | §1: smoothing, window of the coefficient of variation |
| `texture_grid_mm`, `texture_smooth_mm` | 0.16, 1.2 mm | §1: texture grid, smoothing of log C |
| `valley_ratio`, `min_component` | 0.5, 0.01 | §1: MRI valley depth, smallest mask component kept |
| `erode_mm`, `outline_min_outside` | 0.8 mm, 0.01 | §2: depth inside the specimen mask of the OCT points (and in §3 of the region where their gradients are taken), medium share each image needs for O |
| `search_sigma_mm`, `n_rot`, `topk` | 1.2 mm, 8000, 24 | §3: gradient scale, rotations, poses refined |
| `nms_mm`, `nms_deg` | 3 mm, 20° | §3: poses closer in both count as one |
| `sigmas_mm`, `iters`, `lam`, `lr_mm` | 0.6, 0.4, 0.3 mm, 150, 2, 0.06 mm | §4: gradient scales of the passes, Adam iterations per pass, prior weight λ, step |

## Scope

Both thresholds of the OCT specimen mask take the tallest histogram mode to be the specimen's, so the specimen should be the
larger part of the measured volume, as in a block imaged with a margin of medium around it. A mask file replaces the specimen
mask otherwise (`--oct-mask`), and the OCT file itself serves when it is exactly zero outside the tissue, since a mask file is
read as its positive voxels. A TIFF or NPY read in its array frame can be the mirror image of the specimen, which no proper
rotation aligns: the array frame then needs one axis flipped (a NIfTI header with one negative spacing). A NIfTI without sform
or qform is read in its array frame, diag(pixdim), as ITK reads it. FreeSurfer reads such a file in another frame, so
`oct2mri.lta` applies only to NIfTI files with an sform or qform.

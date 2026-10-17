# octreg: method

octreg registers an OCT volume of an embedded tissue block to an ex-vivo MRI cropped around it. It finds an affine transform
from any starting orientation (§1 to §4) and adds a small smooth deformation on top (§5). The search, the refinement and the
deformation maximise one score, the correlation of gradient-orientation tensors of the two volumes (§2), the search together
with an outline term. The tensor ignores the sign of an edge, so the score needs neither an intensity mapping between OCT and
MRI nor their contrast polarity.

## Inputs and frames

The OCT is a scattering or intensity map of a block embedded in scatterer-doped agarose. It has section stripes along one array
axis, tile seams in the section plane and exact zeros where data are missing. The MRI is an ex-vivo scan cropped around the
block. The crop must hold the whole block, and it may hold tissue that is not in the block, for example beyond a face where the
block was cut out of a larger specimen. Both voxel sizes are taken as correct, so the true scales are close to 1.

Transforms map millimetres in the world frame of the OCT file to that of the MRI file. The world frame is the NIfTI header frame
or, for TIFF and NPY, diag(spacing) on the axes (x, y, z) = numpy axes (2, 1, 0). The two volumes show the same specimen, so
they differ by a proper rotation and never by a mirror image. The file frames fix the handedness, and §3 searches proper
rotations only. The OCT is streamed plane by plane into a 0.04 mm fine grid, where the specimen mask is computed. Both volumes
then have a 0.15 mm base grid, for the OCT a box average of the fine grid (masks kept above 0.5). The search of §3 runs on the
4³ box average of the base grid, a 0.6 mm grid.

## §1 Masks

**OCT specimen mask.** Doped agarose scatters about as strongly as tissue, but its structure is directional. A section stripe
varies only along the sectioning axis and a tile seam only across the seam, while tissue varies along every direction. The fine
OCT is smoothed (normalised Gaussian, σ 0.08 mm), the running coefficient of variation c_a is computed along each array axis
over 0.36 mm, and the texture field C = min_a c_a is averaged over 0.16 mm blocks. log C is smoothed (σ 1.2 mm), split by Otsu's
threshold, closed (0.48 mm) and reduced to its largest component. Uniform tissue such as a white-matter bundle leaves holes, and
a hole that reaches a cut face is not enclosed in 3-D, so holes are filled in every array plane. Voxels that are exactly zero
are missing data and never specimen.

**MRI foreground.** The threshold is the first histogram valley, walking down from the brightest peak, that lies below 0.5 of
its smaller neighbouring peak and whose darker peak lies below half the brightest peak, so that the medium and not a second
tissue class lies below it. A valley between two tissue peaks is passed over, and the run is flagged `mri_tissue_valley_passed`.
The histogram has 256 bins over the positive values below their 99.5th percentile, smoothed over 5 bins, and its peaks are found
on the square root of the counts. Without such a valley the threshold keeps nearly the whole field of view (the lower of the 1st
percentile and half the lower cut of a three-class Otsu threshold), and the run is flagged `mri_foreground_no_valley`. The mask
is closed with a ball of 2 voxels, its holes are filled and components below 1 % of its volume are dropped. Either mask can be
given as a file instead (`--oct-mask`, `--mri-mask`).

## §2 Score

**Orientation tensors.** A volume I with mask m is differentiated inside the mask by normalised convolution (Knutsson and
Westin, CVPR 1993), g = ∇[G_σ ∗ (I m) / G_σ ∗ m], so the mask edge adds no gradient, and g is taken to world coordinates. The
normalised gradient is n = g / √(|g|² + ε²), with ε the median |g| over the mask (Haber and Modersitzki, Methods Inf. Med.
46:292, 2007), and its trace-free tensor is

$$P(n) = n n^\top - \tfrac13 |n|^2 I, \qquad \langle P(n), P(n')\rangle = (n\cdot n')^2 - \tfrac13 |n|^2 |n'|^2 .$$

P(n) = P(−n), so an edge counts the same whichever side is brighter. The first term is the squared gradient product of
normalised gradient fields, and the second is its mean over unrelated, isotropic orientations, so unrelated edges add nothing on
average. P is stored as six channels, (xx, yy, zz) and √2 (xy, xz, yz), whose dot product is the tensor inner product.

**Structure score F.** For a pose T with linear part A, the score runs over OCT points x_i, every second base voxel along each
axis of the specimen mask eroded by round(0.8 mm / v) steps of the 6-neighbourhood on a grid of spacing v (5 steps on the base
grid). The OCT tensor a_i is taken of the OCT gradient carried through the pose, A⁻ᵀg, and the MRI tensor b_i = P(n_M)(T x_i) is
sampled trilinearly, zero outside the MRI foreground and grid:

$$F(T) = \frac{\sum_i \langle a_i - \bar a,\; b_i - \bar b\rangle}{\sqrt{\sum_i |a_i - \bar a|^2 \,\sum_i |b_i - \bar b|^2}} .$$

F is taken over the block, not over the overlap of the two volumes. OCT tissue that lands on MRI background therefore lowers F,
while MRI tissue outside the block does not enter it.

**Outline O.** O is the correlation of the OCT specimen mask w with the MRI foreground M(T x) over the measured OCT voxels,
weighted by q (1 − (1 − w) E), with q the measured fraction and E = M inside the MRI crop and 1 outside. OCT tissue has to lie
on MRI tissue, but embedding over MRI tissue is no mismatch, since the crop can hold tissue that is not in the block, beyond a
face where the block was cut out. The weight leaves that case out. O places the block where the internal structure is weak, and
it needs an outside in both images: at least 1 % of the measured OCT voxels outside the specimen mask, and of the MRI grid
outside the foreground, each eroded by one voxel. An OCT given as tissue only, or an MRI crop that is all tissue, has none. The
search maximises S = (F + O) / 2 when the outline is used and S = F otherwise (flag `outline_not_used`).

## §3 Orientation search

The crop bounds the position, so the search is over all rotations and every translation inside the crop. The rotations are the
identity and 7,999 points of the Super-Fibonacci spiral, a deterministic near-uniform set on SO(3) (Alexa, CVPR 2022). For each
rotation R, the OCT on the search grid (intensities, specimen mask and measured fraction) is turned about its box centre and
resampled onto a cube of MRI voxels that holds its bounding sphere. Its tensors are computed after the turn, in the MRI axes,
with σ 1.2 mm, and F takes the same form as in the refinement: OCT points of the specimen mask eroded in the same way (one step
on the search grid), MRI tensors inside the foreground and zero outside it. F and O at every translation are masked
correlations, and each of their sums is an FFT correlation of a template channel with a channel of the zero-padded MRI
(Padfield, IEEE Trans. Image Process. 21:2706, 2012). The six tensor channels enter as one sum, as in Cross-Sim-NGF (Öfverstedt
et al., WBIR 2022). Each rotation keeps its best translation by S. Greedy non-maximum suppression then drops a pose when a
better pose already kept lies within 3 mm and 20°, and the best 24 go to §4.

## §4 Affine refinement

The pose model is

$$x_\text{MRI} = R(r)\, \mathrm{Sh}(h)\, \mathrm{diag}(e^{\ell})\, (x_\text{OCT} - c) + t ,$$

with rotation vector r, upper-triangular shears h, log-scales ℓ, translation t and c the OCT box centre. Every one of the 24
search poses is refined on the base grid by Adam on

$$L = 1 - F + \lambda \Big(\sum_i \ell_i^2 + \sum_i h_i^2\Big), \qquad \lambda = 2,$$

in one pass of 150 iterations at each of σ = 0.6, 0.4 and 0.3 mm (learning rates 0.004 rad, 0.06 mm, 0.002 and 0.002 on a cosine
schedule), each pass keeping its lowest-L iterate, and the lowest final L wins. The penalty is the prior that the voxel sizes
are right, since F alone can reward distorting the block.

The search proposes and the finest scale decides. At the search scale the outline and the coarse structure place the block, but
they can prefer a wrong orientation: the specimen mask is a texture field smoothed over 1.2 mm, the MRI foreground a threshold,
and coarse gradients carry the intensity trends of both scans. At a few tenths of a millimetre the internal structure (fibre
bundles, vessels, nuclei) is specific, and the right pose, once within reach of the passes, ends with a lower L than the wrong
ones. The outline therefore enters the search only.

## §5 Smooth deformation

The specimen deforms slightly between the MRI scan and the OCT embedding, and sectioning adds small distortions. §5 fits a
displacement d on top of the affine. The OCT point x goes to y = T x + d(T x), with d trilinear on a lattice of 5 mm spacing
over the MRI grid, and its gradient goes through the linear part of the map, (I − ∇dᵀ) A⁻ᵀ g. The score is F of §2 with two
changes that let the tissue surface enter: the OCT points cover the specimen mask dilated by round(0.6 mm / v) steps of the
6-neighbourhood (4 on the base grid), with the OCT gradients taken over all measured voxels, and the MRI gradients are taken
over the whole crop. ε stays the median |g| over the specimen mask and over the MRI foreground, as in §2. The surface between
tissue and medium is then an edge in both volumes. The objective is

$$-F + w \sum \max\!\big(0,\; |\Delta d| / \delta - s\big)^2 ,$$

over every component Δd of the differences of d between neighbouring lattice nodes along the lattice axes, with δ = 5 mm the
lattice spacing, the strain bound s = 0.08 and w = 1000. Every entry of ∇d then stays near or below s, so d is a contraction for
s < 1/3 and id + d is invertible. L-BFGS with a strong-Wolfe line search (at most 200 iterations) runs one pass at σ 0.6 mm and
one at 0.3 mm. The objective never rises, and a lattice node that no OCT point reaches moves only when the strain bound pulls it
along behind a neighbour. The output is the pull-back field u on the MRI base grid, x + u(x) = (id + d)⁻¹(x), by fixed-point
iteration: the registered OCT at MRI point x is OCT(T⁻¹(x + u(x))). The affine stays the primary result, and every transform
file holds it alone.

## Parameters

The parameters are fields of `octreg.params.Params` (override with `--params`).

| parameter | default | role |
|---|---|---|
| `search_mm`, `base_mm`, `fine_mm` | 0.6, 0.15, 0.04 mm | search grid, base grid, OCT grid of the specimen mask |
| `texture_bandpass_mm`, `texture_window_mm`, `texture_grid_mm` | 0.08, 0.36, 0.16 mm | §1: smoothing, window of the coefficient of variation, blocks |
| `texture_smooth_mm`, `texture_close_mm` | 1.2, 0.48 mm | §1: smoothing of log C, closing radius |
| `valley_ratio`, `medium_fraction`, `min_component` | 0.5, 0.5, 0.01 | §1: MRI valley depth, medium below this share of the brightest peak, smallest component |
| `erode_mm`, `outline_min_outside` | 0.8 mm, 0.01 | §2: erosion of the OCT points (in 6-neighbourhood steps), outside share each image needs for the outline |
| `search_sigma_mm`, `n_rot`, `topk` | 1.2 mm, 8000, 24 | §3: gradient scale, rotations, poses refined |
| `nms_mm`, `nms_deg` | 3 mm, 20° | §3: poses closer in both count as one |
| `sigmas_mm`, `iters`, `lam` | 0.6, 0.4, 0.3 mm, 150, 2 | §4: gradient scales of the passes, Adam iterations per pass, prior weight λ |
| `lr_rot`, `lr_t`, `lr_ls`, `lr_sh` | 0.004 rad, 0.06 mm, 0.002, 0.002 | §4: Adam learning rates |
| `df_grid_mm`, `df_max_strain`, `df_band_mm` | 5 mm, 0.08, 0.6 mm | §5: lattice spacing, strain bound, dilation of the OCT points (in 6-neighbourhood steps) |
| `df_sigmas_mm`, `df_iters` | 0.6, 0.3 mm, 200 | §5: gradient scales, most L-BFGS iterations per pass |

## Scope

The specimen mask of §1 needs the agarose embedding: without it the threshold splits the tissue itself. A block without
embedding needs its specimen mask given with `--oct-mask`. The OCT file itself serves when it is exactly zero outside the
tissue, since a mask file is read as its positive voxels.

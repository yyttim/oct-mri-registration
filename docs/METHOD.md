# octreg: method and evaluation

octreg registers an OCT volume of an embedded tissue block to an ex-vivo MRI cropped around it. §1 to §5 find an affine
transform, and §6 adds a small smooth deformation on top of it.

## Inputs and assumptions

The OCT is a scattering or intensity map of a block embedded in scatterer-doped agarose. It has section stripes along one array
axis, tile seams in the section plane and exact zeros where data are missing, and its contrast may be inverted relative to the
MRI. The MRI is an ex-vivo scan roughly cropped around the block. The crop should hold the whole block. It is the only position
prior, and it may hold tissue that is not in the block, for example beyond a face where the block was cut out of a larger
specimen. Both voxel sizes are taken as correct, so the true scales are close to 1. Inside the tissue the MRI must show the
OCT's two classes (§2), or the crop must be tight enough for the outline to fix the pose. In ex-vivo FLASH the grey/white
contrast depends on the flip angle and can vanish or invert between two angles, so the contrast given must show it. The MRI
medium must be darker than half the brightest tissue and the intensities free of an offset, otherwise its foreground is given
with `--mri-mask`.

The OCT and the MRI show the same specimen, so they differ by a proper rotation and never by a mirror image. The file frames fix
the handedness, and §3 searches proper rotations only. A TIFF or NPY stack, or a NIfTI file without sform and qform, is taken in
its array frame, which must then have the handedness of the specimen. The score of §2 cannot tell mirror images apart on a
nearly symmetric block, so it does not reveal a frame of the wrong handedness.

Transforms map millimetres in the world frame of the OCT file to that of the MRI file. The world frame is the NIfTI header frame
or, for TIFF and NPY, diag(spacing) on the axes (x, y, z) = numpy axes (2, 1, 0). The OCT is streamed plane by plane into a
0.04 mm fine grid, where the specimen mask is computed. Both volumes have a 0.15 mm base grid, for the OCT a box average of the
fine grid (masks kept above 0.5). All later steps use the base grid, except the search of §3, which runs on its 4³ box average,
a 0.6 mm grid.

## §1 Specimen mask

Doped agarose scatters about as strongly as tissue, but its structure is directional. A section stripe varies only along the
sectioning axis and a tile seam only across the seam, while tissue varies along every direction. The fine OCT is smoothed
(normalised Gaussian, σ 0.08 mm), the running coefficient of variation c_a is computed along each array axis over 0.36 mm, and
the texture field C = min_a c_a is averaged over 0.16 mm blocks. log C is smoothed (σ 1.2 mm), split by Otsu's threshold, closed
(0.48 mm) and reduced to its largest component. Uniform tissue such as a white-matter bundle leaves holes, and a hole that
reaches a cut face is not enclosed in 3-D, so holes are filled in every array plane. The rule needs the agarose embedding (see
Scope).

The MRI foreground threshold is the first histogram valley, walking down from the brightest peak, that lies below 0.5 of its
smaller neighbouring peak and whose darker peak lies below half the brightest peak (`medium_fraction`). The histogram has 256
bins over the positive values below their 99.5th percentile and is smoothed over 5 bins, and its peaks are found on the square
root of the counts, each with a prominence of at least 5 % of the largest and at least 5 bins from the next. The second
condition assumes that the medium lies below half the tissue intensity and that two tissue classes of one specimen differ by
less. The value 0.5 is the default fractional threshold of BET, which puts the brain edge halfway between a robust background
level and the local tissue intensity (Smith, Hum. Brain Mapp. 17:143, 2002). A deep valley between grey and white matter in a
crop without medium is then passed over, and the run is flagged `mri_tissue_valley_passed`. Without such a valley the threshold
is the lower of the 1st percentile and half the lower cut of a three-class Otsu threshold, which keeps nearly the whole field of
view, and the run is flagged `mri_foreground_no_valley`. The mask is closed with a ball of 2 voxels, its holes are filled in 3-D
and components below `min_component` of the foreground volume are dropped. Either mask can be supplied as a file and then
replaces the rule. A supplied OCT mask is still restricted to the measured voxels.

## §2 Two-class maps and outline

OCT scattering and MRI intensity follow no fixed mapping, but in both the tissue falls into a brighter and a darker class. Each
volume I with foreground M is flattened: it is divided by its local foreground mean G(I M) / G(M) (Gaussian σ 10 mm) to give x.
x is blurred by one voxel as G(xM) / G(M), so that background and masked-out gaps do not bleed into tissue. With t the Otsu
threshold and s the standard deviation of the foreground values of x clipped at their 99.5th percentile, the two-class map is
p = sigmoid((x − t) / (0.25 s)).

The OCT enters as channels u = (p_O, 1 − p_O) with its specimen mask w as weight, the MRI as v = (p_M M, (1 − p_M) M). The class
structure alone is a weak signal, so the score also compares the outlines, w against M, over the measured OCT voxels q. The
outline is one-sided. OCT tissue has to lie on MRI tissue, but agarose over MRI tissue is no mismatch, since the crop can hold
tissue that is not in the block. The weight leaves that case out: q' = q (1 − (1 − w) E), with E = M inside the crop and 1
outside it.

    S_class = ½ [NCC_w(u_1, v_1) + NCC_w(u_2, v_2)],   S_outline = NCC_q'(w, M),   S = (2 |S_class| + S_outline) / 3.

The outline compares what lies outside the tissue in both images, so it is used only when both have a measured outside: OCT
voxels outside the specimen mask and MRI voxels outside the foreground, each eroded by one base voxel to remove the resampling
shell of a mask, must make up at least 1 % of the measured OCT and of the MRI grid. An OCT given as tissue only (zero outside
it, and its own `--oct-mask`) or an MRI crop that is all tissue has no outside. S_outline is then 0 in §3 and §4 and the run is
flagged `outline_not_used`.

S is the mean of the three correlations and has no weight to set. Without the outline S = 2 |S_class| / 3, so the prior of §4
weighs 1.5 times more against the data. Weighted covariance is linear and ignores constants, so NCC_w(1 − u, v) = −NCC_w(u, v).
Swapping the OCT classes negates S_class exactly and leaves S_outline unchanged. One correlation therefore scores both
contrasts, and the sign of S_class is the polarity (−1: inverted). The identity needs the OCT channels to sum to 1 wherever
w > 0, which is why the mask is a weight and never multiplies them.

## §3 Orientation search

The crop bounds the position, so no global search is needed. For each of 8,000 rotations sampled uniformly on SO(3), the OCT
template is rotated about its box centre and S is computed with FFTs over every translation that puts that centre inside the
crop, in a weighted form of the masked correlation of Padfield (IEEE Trans. Image Process. 21:2706, 2012). Since q' = q − b E
with b = q (1 − w), every sum of the outline NCC is a correlation of a template with E, M, M² or M³. Outside the crop, OCT
tissue counts as lying over background and agarose is left out. Each rotation keeps its best translation. Greedy non-maximum
suppression then drops a pose when a better pose already kept lies within 3 mm and 20°, and the best 24 that remain go to §4.
The 20° is of the order of the critical distance of multi-level single linkage (Rinnooy Kan and Timmer, Math. Program. 39:57,
1987) for 8,000 rotations of SO(3).

## §4 Affine refinement

All 24 poses are refined on the base grid with their polarity fixed, since the search score does not settle their order:

    x_MRI = R(r) Sh(h) diag(exp(ℓ)) (x_OCT − c) + t,   L = 1 − S + λ (Σ ℓ_i² + Σ h_i²),   λ = 2,

with rotation vector r, shears h, log-scales ℓ and c the OCT box centre (Adam, 200 iterations, cosine schedule). The lowest L
wins. The penalty is the prior, since S alone rewards distorting the block. Each ℓ_i and h_i is also clamped to ±0.15, and a run
that reaches the clamp is flagged `clamp_saturated`.

## §5 Fine-structure refinement

The score of §2 finds the block, but an outline that carries a margin of agarose, or a cut face, leaves the last millimetres
open. The best pose of §4 is therefore refined on the internal structure of both volumes, such as fibre bundles, vessels and
nuclei, compared through the orientation of their gradients. Each volume I with mask m is smoothed and differentiated inside
the mask by normalised convolution, g = ∇(G_σ(I m) / G_σ(m)), so the mask edge adds no gradient. The normalised gradient (Haber
and Modersitzki, Methods Inf. Med. 46:292, 2007) is n = g / √(|g|² + ε²), with ε the median |g| of each volume, over the MRI
points used for the MRI and over the OCT specimen mask for the OCT. Both masks are eroded by five steps of the 6-neighbourhood
on the base grid (0.8 mm rounded to base voxels), which removes 0.75 mm along the array axes and 0.43 mm along their diagonals,
so the outline counts mainly in §2 to §4. Over every second interior MRI voxel along each axis, with the eroded OCT mask w,

    a(x) = P(n_O(x)),  b(x) = P(n_M(x)),  F(T) = Σ_x w ⟨a − ā, b − b̄⟩ / √(Σ_x w |a − ā|² · Σ_x w |b − b̄|²),   g_O(x) = A⁻ᵀ g_OCT(T⁻¹x),

with w = w(T⁻¹x), ā and b̄ the w-weighted means and A the linear part of T. P(n) = n nᵀ − |n|² I / 3 is the trace-free part of
the outer product n nᵀ, by which Knutsson (Proc. 6th Scand. Conf. Image Anal., 244, 1989) represents orientation, so that n and
−n give the same tensor. Then ⟨P(n), P(n′)⟩ = (n · n′)² − |n|² |n′|² / 3 compares the orientation of edges weighted by |n|²
|n′|². An inverted contrast flips the sign of a gradient and the tensor removes it, so F needs neither an intensity mapping nor
the polarity. F is the w-weighted correlation of the two tensor fields, the score of the §6 block matching taken over the whole
interior. The trace-free part leaves out the term |n|² |n′|² / 3, centring leaves out the agreement of the mean orientations,
and normalising makes F a correlation. L = 1 − F + λ (Σ ℓ_i² + Σ h_i²) is minimised with Adam as in §4, from the §4 pose and
under the same prior (λ = 2), with a fifth of the learning rates of §4, in one pass of 150 iterations at each of σ = 0.6, 0.4
and 0.3 mm. Each pass keeps its lowest-L iterate. At these scales section stripes a few tenths of a millimetre apart are
smoothed away.

## §6 Smooth deformation

The specimen deforms slightly between the MRI scan and the OCT embedding, and serial sectioning adds small distortions. After
the affine, local misfits of a fraction of a millimetre remain. §6 fits a small smooth displacement field on top of the affine.
The affine is the primary result. Every transform file holds it alone, and the field is written next to them.

The field u is a pull-back on the MRI base grid, in mm along the MRI world axes. The registered OCT at the MRI point x is
OCT(T⁻¹(x + u(x))). The OCT and its specimen mask are resampled onto the MRI base grid through T (mask kept above 0.99), and two
kinds of label-free evidence are measured there, interior matches and surface-edge offsets. Both have one reach, 1.35 mm, so §6
does not bring back tissue that lies farther than that from its MRI position. The masks of §1 do not place the surface to a
fraction of a millimetre, since the specimen mask is smoothed at σ 1.2 mm and the MRI foreground is a threshold. §6 therefore
reads the surface from the intensity of each volume.

Interior matches. MRI blocks of 4.5 mm, every 1.5 mm, with at least 70 % of the block inside both masks eroded by four steps of
the 6-neighbourhood (0.6 mm, 0.35 to 0.60 mm deep depending on the surface), are searched in the OCT within ±1.35 mm per axis.
The score is the zero-mean normalised cross-correlation of the trace-free part of n nᵀ (six channels, `octreg.blockmatch`),
with n = g / √(|g|² + ε²) as in §5 at σ 0.24 mm, ε the median |g| over the whole mask of each volume. It is the local
counterpart of F and also needs neither an intensity mapping nor the polarity. A match counts when its peak stands 4 standard
deviations above the mean of its score map, off the border of the range, and its length is at most 1.35 mm. Each match gives a
displacement at its block centre.

Surface-edge offsets. Up to 50,000 outermost voxels of the MRI foreground are sampled as boundary points, with outward normals ν
from the gradient of the smoothed signed distance to the surface (σ 2.5 voxels). Along each normal the intensity of each volume
is read from −2.1 to +2.1 mm, the reach and 0.75 mm beyond it, in base-grid steps, and differentiated with a Gaussian of
σ 1 sample. The surface edge is the steepest fall of the intensity going outwards. This assumes that in both volumes the tissue
at the surface is brighter than what surrounds it. In the MRI the tissue falls into background. In the OCT the surface carries a
bright rim, so the intensity falls going outwards into the agarose even where the bulk scatters alike, whatever the polarity of
the interior classes. A prominent fall is a local maximum of the fall more than 5 MADs above its median along the profile and
within ±1.35 mm, located to sub-sample precision by a parabola, which can carry it up to half a step beyond the reach. The
offset δ is the OCT edge minus the MRI edge along ν, and it constrains ν · u. A boundary point is used only when at least 95 %
of its OCT profile is measured and each profile has exactly one prominent fall. Faces crossed by section stripes show several
falls and mostly drop out, so the sectioning axis need not be known.

Model. Displacements c on a control lattice of 5 mm spacing over the bounding box of the MRI grid, trilinear in between, are
fitted by regularised least squares. Each interior match gives three rows and each boundary point one row ν · u = δ, weighted by
√(3N / M) for N matches and M points so that both kinds carry the same total weight. A membrane penalty with weight λ acts on
the first differences of c along the lattice axes, and a small ridge penalty keeps lattice nodes without evidence determined.
The system is solved four times with Huber re-weighting at 0.3 mm on both kinds of residual.

Model choice. Only λ is chosen, and the strain limit sets it. The strain, the largest first difference of c divided by the
spacing, must stay below 0.15 for the field to remain plausible for fixed tissue. The strain falls as λ grows, so λ is the
smallest weight between 0.3 and 30 that keeps the limit, found by bisection in log λ to a factor of 1.05. This gives the most
flexible field the limit allows. The field is then accepted only when it predicts evidence it was not fitted to. Both kinds of
evidence are split into four spatial folds (cells of 7 mm, with no buffer, so the test measures interpolation). At the chosen λ
the field is fitted without one fold and tested on it, for each of the four folds. The held-out median error of the interior
matches plus that of the surface-edge offsets, over all folds, must fall below 0.9 times the same score without deformation. A
field is also refused with fewer than 100 interior matches or 300 boundary points, or when no weight up to 30 keeps the limit.
Whenever a field is refused, the run is flagged `deformation_not_supported` and no field is written.

The evidence is measured and the field fitted once. Both kinds of evidence are then measured again on the warped OCT, the
interior matches over every confident match before the length cut and the surface-edge offsets over every boundary point with
one edge. These residuals are the read-outs of the stage.

The field is written as oct2mri_warp.nii.gz, the read-outs go to result.json under `deform`, and [README](../README.md#outputs)
lists the outputs. The field is not inverted, so the MRI to OCT direction is affine only.

## Evaluation

A registration is judged by looking at its QC images ([README](../README.md#outputs)). The test on block I46 of Costantini et
al. (2023) compares octreg and eight standard tools with the authors' registration ([bench/i46](../bench/i46/README.md)). The
valley rule of §1, the outline rule of §2, the candidate suppression of §3 and the score of §5 were designed with runs on these
data, so I46 is not a held-out test.

### I58 brainstem pair

The pair has no reference registration, so its result is shown and not scored. The
[README](../README.md#results-on-the-i58-brainstem-block) shows it on three planes in freeview and compares it with eight
affine and five deformable standard tools. [bench/baselines/BASELINES.md](../bench/baselines/BASELINES.md) shows every tool on
all three planes, and [bench/BENCHMARK.md](../bench/BENCHMARK.md) gives the values of the run.

### Ablations

Each variant removes or replaces one element of the method and keeps the rest (`bench/ablate.py` for §1 to §5,
`bench/ablate_deform.py` for §6). The [README](../README.md#ablations) shows the main variants in freeview.
[bench/BENCHMARK.md](../bench/BENCHMARK.md) lists every variant of §1 to §5 with its own values and the start test from turned
orientations, and [bench/results/I58/deform_ablation.md](../bench/results/I58/deform_ablation.md) every variant of §6.

## Parameters

The constants of the method are fields of `octreg.params.Params` (override with `--params`). The I58 pair uses the defaults,
whose hash is `9148790ad6c39454`. The first ten rows state what the method assumes about the data, and the rest set how a step
is computed.

§3 to §6 also hold guards and fixed rules at the top of their modules, outside Params, and the fixed rules of §1 and §2 are
given in their sections. In §3 the local variance of each MRI channel under the template is floored at 0.02 times its variance
over the MRI foreground (for M over the whole grid). In §4 and §5 these are the clamp of ±0.15 on log-scales and shears, and §5
uses a fifth of the learning rates of §4. In §6 these are the floor of 1e-3 per voxel on the OCT feature variance of a block
window, the range 0.3 to 30 of λ and the factor of 1.05 to which it is found, the 10 % held-out gain, the least evidence of 100
interior matches and 300 boundary points, the 0.99 above which the resampled OCT mask is kept, the 0.75 mm a boundary profile
runs beyond the reach, the σ of 1 sample of its derivative, the 95 % of an OCT profile that must be measured, the limit of
50,000 boundary points drawn with seed 0, the σ of 2.5 voxels on the signed distance, the four folds of 7 mm cells, where the
cell with index k goes to fold (k · (1, 3, 5)) mod 4, the ridge weight 1e-2 on the lattice displacements, the four Huber rounds
and the 70 % of a block that must lie inside both masks.

| parameter | default | role |
|---|---|---|
| `texture_bandpass_mm`, `texture_window_mm` | 0.08, 0.36 mm | §1: smoothing before, and window of, the directional coefficient of variation |
| `texture_smooth_mm`, `texture_close_mm` | 1.2, 0.48 mm | §1: smoothing of log C, closing radius |
| `flatten_sigma_mm` | 10 mm | §2: scale of the local foreground mean that flattens the intensities |
| `medium_fraction` | 0.5 | §1: the darker peak of an MRI foreground valley lies below this fraction of the brightest peak (medium, not tissue) |
| `lam` | 2.0 | §4 and §5: weight λ of the scale and shear prior |
| `ngf_sigmas_mm`, `ngf_erode_mm` | 0.6, 0.4, 0.3 mm, and 0.8 mm | §5: gradient scales of the passes, erosion of both masks (steps of the 6-neighbourhood) |
| `df_sigma_mm`, `df_block_mm` | 0.24, 4.5 mm | §6: gradient scale of the interior matches, edge of the matched blocks |
| `df_reach_mm` | 1.35 mm | §6: reach of the evidence, search range of the interior matches and largest distance of a surface edge from the MRI surface |
| `df_erode_mm` | 0.6 mm | §6: erosion of both masks that keeps the blocks inside (steps of the 6-neighbourhood) |
| `df_grid_mm`, `df_max_strain` | 5 mm, 0.15 | §6: spacing of the control lattice, strain limit |
| `search_mm`, `base_mm`, `fine_mm` | 0.6, 0.15, 0.04 mm | search grid, base grid, OCT grid of the specimen mask |
| `valley_ratio`, `min_component` | 0.5, 0.01 | MRI histogram valley / smaller peak, and the smallest MRI foreground component kept, as a fraction of the foreground |
| `texture_grid_mm`, `sigmoid_std` | 0.16 mm, 0.25 | texture blocks, and the sigmoid width in foreground standard deviations |
| `n_rot`, `seed`, `topk` | 8000, 0, 24 | rotations, rotation set seed, poses refined |
| `nms_mm`, `nms_deg` | 3 mm, 20° | poses closer in both count as one |
| `outline_min_outside` | 0.01 | §2: share of measured voxels outside the tissue each image needs for the outline to be used |
| `iters`, `lr_rot`, `lr_t`, `lr_ls`, `lr_sh` | 200, and 0.02 rad, 0.3 mm, 0.01, 0.01 | §4: Adam iterations and learning rates (§5: `ngf_iters` 150 per pass, a fifth of the rates) |
| `df_step_mm`, `df_z_min` | 1.5 mm, 4 | §6: grid step of the block centres, standard deviations above the mean of its score map a match needs |
| `df_edge_mad`, `df_huber_mm` | 5, 0.3 mm | §6: prominence of a fall in MADs above the median of the profile derivative, Huber threshold of both residuals |

## Scope

octreg was developed on one brainstem block, I58, which is unpublished, and tested on block I46 (see Evaluation). The specimen
mask of §1 needs the agarose embedding. Without it there are not two classes, and the threshold splits the tissue itself, more
so where section stripes make tissue quiet along one axis. A block without embedding needs its specimen mask given with
`--oct-mask`. The OCT file itself can serve when it is exactly zero outside the tissue, since a mask file is read as its
positive voxels.

# octreg 2.0: method

## Data and assumptions

The OCT is a scattering or intensity map of a tissue block embedded in scatterer-doped agarose, with section stripes along one
array axis, tile seams in the section plane and exact zeros where data are missing. Its contrast may be inverted relative to the
MRI. The MRI is an ex-vivo scan roughly cropped to a region containing the block. The crop is larger than the block and may hold
tissue that is not in it, for example beyond a face where the block was cut out of a larger specimen. It is the only position
prior. Both voxel sizes are taken as correct, so the true scales are close to 1, and orientation headers, where the files have
them, as having the correct handedness.

Transforms map millimetres in the OCT file world to the MRI file world: the NIfTI header frame, or for TIFF and NPY
diag(spacing) on the axes (x, y, z) = numpy axes (2, 1, 0). The OCT is streamed plane by plane into a 0.04 mm fine grid, where
the specimen mask is computed. Box averages of it give a 0.15 mm base grid (masks kept above 0.5), on which everything else
runs, and oct_in_mri.nii.gz. The search runs on the 4³ box average of the base grid, a 0.6 mm grid.

## 1. Specimen mask from isotropic texture

Doped agarose scatters about as strongly as tissue, but its structure is directional: a section stripe varies only along the
sectioning axis and a tile seam only across the seam. Tissue varies along every direction. The fine OCT is smoothed (normalised
Gaussian, σ 0.08 mm), the running coefficient of variation c_a is computed along each array axis over 0.36 mm, and the texture
field F = min_a c_a is averaged over 0.16 mm blocks. log F is smoothed (σ 1.2 mm) and split by Otsu's threshold, followed by
closing (0.48 mm) and the largest component. Uniform tissue such as a white-matter bundle leaves holes, and a hole that reaches
a cut face is not enclosed in 3-D, so holes are filled in every array plane. On I58 the mask measures 19.24 cm³ on the base
grid, while an intensity threshold on the same OCT takes in the agarose and gives 29.52 cm³ (both after the mask is intersected with
the measured OCT, as bench/BENCHMARK.md reports them).

The rule needs an embedding to separate. Without one there are not two classes, and the threshold splits the tissue itself,
the more so when section stripes are strong, since a stripe makes tissue quiet along one axis and so look like embedding. A
block that is not embedded should be given its mask with `--oct-mask`, and the OCT file itself serves, since a mask file is
read as its positive voxels.

The MRI foreground threshold is the first histogram valley, walking down from the brightest peak, below 0.5 of its smaller
neighbouring peak (peaks found on the square root of the smoothed counts). Without one, nearly the whole field of view is kept
and the run is flagged. The mask above the threshold is then closed with a ball of 2 voxels, its holes are filled in 3-D and
components below min_component of the foreground volume are dropped. Either mask can be supplied as a file
instead, and a supplied mask is used as given.

## 2. One score for structure and outline, polarity as a sign

OCT scattering and MRI intensity follow no fixed mapping, but in both the tissue falls into a brighter and a darker class. Each
volume I with foreground M is divided by its local foreground mean G(I M) / G(M) (Gaussian σ 10 mm) to give x, then blurred by
one voxel as G(xM)/G(M), normalised so background and masked-out gaps do not bleed into tissue. With t the Otsu threshold
and s the standard deviation of the foreground values of x clipped at their 99.5th percentile,
p = sigmoid((x − t) / (0.25 s)).

The OCT enters as channels u = (p_O, 1 − p_O) with its specimen mask w as weight, the MRI as v = (p_M M, (1 − p_M) M). The class
structure alone is a weak signal on real blocks, so the score also compares the outlines, w against M over the measured OCT
voxels q. OCT tissue has to lie on MRI tissue and agarose may lie on MRI background, but agarose over MRI tissue is no mismatch,
since the MRI can hold tissue that is not in the block. The outline weight therefore leaves that case out: q' = q (1 − (1 − w)
E), with E = M inside the crop and 1 outside it.

    S_class = ½ [NCC_w(u_1, v_1) + NCC_w(u_2, v_2)],   S_outline = NCC_q'(w, M),   S = (2 |S_class| + S_outline) / 3.

S is the mean of the three correlations, the two class channels and the outline, with no weight to set.
Weighted covariance is linear and ignores constants, so NCC_w(1 − u, v) = −NCC_w(u, v). Swapping the OCT classes therefore
negates S_class exactly and leaves S_outline unchanged: one correlation scores both contrasts, and the sign of S_class is the
polarity (−1: inverted). The identity needs the OCT channels to sum to 1 wherever w > 0, which is why the mask is a weight and
never multiplies them.

## 3. Orientation search in the crop

The crop bounds the position, so no global search is needed: 8,000 rotations uniform on SO(3) are sampled, and for each the OCT
template is rotated about its box centre and S is computed over all translations with FFTs in the masked form of Padfield (IEEE
Trans. Image Process. 21:2706, 2012). With b = q (1 − w), every sum of the outline NCC is a correlation of a template with E, M,
M² or M³. Outside the crop, OCT tissue counts as over background (the crop contains the block) and agarose is left out. Each
rotation keeps its best translation, and non-maximum suppression (3 mm, 10°) leaves 24 poses.

## 4. Prior-bounded affine refinement

Each of the 24 poses is refined on the base grid with its polarity fixed:

    x_MRI = R(r) Sh(h) diag(exp(ℓ)) (x_OCT − c) + t,   L = 1 − S + λ (Σ ℓ_i² + Σ h_i²),   λ = 2,

with rotation vector r, shears h, log-scales ℓ and c the OCT box centre. Adam, 200 iterations, cosine schedule, each ℓ_i and h_i
clamped to ±0.15. The lowest L wins. All of the prior is the penalty, which is needed because S alone rewards distorting the
block. The clamp is a guard, not a parameter: with the penalty it has never been reached, and a run that reaches it says so
with the flag `clamp_saturated` rather than returning a distorted block quietly. The two are ablated apart below.

All 24 poses are refined because the search score does not settle the order. On I58 the winner after §4 is the first of
the 24 by search score (`search_rank` in result.json), but with the maps of releases 1.0 and 1.1, which end at the same
affine as 2.0, it was the third. The refinement is also where the time goes, 345 s of the run's 685 s against 33 s for
the search.

## 5. Fine-structure refinement

The score of §2 compares large bright and dark regions and the outline. It finds the block, but it is dominated by the outline,
and an outline that carries a margin of agarose, or a cut face, leaves the last millimetres open. The best pose of §4 is
therefore refined on the internal structure of both volumes, fibre bundles, vessels and nuclei, compared by normalised gradient
fields (Haber and Modersitzki, 2006). On its base grid each volume I with mask m is smoothed and differentiated inside the mask
by normalised convolution, g = ∇(G_σ(I m) / G_σ(m)), so the mask edge adds no gradient, and n = g / √(|g|² + ε²), with ε the
median |g| over the MRI points used and over the OCT specimen mask. Both masks are eroded by 0.8 mm, which leaves the outline to
§2-4. Over every second interior MRI voxel along each axis, one in eight, with the eroded OCT mask w,

    F(T) = Σ_x w(T⁻¹x) (n_O(x) · n_M(x))² / Σ_x w(T⁻¹x),   g_O(x) = A⁻ᵀ g_OCT(T⁻¹x),

A the linear part of T. A squared cosine needs neither an intensity mapping nor the polarity. L = 1 − F + λ (Σ ℓ_i² + Σ h_i²) is
minimised as in §4 from the §4 pose, under the same prior (λ = 2), with learning rates a fifth of those of §4, one pass of 150
iterations at each of σ = 0.6, 0.4 and 0.3 mm, at which section stripes a few tenths of a millimetre apart are smoothed away.
§5 refines and does not search: F alone does not find the block (among the 24 poses of §4 on I58 with the maps of release 1.0, one 41 mm away
reached a higher F after refinement than the result), and it can lower S, whose outline and tissue classes are coarse.

Handedness. Two physical specimens are never mirror images, so the file frames fix the handedness and §3 searches proper
rotations only. The score of §2 cannot tell mirror images apart on a nearly symmetric block (on I58 the best mirrored pose has
the lower L and ends 30 mm from the result, the mirrored row of the ablation table below). A NIfTI header gives the frame. A TIFF or NPY stack, or a NIfTI file
without sform and qform, is taken in its array frame, which must therefore have the handedness of the physical specimen, as
its spacing must be right.

## 6. Smooth deformation

A specimen deforms slightly between the MRI scan and the OCT embedding, and serial sectioning adds its own small distortions.
After the affine, local misfits of a fraction of a millimetre remain that no affine removes. §6 fits a small smooth
displacement field on top of the affine. The affine stays the primary result: every transform file holds it alone, and the
field is written next to them.

The field u is defined on the MRI base grid, in mm along the MRI world axes, as a pull-back. The OCT content that belongs at
the MRI point x sits at x + u(x) in the affinely registered OCT, so the registered OCT at x is OCT(T⁻¹(x + u(x))). The OCT and
its specimen mask are resampled onto the MRI base grid through T (mask kept above 0.99), and two kinds of label-free local
evidence are measured on that volume.

Both kinds reach 1.35 mm, the one reach of the stage: the interior search range and the largest distance of a surface edge
from the MRI surface. The mask outlines of §1 serve the global stages but not this one: the specimen mask is
smoothed at σ 1.2 mm and the MRI foreground is a threshold, so neither places the surface to a fraction of a millimetre, and
the surface is read from the intensity of each volume by one rule.

Interior. Blocks of the MRI (4.5 mm, every 1.5 mm, at least 70 % of the block inside both masks eroded by 0.6 mm) are searched
in the OCT within ±1.35 mm by zero-mean normalised cross-correlation of the same normalised gradient n as §5, at σ 0.24 mm,
compared through the trace-free part of n nᵀ (six channels, `octreg.blockmatch`): the Frobenius product of two of them is
(n · n′)² up to a term in the two magnitudes, so a block score is the local counterpart of F, and it needs neither an intensity mapping nor the
polarity. A match counts when its peak stands 4 standard deviations above its score map and off the border of the range.
Matches longer than 1.35 mm are dropped, the search range being per axis. Each match gives a displacement vector at its block
centre.

Boundary. The surface of the MRI foreground is sampled at up to 50,000 of its outermost voxels, with outward normals from the
gradient of the signed distance to the surface (smoothed, σ 2.5 voxels). Along each normal the intensity is read over the reach
and 0.75 mm beyond it, from −2.1 to +2.1 mm, in base-grid steps and differentiated with a Gaussian of σ 1 sample. The surface edge of a volume is the steepest fall
of its intensity going outwards, which assumes that in both volumes the tissue at the surface is brighter than what surrounds
it. In the MRI the tissue falls into background. In the OCT the surface carries a bright rim, so the intensity falls going
outwards into the agarose even where the bulk scatters alike, and this does not depend on the polarity of §2, which concerns
the interior classes. A prominent fall is a local maximum of the fall more than 5 MADs above the median of the fall along the profile and within
±1.35 mm, and its position gets a parabolic sub-sample correction, which can carry it up to half a step beyond the reach. The edge is found by this one rule in the MRI and in
the OCT. The offset δ is the OCT edge minus the MRI edge along the normal n, and it constrains n · u. What the rule does to an
edge is the same in both volumes and cancels in the difference, and so does the place of the sampled surface voxels relative to
the true surface. A point is used only when at least 95 % of its OCT profile is measured and each of the two profiles has
exactly one prominent fall. Faces crossed by section stripes show several falls and mostly drop out, so the sectioning axis need
not be known. On the synthetic pair of tests/test_deform.py a quarter of the points on a striped face survive, against three
quarters elsewhere, and what survives there is what the Huber weights of the fit then hold down.

The edge is used and not the bright rim of the OCT, because the rim is a layer below the surface. Its ridge lies inside the
surface by half the thickness of the layer, which differs from face to face, so a rule that takes the ridge for the surface
reads an offset where there is none.

Model. Displacements c on a regular control lattice of 5 mm spacing over the bounding box of the MRI grid, trilinear in between,
are fitted by regularised least squares: three rows per interior match, one row n · u = δ per boundary point with the weight
√(3N / M) for N matches and M points, so that both kinds of evidence carry the same total weight, a membrane penalty λ on the
first differences of c along the lattice axes, and a small ridge on c. The system is solved four times, re-weighted by Huber's
rule at 0.3 mm on both kinds of residual.

Model choice. Only λ is chosen, and the strain limit chooses it. The field has to stay plausible for fixed tissue, so its
strain, the largest first difference of c divided by the spacing, must stay below 0.15. The strain falls as λ grows, and λ is
the smallest weight between 0.3 and 30 whose fit keeps the limit, found by bisection in log λ to a factor of 1.05: the most
flexible field the limit allows. The field is then accepted only when it predicts evidence it was not fitted to. Both kinds of
evidence are split into four spatial folds (cells of 7 mm), the field is fitted without one fold and tested on it, and the
held-out median interior error plus the held-out median boundary error must fall below 0.9 times the same score of no
deformation. A field is also refused with fewer than 100 interior matches or 300 boundary points, and when no weight up to 30
keeps the strain limit. Then the run is flagged `deformation_not_supported`, no field is written and oct_in_mri.nii.gz is the
affine overlay. While this stage was developed the held-out score was also compared across a list of weights above the
strain-limited one. On the 1.1 run of I58 it fell monotonically down to that weight in every variant tried, so the limit decides. The evidence is measured once and the field is fitted once. Both kinds of evidence are then measured again on the
warped OCT, the interior over every confident match before the length drop and the boundary over every point with one edge,
and these residuals are the read-outs of the stage.

The field is small by construction. Both kinds of evidence are searched within 1.35 mm, the strain limit bounds the model and
the held-out gain gates it. result.json holds, under `deform`, the lattice spacing and λ, the held-out errors without and with
the field, the residuals of both kinds measured before and after, the field magnitude over the MRI foreground and the largest
strain. oct2mri_warp.nii.gz holds u as a NIfTI vector
image on the MRI base grid, with components along the NIfTI world axes (ITK and ANTs expect the first two negated).
oct_in_mri.nii.gz is resampled through T and u, oct_in_mri_affine.nii.gz through T alone, and qc_deform.png shows the MRI, both
overlays and the field magnitude on three planes, each with the outline of the MRI foreground. qc.png and qc_montage.png keep
showing the affine. `octreg apply` takes OCT-frame volumes through T and u. The field is not inverted, so the MRI to OCT
direction stays affine.

## Evaluation

A registration is judged by looking at it, because label-free numbers can prefer a wrong pose. qc_montage.png shows four
planes per OCT axis, each as the OCT, the MRI through the transform on the grey scale of the OCT, a checkerboard of the two
wherever either map reaches, and the OCT with both outlines. In every plane the MRI boundary must follow the edge of the OCT
specimen, the OCT specimen must lie on the same anatomy in the MRI, and internal structures must continue across the
checkerboard. Folded pieces, and cut faces that the MRI also shows, must correspond and lie on the same side (where the block
was cut out of a larger specimen, the MRI tissue continues beyond the OCT cut face). The contrast must be consistently
inverted or not. Other candidate
transforms can be rendered with `octreg qc --T` and compared side by side.

qc.png shows the same four panels on one plane per OCT array axis through the specimen centroid.

The region the checkerboard is drawn over covers everything either the specimen mask or the MRI foreground reaches, and
leaves out only what both call background, where the two cannot disagree. Inside the intersection of the two maps they agree by construction, so a checkerboard drawn only there shows no
disagreement at any imposed error, and the worse the pose, the more such a panel hides.

### I58 brainstem pair

On the two original files (OCT 1457×2013×1595 at 20 µm, MRI crop 343×489×495 at 0.08 mm) the run takes 11 min 25 s, of which §6 is 20 s, with
5.2 GiB of RAM and 1.6 GiB of GPU memory allocated by torch (RTX 5090, Windows). §1-4 give S 0.2781 (S_class
−0.1020, S_outline 0.6302) and polarity −1. §5 raises F from 0.089 to 0.103 and moves the block by 3.8 mm at its corners
(2.0 mm over the specimen, 8.8°), with scales 0.977 / 0.974 / 0.985. At the §5 pose S is lower (0.244, S_class −0.055, measured on the run's base grids with the score of §2). Raw
20 µm OCT values mapped through the header and the transform correlate with the affine overlay (oct_in_mri_affine.nii.gz) at
Spearman 0.991, against at most 0.266 with any OCT axis flipped.

The figures of this pair are drawn locally by the bench scripts and are not published. bench/fig_registration.py shows the
result on three MRI planes, read from the original files: the MRI, the registered OCT, and those two panels cut into 8 mm
squares and interleaved, on their own grey scales and neither inverted nor matched nor masked.

After §5, misfits remain on I58 between the OCT surface and the MRI tissue boundary, mostly on the faces towards one end
of the sectioning axis, and the interior fine structure asks for an OCT that is smaller by 4, 2 and 0 % along three
axes. No affine removes both. The interior signal is not an artefact of the rim: matches at least 5 mm below the surface give
the same local affine as shallow ones (singular values 1.043, 1.018, 1.004 against 1.033, 1.015, 1.005), also with the masks
eroded by 1.5 mm (measured while §6 was developed, at the affine of release 1.0). In the full run §6 found 502 interior matches and 18,041 boundary points and took λ 0.82 at strain 0.148.
The held-out median errors fell from 0.247 to 0.155 mm (interior) and from 0.285 to 0.114 mm (boundary). The re-measured
residuals fell from 0.250 to 0.124 mm (interior) and from 0.285 to 0.089 mm (boundary, with 80 % of the offsets within 0.3 mm,
from 51 %). Over the MRI foreground the field has a median of 0.28 mm and a maximum of 1.05 mm. It is largest at the superior end of
the block. Measured per 4 mm along the sectioning axis in the ablation run below, the boundary residual there falls from
0.81 to 0.22 mm in the first 4 mm and from 0.65 to 0.19 mm in the next, and over the rest of the block from 0.10 to 0.40 mm
down to 0.04 to 0.13 mm. The affine of the run is that of release 1.0 to 0.001 mm at the block corners, and that of release
1.1 within 0.003 mm, the float noise between runs, since §1-5 compute the same affine.

Detached cerebellar parts that moved by more than the 1.35 mm reach, folia, a lobule and whole torn flaps, stay where the
affine puts them: no evidence reaches them, and no part of the method is built for them. In a checkerboard of this specimen the
brainstem continues across the tiles and the torn cerebellum breaks.

The boundary evidence of §6 is the surface edge because of what the rim does on this specimen (measured while §6 was
developed, at the affine of release 1.0). The OCT intensity along the MRI boundary normals peaks 0.3 mm inside the MRI
boundary and falls to half exactly at it. Taking the ridge of the rim for the
surface reads −0.77 mm on faces whose normal points along the sectioning axis, +0.43 mm on the opposite faces and −0.20 mm on
the faces across it, while the interior matches show no such pattern (against their normal component the ridge offsets follow
δ = 0.88 × interior − 0.36 mm, median disagreement 0.44 mm). The edge rule gives −0.09, +0.60 and +0.12 mm on the same faces
and a median disagreement with the interior of 0.27 mm, and in the ablation below it yields eight times as many usable points
(18,040 against 2,270) with a median |δ| of 0.29 instead of 0.56 mm. With the ridge evidence, in the same development runs, the strain of the field crossed the limit between weights 1 and 3
(strains 0.143 and 0.155), which is why the weight is found by bisection. With the edge evidence the held-out
score still improves a little past the strain limit (the strain 0.20 row of the ablation), so the limit sets λ and the
held-out score only gates the field. The folds test interpolation, which is what the field is used for: folds with a
buffer of 4.5 mm between training and held-out blocks (cells of 14 or 21 mm) predicted nothing in those runs, since the field
has a correlation length of a few millimetres.

The elements of §6 were ablated one at a time on the base grids of the I58 benchmark, at the affine of release 1.1 (within 0.003 mm
of the one above), in a separate run of the stage, so its rows can differ in the last digit from the run's own numbers above. Each row is scored
by the read-outs of the stage on the warped OCT, by the held-out score of its fit (without a field that score is the residual
itself, 0.247 / 0.285 mm) and by two numbers the fit never sees: F of §5 at σ 0.3 mm, and the two-class score of §2
(polarity × S_class) over the core 1.5 mm below both surfaces. The held-out score is taken on the evidence a row was fitted
to, so it compares the rows that keep the evidence (Huber, lattice, strain) and not the rows that change it.

| variant | λ | strain | interior (mm) | boundary (mm) | within 0.3 mm | held-out interior / boundary (mm) | F | two-class, core | field median / max (mm) |
|---|---|---|---|---|---|---|---|---|---|
| no deformation | | | 0.249 | 0.285 | 51 % | | 0.1029 | 0.0578 | 0 |
| §6 | 0.82 | 0.148 | 0.124 | 0.089 | 80 % | 0.155 / 0.114 | 0.1086 | 0.0641 | 0.28 / 1.05 |
| interior evidence only | 0.30 | 0.134 | 0.101 | 0.237 | 58 % | 0.143 / | 0.1058 | 0.0536 | 0.21 / 1.02 |
| boundary evidence only | 8.22 | 0.149 | 0.457 | 0.090 | 80 % | / 0.110 | 0.1052 | 0.0661 | 0.35 / 1.10 |
| rim ridge instead of the edge | 0.98 | 0.148 | 0.140 | 0.224 | 63 % | 0.169 / 0.309 | 0.1064 | 0.0599 | 0.27 / 1.02 |
| boundary points only within 5 mm of a match (the rule of 1.1) | 0.51 | 0.148 | 0.121 | 0.104 | 74 % | 0.155 / 0.093 | 0.1068 | 0.0682 | 0.22 / 1.16 |
| no Huber re-weighting | 1.51 | 0.149 | 0.141 | 0.104 | 77 % | 0.169 / 0.130 | 0.1080 | 0.0624 | 0.25 / 0.96 |
| lattice 7 mm | 0.79 | 0.148 | 0.140 | 0.094 | 79 % | 0.168 / 0.113 | 0.1086 | 0.0597 | 0.31 / 1.10 |
| lattice 10 mm | 0.30 | 0.148 | 0.156 | 0.088 | 81 % | 0.179 / 0.106 | 0.1086 | 0.0628 | 0.33 / 1.24 |
| strain limit 0.10 | 1.95 | 0.100 | 0.146 | 0.113 | 76 % | 0.169 / 0.144 | 0.1074 | 0.0646 | 0.24 / 0.87 |
| strain limit 0.20 | 0.40 | 0.199 | 0.108 | 0.079 | 82 % | 0.146 / 0.101 | 0.1092 | 0.0638 | 0.31 / 1.25 |
| reach 2.0 mm | 1.51 | 0.149 | 0.130 | 0.090 | 80 % | 0.212 / 0.145 | 0.1092 | 0.0595 | 0.29 / 1.57 |
| reach 2.7 mm | 3.46 | 0.149 | 0.145 | 0.115 | 76 % | 0.304 / 0.192 | 0.1082 | 0.0577 | 0.28 / 1.59 |

Each kind of evidence alone leaves the other kind where it was or makes it worse: the interior matches alone move the edges
only a little (0.285 to 0.237 mm), and the edge offsets alone pull the interior further off than the affine left it (0.249 to
0.457 mm). So both stay. A field fitted to the ridge offsets moves the edges part of the way (0.285 to 0.224 mm) and fits the
interior less closely (0.140 against 0.124 mm). Release 1.1 used only the boundary points within 5 mm of an interior match: the rule
withheld the edge evidence at the superior end of the block, where the OCT holds few matches, and left the edges there 0.39 mm off instead
of 0.22 mm (measured per 4 mm along the sectioning axis), without a gain elsewhere, so 2.0 uses every boundary point. Huber re-weighting, the 5 mm lattice
and the reach of 1.35 mm hold. Without the re-weighting and on the coarser lattices the fit is less close inside, and a longer
reach admits wrong matches: at 2.0 mm the residuals hardly move (0.130 / 0.090 mm) while the field grows (max 1.57 against
1.05 mm), and at 2.7 mm every residual is worse (0.145 / 0.115 mm). A strain limit of 0.20 fits a little closer on every residual and scores a little better held out.
The limit is a bound on plausible tissue strain rather than a fitted number, and 0.15 is kept. A second and a third round of
warping, measuring again and refitting, tried while the stage was developed, left the residuals within 0.015 mm of where they
were, so the stage runs once.

Each ablation changes one element of the method (the §1 grids and masks are computed once per mask source, and the no-§5 row
stops after §4). Pose changes are block-corner means, of the pose after §4 against the method's pose after §4 and of the final
pose against the result. The cut-face row compares the two outlines on the same cut.

| change | after §4 (mm) | final (mm) |
|---|---|---|
| intensity-threshold OCT mask instead of the texture mask | 41.9 | 42.9 |
| no outline term | 42.1 | 44.0 |
| polarity forced to +1 | 41.1 | 42.3 |
| no scale prior (λ 0, clamp 1.0): S 0.530 at the §4 pose, final scales 0.40-0.58 | 40.1 | 38.4 |
| the other handedness (OCT mirrored) | 29.8 | 30.5 |
| of the scale prior, the clamp alone (λ 0, clamp kept): final scales 0.86-0.88, at the bound | 41.2 | 38.7 |
| of the scale prior, the penalty alone (clamp 1.0, λ kept) | 0.00 | 0.00 |
| no fine-structure refinement (§5) | 0 | 3.75 |
| OCT flattening off | 0.24 | 0.00 |
| holes filled in 3-D only | 0.24 | 0.17 |
| standardised intensities instead of two-class maps | 2.19 | 0.00 |
| two-sided outline (agarose over MRI tissue counted as a mismatch) | 2.18 | 0.00 |
| MRI flattening off | 3.68 | 0.00 |
| cut face simulated by removing the mask beyond 70 % of its extent along OCT axis 1: method / two-sided outline | 4.04 / 2.55 | 0.11 / 0.11 |

The first five turn the block over or misplace it, and §5, which only refines, does not bring it back. The two rows that
split the scale prior show that all of it is the penalty: without the penalty the block collapses onto the clamp, at 0.86 to
0.88 of its length, and without the clamp the pose does not move at all. §5 itself moves the result by 3.8 mm. The next
five changes of the §4 pose lie within the reach of §5, which settles them, and with a simulated cut face both outlines
end at the same pose.

Two of the five move that pose by less than the 0.5 mm at which the benchmark deletes a step, OCT flattening (0.15 mm
over the specimen) and per-plane hole filling (0.22 mm), and both are kept for what they do rather than for this pose.
Flattening is one rule for both volumes, and on the MRI side it moves the §4 pose by 1.9 mm (3.7 mm at the corners).
Per-plane filling is what keeps the white matter of the block inside the specimen mask: uniform white matter has no
texture and forms cavities in the mask that reach its surface, which a 3-D fill leaves open, so filled in 3-D only the
mask loses 2.4 of its 19.2 cm³, and the evidence of §6 is read inside that mask. The one-sided outline moves the §4 pose
by 1.0 mm over the specimen (2.2 mm at the corners) on this pair, whose crop holds tissue below the block, and with a
simulated cut face (the last row) both outlines end 0.11 mm from the result.

On this pair the two-class maps are the weaker feature: |S_class| is 0.10 at the §4 pose and the standardised intensities of the
ablation table reach 0.14, and the block is found by the outline.

The prior of §5 was measured on its own. From the §4 pose of the released run, §5 was fitted at λ 2, 1, 0.5, 0.2 and 0 on
cached base grids with `bench/ngf_lam.py`, which runs §5 alone, so the weight reaches nothing else. Every pose was scored by
the outline agreement of `bench/evaluate.py`, which §5 does not read. The sweeps are stored under `bench/results/I58/ngf_lam/`.

| λ | F | scales per OCT axis | OCT to MRI rim (mm) | MRI to OCT rim (mm) |
|---|---|---|---|---|
| 2.0 (the method) | 0.1026 | 0.977 / 0.974 / 0.985 | 1.119 | 1.676 |
| 1.0 | 0.1080 | 0.955 / 0.952 / 0.972 | 0.986 | 1.495 |
| 0.5 | 0.1113 | 0.942 / 0.932 / 0.965 | 0.911 | 1.387 |
| 0.2 | 0.1140 | 0.932 / 0.899 / 0.964 | 0.878 | 1.289 |
| 0 | 0.1181 | 0.861 / 0.861 / 0.907 | 1.122 | 1.375 |

F rises as the weight falls, which it must, since F is what §5 maximises. The outline agreement, which it never sees, improves
with it down to λ 0.2, by 0.24 mm forward and 0.39 mm reverse, and the gain is spread over the faces rather than taken on one
of them: a0+ 1.34 to 1.06, a1+ 0.93 to 0.56, a1− 1.07 to 0.76, a2+ 2.28 to 2.01, a2− 2.40 to 2.04, with a0− flat at 0.77 to
0.79.
At λ 0 two log-scales sit on the clamp (0.8607 = exp(−0.15)) and the agreement breaks: a0+ falls to 0.81 mm while a0− rises to
1.41 and a1+ to 1.28, opposite faces moving opposite ways. The penalty is therefore load bearing in §5 as it is in §4, and the
clamp binds as soon as it goes.

What one weight cannot do is give the correction the shape the interior asks for. The local affine of the interior matches is
anisotropic, singular values 1.043, 1.018 and 1.004, a spread of 3.9 points between its largest and smallest (measured while
§6 was developed, at the affine of release 1.0). Sorted the same way, λ 1 shrinks by 4.8, 4.5 and 2.8 % (spread 2.0) and λ 2
by 2.6, 2.3 and 1.5 % (spread 1.1). The two sets are not in the same frame, so only the spread compares: lowering the weight
scales the block down as a whole and does not reach the shape.

A finer last pass costs instead of gains. With σ 0.25 appended to the schedule the rim is 1.134 / 1.696 mm and with 0.2 it
is 1.135 / 1.696, against 1.119 / 1.676 at the method's 0.6, 0.4, 0.3. The schedule 0.45, 0.3, 0.2 lands on the pose of 0.6,
0.4, 0.3, 0.2 (rim 1.135 / 1.696 in both), so the last pass is what decides. 0.2 mm is barely above the 0.15 mm base grid and
at the spacing of the section stripes, the scale §5 smooths away by stopping at 0.3. (F does not compare between schedules, each row
reporting it at its own finest σ, but the outline does.) Neither the stripes nor the grid is what stops the finer passes. A
per-section flat field along the array axis the stripes vary on, applied to the OCT before §5, is worth 0.005 mm of rim at
the method's schedule and leaves the finer one where it was (1.131 / 1.695 mm with σ 0.25 appended). And §5 on a 0.08 mm
grid of its own, the resolution of the MRI, moved the pose by about 0.2 mm at the method's schedule and 0.3 mm at 0.3, 0.2,
0.15 while the schedule was chosen, with the outline on that grid no better: at 0.08 mm the MRI carries its own noise and the
OCT its speckle, and the gradient directions are noisier, not sharper.

So the weight of the prior of §5, the scale its passes stop at and the grid it runs on are each where they should be. What
the interior asks for is not an affine the prior is holding back, and the misfit that is left is what §6 fits.

### Beyond the target data: DANDI:000026

The Broca-area blocks of DANDI:000026 (Costantini et al. 2023, public under CC-BY 4.0) are cortex slabs cut from a hemisphere
and imaged without embedding at 12 to 35 µm, each with an ex-vivo MRI of the whole hemisphere and labels the registration never
sees. They are not the target data of this method. The subjects are written sub-IXX here and the brainstem pair is never
written that way.

Eight subjects were run through `octreg register` by `bench/dandi.py`, with each MRI cut around the subject's own Broca-area
label. The script read six of those labels through headers that do not match the MRI, so only the crops of sub-I46 and sub-I55
were placed correctly. All DANDI results are withdrawn until a re-run, and the method is evaluated here on the brainstem pair
alone.

## Parameters

The constants of the method are fields of `octreg.params.Params` (override with `--params`). The brainstem pair used the
defaults, and the hash of the defaults is `da914d8ccc555207`. The first nine rows state what the method assumes about the
data, the scales of the embedding artefacts, of the intensity bias, of the fine structure and of the deformation, and the
weight of the scale prior. The rest set how a step is computed. The scales of §1, the sigmoid width, the rotation set and the
erosions were fixed by hand on this pair and not swept. A stage also holds guards and fixed rules at the top of its module,
outside Params, among them the clamp of ±0.15 on log-scales and shears in §4-5, and in §6 the range 0.3 to 30 of λ, the 10 %
held-out gain, the least evidence of 100 matches and 300 boundary points, the 0.75 mm a boundary profile runs beyond the
reach, the 50,000 surface points, the σ of 2.5 voxels on the signed distance, the four folds of 7 mm, the four Huber rounds and
the 70 % of a block that must lie inside both masks.

| parameter | default | role |
|---|---|---|
| `texture_bandpass_mm`, `texture_window_mm` | 0.08, 0.36 mm | §1: smoothing before, and window of, the directional coefficient of variation |
| `texture_smooth_mm`, `texture_close_mm` | 1.2, 0.48 mm | §1: smoothing of log F (set on the brainstem pair), closing radius |
| `flatten_sigma_mm` | 10 mm | §2: scale of the local foreground mean that flattens the intensities |
| `lam` | 2.0 | §4-5: weight λ of the scale and shear prior |
| `ngf_sigmas_mm`, `ngf_erode_mm` | 0.6, 0.4, 0.3 mm, and 0.8 mm | §5: gradient scales of the passes, erosion of both masks |
| `df_sigma_mm`, `df_block_mm` | 0.24, 4.5 mm | §6: gradient scale of the structure feature, edge of the matched blocks |
| `df_reach_mm` | 1.35 mm | §6: the reach of the local evidence, search range of the matches and largest distance of a surface edge from the MRI surface |
| `df_erode_mm` | 0.6 mm | §6: erosion of both masks that keeps the blocks inside |
| `df_grid_mm`, `df_max_strain` | 5 mm, 0.15 | §6: spacing of the control lattice, the strain limit |
| `search_mm`, `base_mm`, `fine_mm` | 0.6, 0.15, 0.04 mm | search grid, base grid, OCT grid of the specimen mask |
| `valley_ratio`, `min_component` | 0.5, 0.01 | MRI histogram valley / smaller peak, and the smallest MRI foreground component kept, as a fraction of the foreground |
| `texture_grid_mm`, `sigmoid_std` | 0.16 mm, 0.25 | texture blocks, and the sigmoid width in foreground standard deviations |
| `n_rot`, `seed`, `topk` | 8000, 0, 24 | rotations, rotation set seed, poses refined |
| `nms_mm`, `nms_deg` | 3 mm, 10° | poses closer in both count as one |
| `iters`, `lr_rot`, `lr_t`, `lr_ls`, `lr_sh` | 200, and 0.02 rad, 0.3 mm, 0.01, 0.01 | §4: Adam iterations and learning rates (§5: `ngf_iters` 150 per pass, a fifth of the rates) |
| `df_step_mm`, `df_z_min` | 1.5 mm, 4 | §6: grid step of the block centres, standard deviations above the mean of its score map a match needs |
| `df_edge_mad`, `df_huber_mm` | 5, 0.3 mm | §6: prominence of a fall in MADs above the median of the profile derivative, Huber threshold of both residuals |

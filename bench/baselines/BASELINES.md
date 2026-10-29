# Baselines on the I58 brainstem pair

Eight standard registration tools were run on the same two files as octreg, each with the settings its documentation gives
for volumes of different contrast and with each of its standard starting points (header alignment, image centres, the tool's
own global search, and the same with a field-of-view mask of the OCT). Nothing was tuned on this pair. Every transform was
checked against the image the tool itself resampled, and every result was resampled from the 20 um OCT by the same code, so
the overlays differ only by registration. The result is judged by the overlays on 13 planes through the block and four zooms
of its superior end, as for octreg. Settings and scripts: [README.md](README.md). All runs: `bench/results/I58/baselines/`.

## Affine

Each tool is shown with its best standard setting, judged on the overlays. The pose distance is the mean displacement over
the OCT specimen against octreg's affine, and the rotation the angle between the two. Times are wall clock with other
jobs running on the same machine and only indicative.

| method | setting | result | scale per OCT axis | from octreg's affine | time |
|---|---|---|---|---|---|
| image centres aligned (start, no registration) | header orientation | | 1.00 / 1.00 / 1.00 | 2.4 mm, 11.1° | |
| octreg 2.0, §1-5 | defaults | correct | 0.98 / 0.97 / 0.99 | | 10 min |
| NiftyReg reg_aladin | image centres | correct | 0.97 / 0.96 / 0.98 | 0.3 mm, 1.0° | 5 min |
| greedy, NMI | image centres, FOV mask | stays at the start | 1.00 / 1.00 / 1.00 | 2.4 mm, 11.0° | 1 min |
| elastix, Mattes MI | centre of gravity, FOV mask | stays near the start | 1.03 / 0.99 / 1.01 | 2.4 mm, 10.6° | 0.5 min |
| mri_robust_register, NMI | centre of mass | stays near the start | 0.99 / 1.00 / 1.07 | 3.0 mm, 11.7° | 14 min |
| ANTs, Mattes MI | centre of mass, FOV mask | shifted 2 to 3 mm, enlarged | 1.28 / 1.01 / 1.10 | 2.9 mm, 6.6° | 1 min |
| mri_coreg, NMI | brute-force start | shrunk | 0.92 / 0.81 / 1.00 | 4.1 mm, 16.6° | 22 min |
| FLIRT, normmi | default search, FOV weight | shrunk, rotated | 0.86 / 0.81 / 0.78 | 4.9 mm, 6.0° | 74 min |
| SynthMorph | as documented | fails (out of domain) | | 28 mm, 38° | 1 min |

The header orientation of this pair is 11 degrees from the pose, and the image centres put the block within a few
millimetres of it. From there greedy, elastix and mri_robust_register stay within about a degree of their start, ANTs, mri_coreg and
FLIRT change the size of the block by 10 to 30 % along some axis, and the three global searches (antsAI, FLIRT over ±180°,
greedy -search) end on a flipped or non-overlapping pose. Only reg_aladin, started from the image centres, finds the rotation:
it agrees with octreg's affine to 0.3 mm over the specimen and 1 degree, with a block up to 2 % smaller, and its overlays cannot
be told apart from octreg's. From the header alignment it ends 32 degrees off. octreg needs neither start, since it searches
every orientation inside the crop.

The boundary scores of `bench/evaluate.py` (rim distance, Dice of the masks) are reported in the full table but do not rank
the tools: a block that shrinks inside the MRI tissue scores well on them (elastix's run started from the image centres is
26 % smaller in volume and has a lower rim distance than octreg's affine). The scale column and the overlays decide.

## Deformable

Every tool starts from octreg's affine, with both volumes on the MRI grid, so the fields are the only difference. The three
residual columns are the read-outs of §6 measured again on the warped OCT (they are octreg's own measurement, not ground truth),
the field statistics are over the MRI foreground, and the last column is what the overlays show.

| method | setting | interior (mm) | edges (mm) | edges, superior 4 mm | field median / max (mm) | Jacobian min | overlays | time |
|---|---|---|---|---|---|---|---|---|
| no deformation | octreg's affine | 0.250 | 0.285 | 0.82 | 0 | 1 | | |
| octreg 2.0, §6 | defaults | 0.124 | 0.089 | 0.22 | 0.28 / 1.05 | 0.94 | no artefact | 20 s |
| ANTs SyN, CC | FOV mask | 0.063 | 0.027 | 0.05 | 0.29 / 1.62 | 0.34 | as octreg's, one corner slightly bowed | 48 min |
| ANTs SyN, CC | script default | 0.049 | 0.024 | 0.03 | 0.68 / 3.41 | 0.03 | cut faces pulled into MRI tissue the OCT never imaged, stripes bent | 88 min |
| greedy, WNCC | moving mask | 0.181 | 0.033 | 0.09 | 0.14 / 1.53 | 0.14 | superior cut face raised by up to 0.5 mm | 4 min |
| ConvexAdam, MIND-SSC | FOV mask | 0.220 | 0.052 | 0.09 | 0.48 / 1.92 | -0.54 | superior cut face lifted, corners bulged | 13 s (GPU) |
| elastix B-spline, Mattes MI | FOV mask | 0.367 | 0.254 | 0.65 | 0.27 / 0.59 | 1.00 | as octreg's, two small tears in the agarose | 43 s |
| NiftyReg reg_f3d, NMI | 5 mm grid | 1.232 | 0.615 | n/a | 4.82 / 12.70 | -0.70 | stripes bent into fans, block collapses in places | 8 min |
| SYMNet (test-time CNN) | affine start (smooth=3.0) | 0.000 | 2.399 | n/a | 0.00 / 0.01 | 0.97 | Good alignment, smooth gray-level matching with red MRI boundary contour tightly fitting | 7.6 min (GPU) |
| SYMNet (test-time CNN) | centre of mass (smooth=3.0) | 1.467 | 4.332 | n/a | 1.23 / 4.94 | -0.02 | Distorted structure relative to red contour, folded voxels | 7.7 min (GPU) |
| SYMNet (test-time CNN) | header start (smooth=3.0) | 0.404 | 5.006 | n/a | 0.00 / 3.28 | -0.01 | Partial alignment improvement; visible boundary offsets | 7.5 min (GPU) |
| GCPD (spectral mesh) | λ=1.0, outliers=0.1 | n/a | 0.98 | n/a | 16.48 / 48.99 | n/a | Surface residual 4.79 -> 0.98 mm, Dice 0.002 -> 0.370, LB energy 128.17 | 32 s |

Interior and edges are the median interior-match residual and the median surface-edge offset, and the superior 4 mm is the
edge residual at the end of the block where the affine leaves the largest misfit. A Jacobian below 0 means folded voxels
(0.08 % for ConvexAdam, 4.7 % for reg_f3d).

On the overlays no tool fits the intact tissue visibly better than §6, and the fields that keep the block shape look the same
as octreg's by eye, also in the zooms of the superior end (0.05 mm per pixel). The read-outs resolve
more, and they favour no method in particular. SyN with CC, given the field-of-view mask, fits both kinds of evidence closer
than §6 (interior 0.063 against 0.124 mm, edges 0.027 against 0.089 mm), with a field of the same median size, stronger local
compression (Jacobian down to 0.34 against 0.94) and about 150 times the run time. Without the mask, which is the default of
the official script, SyN reaches still lower residuals by pulling the straight cut faces of the OCT out into MRI tissue the
block does not contain, so those residuals are not a better registration. greedy and ConvexAdam close the edges but improve
the interior only a little (0.18 and 0.22 against 0.25 mm without a field), elastix leaves the edges near the affine and the
interior worse, and reg_f3d distorts the block in every standard form.

SYMNet (CVPR 2020), an unsupervised diffeomorphic deep learning model evaluated in test-time optimization mode across 800 iterations on this pair, achieves smooth gray-level matching without folding (Jacobian min 0.97) when pre-aligned with octreg's affine start; from unaligned starts (centre of mass or header) it struggles with large initial displacements and induces negative Jacobians (folding). GCPD (CVPR 2022), a spectral non-rigid surface matching baseline using the Laplace-Beltrami eigenbasis, registers the extracted specimen surface meshes (reducing surface residual from 4.79 mm to 0.98 mm and improving Dice from 0.002 to 0.370 at λ=1.0) without performing volumetric intensity deformation.

The full tables, with every variant, are in `bench/results/I58/baselines/affine.md` and `deform.md`.

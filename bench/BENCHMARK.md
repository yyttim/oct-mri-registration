# Benchmark: the I58 brainstem pair

octreg registered the two original files as given (OCT 1457x2013x1595 at 20 um, header LPI, and MRI crop 343x489x495 at 0.08 mm, header RIA) with `octreg register OCT MRI -o OUT` and default parameters (Params hash da914d8ccc555207). The pair has no labels, so every number here is label-free. The numbers are read from the run's result.json and eval.json and from ablations.json (copies in bench/results/I58/). Commands: `python bench/run_i58.py` (see bench/README.md).

## Visual result

The result is judged by visual inspection of the overlays (docs/METHOD.md, "Evaluation"). The numbers below support that judgement. The figures of the I58 pair are drawn locally by the bench scripts and are not published.

fig_handedness_I58.png (bench/compose_pair.py) shows planes through the centroid of the specimen mask, normal to each OCT array axis: the OCT, then for the result and for the best pose of the other handedness (ablation A8, also refined by §5) the MRI through the transform (inverted inside its foreground, polarity -1) and a 2 mm checkerboard. The mirrored pose fits the outline better (S_outline 0.6848 against 0.6302) and has the lower §4 loss (L 0.7073 against 0.7287), but its fine structure matches less (F 0.078 against 0.103).

qc.png of the run shows the MRI foreground through the transform (red) and the OCT specimen mask (cyan) in the fourth column, and qc_montage.png shows four planes per axis.

qc_deform.png of the run shows the MRI, the OCT through the affine, the OCT through the affine and the smooth field of §6, and the magnitude of that field, on three planes with the outline of the MRI foreground. The field is 0.28 mm in the median and 1.05 mm at most.

## Main result

| | |
|---|---|
| raw-data frame check (Spearman) | 0.991 at the pose, axis flips <= 0.266, 2 mm shifts <= 0.475, pass |
| score S of the §4 pose = (2 polarity S_class + S_outline) / 3, with S_class, S_outline and the polarity | 0.2781, -0.1020, 0.6302, -1 |
| search top-1 / top-2 | 0.2740 / 0.2667 |
| fine-structure agreement F (§5), at the §4 pose -> refined | 0.0886 -> 0.1026 |
| pose change of §5 (block-corner mean) | 3.75 mm |
| scale per OCT array axis | 0.977 / 0.974 / 0.985 |
| smooth deformation (§6): status, model, interior matches and boundary points used by the fit | applied, control lattice 5 mm, lambda 0.82, 502 and 18041 |
| §6 held-out median error, interior + boundary: no deformation -> the field | 0.247 -> 0.155 mm + 0.285 -> 0.114 mm |
| §6 residuals affine -> deformed, measured again: interior matches, surface-edge offsets (within 0.3 mm) | 0.25 -> 0.12 mm, 0.28 -> 0.09 mm (51 -> 80 %) |
| §6 field over the MRI foreground, median / p95 / max, and max strain | 0.28 / 0.72 / 1.05 mm, 0.15 |
| flags | none |
| pose vs the previous run, mean / corners mean / corners max (the mean is over the run's OCT specimen mask) | 0.00 / 0.00 / 0.00 mm |
| rim boundary agreement forward / reverse, method masks | 1.12 / 1.68 mm, previous 1.12 / 1.68 |
| OCT specimen mask | 19.24 cm3 |
| registration time in process, peak RAM, peak GPU memory allocated by torch | 11.4 min, 5.2 GiB, 1.59 GiB (wall clock 11.5 min, GPU memory in use on the device during the run, all processes, 9.7 GiB) |
| time per step (s) | mri 2, oct fine grid 127, oct mask 93, two class 3, search 33, refinement 345, fine-structure refinement 5, smooth deformation 20, outputs 56 |

The raw-data frame check passes. The pair has no labels, so none of these numbers is a success criterion.

## Ablations

Each variant is the method with one explicit change, run from the same preprocessed grids (one per OCT mask source). Pose change is against base (the method through the same driver), as the mean over the points of the base specimen mask and the mean and max over the 8 corners of the OCT array. The boundary agreement uses the base masks for every variant, so it reflects the pose only. The smooth deformation (§6) leaves the pose alone, so no variant runs it: 'no §6' would be base with the identical pose, and its read-outs are the 'before' residuals in the Main result table above.

| variant | change | §4 pose change: mean / corners mean (mm) | final pose change: mean / corners mean / corners max (mm) | S | L | polarity | scale | rim fwd / rev (mm) | OCT mask cm3 | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| base | the method | 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.2781 | 0.7287 | -1 | 0.977 / 0.974 / 0.985 | 1.12 / 1.68 | 19.24 | 399 |
| A0 | OCT intensity foreground (histogram valley) instead of the texture specimen mask | 19.94 / 41.85 | 23.89 / 42.92 / 66.12 | 0.0734 | 0.9272 | -1 | 0.994 / 0.994 / 1.000 | 2.61 / 2.11 | 29.52 | 418 |
| A0c | texture specimen mask with holes filled in 3-D only (method: in every array plane) | 0.22 / 0.24 | 0.11 / 0.17 / 0.34 | 0.2793 | 0.7280 | -1 | 0.971 / 0.974 / 0.986 | 1.11 / 1.65 | 16.88 | 398 |
| A1 | MRI flattening off | 1.94 / 3.68 | 0.00 / 0.00 / 0.00 | 0.2883 | 0.7259 | -1 | 0.977 / 0.974 / 0.985 | 1.12 / 1.68 | 19.24 | 388 |
| A2 | OCT flattening off | 0.15 / 0.24 | 0.00 / 0.00 / 0.00 | 0.2740 | 0.7330 | -1 | 0.977 / 0.974 / 0.985 | 1.12 / 1.68 | 19.24 | 396 |
| A4 | standardised intensity channels (z, -z) instead of two-class maps | 1.10 / 2.19 | 0.00 / 0.00 / 0.00 | 0.2967 | 0.7075 | -1 | 0.977 / 0.974 / 0.985 | 1.12 / 1.68 | 19.24 | 381 |
| A5+1 | polarity forced +1 | 19.63 / 41.14 | 20.15 / 42.27 / 46.87 | 0.2575 | 0.7505 | 1 | 0.992 / 0.995 / 0.994 | 1.79 / 2.51 | 19.24 | 402 |
| A5-1 | polarity forced -1 | 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.2781 | 0.7287 | -1 | 0.977 / 0.974 / 0.985 | 1.12 / 1.68 | 19.24 | 421 |
| A6 | no scale prior: lam 0 and clamp 1.0 (method: 2 and 0.15) | 17.41 / 40.10 | 17.09 / 38.42 / 44.65 | 0.5303 | 0.4697 | -1 | 0.502 / 0.583 / 0.398 | 3.54 / 2.39 | 19.24 | 480 |
| A6p | no penalty, clamp kept: lam 0, clamp 0.15 | 16.94 / 41.19 | 17.27 / 38.75 / 58.16 | 0.3339 | 0.6661 | -1 | 0.861 / 0.870 / 0.882 | 1.90 / 3.08 | 19.24 | 456 |
| A6c | no clamp, penalty kept: lam 2, clamp 1.0 | 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.2781 | 0.7287 | -1 | 0.977 / 0.974 / 0.985 | 1.12 / 1.68 | 19.24 | 390 |
| A8 | the other handedness: OCT world mirrored (z negated) before the search | 15.61 / 29.84 | 17.29 / 30.49 / 47.39 | 0.3087 | 0.7073 | -1 | 0.995 / 0.992 / 0.996 | 1.50 / 2.23 | 19.24 | 376 |
| A9 | no outline term: S = 2 S_class / 3 in the search and the refinement | 20.87 / 42.07 | 23.35 / 44.03 / 64.66 | 0.0975 | 0.9032 | -1 | 0.998 / 0.997 / 1.001 | 2.46 / 2.25 | 19.24 | 274 |
| A10 | two-sided outline: OCT embedding over MRI tissue or outside the crop counted as a mismatch | 0.97 / 2.18 | 0.00 / 0.00 / 0.00 | 0.2647 | 0.7398 | -1 | 0.977 / 0.974 / 0.985 | 1.12 / 1.68 | 19.24 | 339 |
| A11 | simulated cut face: specimen mask removed beyond 70 % of its extent along OCT axis 1 (data kept as embedding) | 2.25 / 4.04 | 0.09 / 0.11 / 0.21 | 0.3514 | 0.6617 | -1 | 0.974 / 0.975 / 0.986 | 1.12 / 1.67 | 19.24 | 403 |
| A11b | the same cut face with the two-sided outline | 1.54 / 2.55 | 0.09 / 0.11 / 0.21 | 0.3123 | 0.6909 | -1 | 0.974 / 0.975 / 0.986 | 1.12 / 1.67 | 19.24 | 358 |
| A12 | no fine-structure refinement (§5): the pose of §4 | 0.00 / 0.00 | 1.98 / 3.75 / 5.24 | 0.2781 | 0.7287 | -1 | 0.995 / 0.957 / 0.967 | 1.18 / 1.63 | 19.24 | 471 |

Driver check: base through bench/ablate.py lies 0.00 mm (corners max 0.00 mm) from the CLI run.

Deletion rule: a step goes when removing it moves the pose by at most 0.5 mm (mean over the specimen-mask points, the §4 pose for steps of §1-4 and the final pose for §5), changes no other metric beyond noise, and no test outside this pair shows it load bearing. Removing per-plane hole filling (A0c, 0.22 mm), OCT flattening (A2, 0.15 mm) or the scale clamp (A6c, 0.00 mm) stays within 0.5 mm, and the reason for keeping each is in docs/METHOD.md after the ablation table. Removing MRI flattening (A1, 1.94 mm), the two-class maps (A4, 1.10 mm), the scale prior (A6, 17.41 mm), the outline term (A9, 20.87 mm), the one-sided outline (A10, 0.97 mm) or the fine-structure refinement (A12, 1.98 mm) moves the pose further.

### Removed steps

Copied from the first ablation run, whose base still had the section-stripe flat field and the ladder (and the earlier score, two-class maps under an overlap gate), so pose changes in these rows are against that base. Removing either step moved the pose by less than the 0.5 mm deletion threshold and both were deleted.

| variant | change | §4 pose change: mean / corners mean (mm) | final pose change: mean / corners mean / corners max (mm) | S | L | polarity | scale | rim fwd / rev (mm) | OCT mask cm3 | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| A3 | section-stripe flat field off | n/a / n/a | 0.07 / 0.15 / 0.24 | 0.1454 | 0.8576 | -1 | 1.000 / 0.968 / 0.984 | 1.88 / 1.24 | 16.88 | 147 |
| A7 | affine directly at the finest level from every search pose, no ladder | n/a / n/a | 0.00 / 0.00 / 0.00 | 0.1486 | 0.8544 | -1 | 0.999 / 0.969 / 0.983 | 1.88 / 1.23 | 16.88 | 121 |

The present base lies 12.20 mm (corners mean 29.38 mm, corners max 46.12 mm, the two bases differ in handedness) from the base of that run. Source: ablate/ablations.json.

## Runtime and memory of the ablation driver

| step | seconds | peak RAM (GiB) |
|---|---|---|
| OCT fine grid 728x1006x797 (streamed once) | 134 | 3.3 |
| prep intensity | 30 | 3.8 |
| prep texture | 82 | 5.0 |
| prep texture3d | 81 | 5.0 |
| prep mri | 2 | 0.7 |
| all variants and preprocessing | 7080 | 5.0 |

Peak GPU memory allocated by torch over the variants: 1.4 GiB.

<!-- reading: written by hand below this line; bench/report.py keeps it when it rewrites the file -->

## Reading

Every variant except A12 ends with §5 on its own best pose. The table gives the change of the pose after §4 against the base's §4 pose and of the final pose against the base. All distances below are block-corner means.

A specimen mask rather than the intensity foreground, the outline term, the polarity rule, the scale prior and the handedness decide this pair. With the intensity-threshold OCT mask (A0, 29.5 cm3 against 19.2 cm3), without the outline term (A9), with the polarity forced to +1 (A5+1), without the scale prior (A6) or without its penalty (A6p) the block turns over, 38 to 44 mm from the result, and the mirrored OCT (A8) lies 30 mm away. §5, which only refines, does not bring any of them back. Without the scale prior S rises to 0.5303 (§4 pose) while the final OCT shrinks to 0.40-0.58 of its length along the three axes, and without the penalty alone it sits on the clamp at 0.86-0.88. The mirrored pose has the lower §4 loss (L 0.7073 against 0.7287) but the lower F (0.078 against 0.103).

§5 (A12) moves the result by 3.75 mm (1.98 mm over the specimen, 8.8 degrees). MRI flattening off (A1, 3.68 mm after §4), standardised intensities (A4, 2.19), the two-sided outline (A10, 2.18), OCT flattening off (A2, 0.24) and holes filled in 3-D only (A0c, 0.24) change the §4 pose, and §5 settles them to 0.00-0.17 mm. A2 and A0c lie within the deletion rule and are kept for the reasons given in docs/METHOD.md after the ablation table: flattening is one rule for both volumes and moves the §4 pose by 1.94 mm on the MRI side, and 3-D-only filling leaves 2.4 cm3 of white matter out of the specimen mask (16.88 against 19.24 cm3).

With a cut face simulated by removing the specimen mask beyond 70 % of its extent along OCT axis 1 and keeping the data there as embedding, the §4 pose moves 4.04 mm with the method's one-sided outline (A11) and 2.55 mm with the two-sided one (A11b), and §5 settles both to 0.11 mm. On the full block the one-sided outline shows in A10: counting agarose over MRI tissue as a mismatch moves the §4 pose by 2.18 mm, since this crop holds tissue below the block.

The "previous run" in the table is release 1.0, which gives the same affine, 0.001 mm at the block corners (release 1.1 is 0.003 mm away), float noise between runs, and the same boundary agreement. The §6 ablations are in docs/METHOD.md. The section-stripe flat field and the refinement ladder were removed after the first ablation run: switching off the flat field moved that run's pose by 0.15 mm (A3), and one affine refinement from every search pose gave the ladder's pose to within 0.0015 mm (A7).

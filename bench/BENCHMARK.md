# Benchmark: the I58 brainstem pair

octreg registered the two original files as given (OCT 1457x2013x1595 at 20 um, header LPI, and MRI crop 343x489x495 at 0.08 mm, header RIA) with `octreg register OCT MRI -o OUT` and default parameters (Params hash da914d8ccc555207). The pair has no labels, so every number here is label-free. The numbers are read from the run's result.json and eval.json and from ablations.json (copies in bench/results/I58/). Commands: `python bench/run_i58.py` (see bench/README.md).

## Visual result

The result is judged on the overlays (docs/METHOD.md, Evaluation), and the numbers below support that judgement. The I58 data are unpublished, so the figures are not in the repository.

The best pose of the other handedness (ablation A8, also refined by §5) fits the outline better than the result (S_outline 0.6848 against 0.6302) and has the lower §4 loss (L 0.7073 against 0.7287), but its fine structure matches less (F 0.078 against 0.103).

## Main result

| | |
|---|---|
| raw-data frame check (Spearman) | 0.991 at the pose, axis flips <= 0.266, 2 mm shifts <= 0.475, pass |
| score S of the §4 pose = (2 polarity S_class + S_outline) / 3, with S_class, S_outline and the polarity | 0.2781, -0.1020, 0.6302, -1 |
| search top-1 / top-2 | 0.2740 / 0.2667 |
| fine-structure agreement F (§5), at the §4 pose -> refined | 0.0886 -> 0.1026 |
| pose change of §5 (block-corner mean) | 3.75 mm |
| scale per OCT array axis | 0.977 / 0.974 / 0.985 |
| smooth deformation (§6): status, model, interior matches and surface-edge offsets used by the fit | applied, control lattice 5 mm, λ 0.82, 502 and 18041 |
| §6 held-out median error, interior matches + surface-edge offsets: no deformation -> the field | 0.247 -> 0.155 mm + 0.285 -> 0.114 mm |
| §6 residuals affine -> deformed, measured again: interior matches, surface-edge offsets (within 0.3 mm) | 0.25 -> 0.12 mm, 0.28 -> 0.09 mm (51 -> 80 %) |
| §6 field over the MRI foreground, median / p95 / max, and max strain | 0.28 / 0.72 / 1.05 mm, 0.15 |
| flags | none |
| rim outline agreement forward / reverse, method masks | 1.12 / 1.68 mm |
| OCT specimen mask | 19.24 cm3 |
| registration time in process, peak RAM, peak GPU memory allocated by torch | 11.4 min, 5.2 GiB, 1.59 GiB (wall clock 11.5 min, GPU memory in use on the device during the run, all processes, 9.7 GiB) |
| time per step (s) | mri 2, oct fine grid 127, oct mask 93, two class 3, search 33, refinement 345, fine-structure refinement 5, smooth deformation 20, outputs 56 |

The raw-data frame check passes. The pair has no labels, so none of these numbers is a success criterion.

## Ablations

Each variant is the method with one explicit change, run from the same preprocessed grids (one per OCT mask source). Pose change is against base, the method run through bench/ablate.py, as the mean over the points of the base specimen mask and the mean and max over the 8 corners of the OCT array. The outline agreement uses the base masks for every variant, so it reflects the pose only. The smooth deformation (§6) does not change the affine, so no variant runs it. Its read-outs are in the Main result above.

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
| A6 | no scale prior: λ 0 and clamp 1.0 (method: 2 and 0.15) | 17.41 / 40.10 | 17.09 / 38.42 / 44.65 | 0.5303 | 0.4697 | -1 | 0.502 / 0.583 / 0.398 | 3.54 / 2.39 | 19.24 | 480 |
| A6p | no penalty, clamp kept: λ 0, clamp 0.15 | 16.94 / 41.19 | 17.27 / 38.75 / 58.16 | 0.3339 | 0.6661 | -1 | 0.861 / 0.870 / 0.882 | 1.90 / 3.08 | 19.24 | 456 |
| A6c | no clamp, penalty kept: λ 2, clamp 1.0 | 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.2781 | 0.7287 | -1 | 0.977 / 0.974 / 0.985 | 1.12 / 1.68 | 19.24 | 390 |
| A8 | the other handedness: OCT world mirrored (z negated) before the search | 15.61 / 29.84 | 17.29 / 30.49 / 47.39 | 0.3087 | 0.7073 | -1 | 0.995 / 0.992 / 0.996 | 1.50 / 2.23 | 19.24 | 376 |
| A9 | no outline term: S = 2 S_class / 3 in the search and the refinement | 20.87 / 42.07 | 23.35 / 44.03 / 64.66 | 0.0975 | 0.9032 | -1 | 0.998 / 0.997 / 1.001 | 2.46 / 2.25 | 19.24 | 274 |
| A10 | two-sided outline: OCT embedding over MRI tissue or outside the crop counted as a mismatch | 0.97 / 2.18 | 0.00 / 0.00 / 0.00 | 0.2647 | 0.7398 | -1 | 0.977 / 0.974 / 0.985 | 1.12 / 1.68 | 19.24 | 339 |
| A11 | simulated cut face: specimen mask removed beyond 70 % of its extent along OCT axis 1 (data kept as embedding) | 2.25 / 4.04 | 0.09 / 0.11 / 0.21 | 0.3514 | 0.6617 | -1 | 0.974 / 0.975 / 0.986 | 1.12 / 1.67 | 19.24 | 403 |
| A11b | the same cut face with the two-sided outline | 1.54 / 2.55 | 0.09 / 0.11 / 0.21 | 0.3123 | 0.6909 | -1 | 0.974 / 0.975 / 0.986 | 1.12 / 1.67 | 19.24 | 358 |
| A12 | no fine-structure refinement (§5): the pose of §4 | 0.00 / 0.00 | 1.98 / 3.75 / 5.24 | 0.2781 | 0.7287 | -1 | 0.995 / 0.957 / 0.967 | 1.18 / 1.63 | 19.24 | 471 |

Base, run through bench/ablate.py, lies 0.00 mm (corners max 0.00 mm) from the CLI run of the Main result.

## Runtime and memory of bench/ablate.py

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

Every variant except A12 ends with §5 on its own best pose. The table gives the change of the pose after §4 against the base's §4 pose and of the final pose against the base. docs/METHOD.md reads this table in its section on the ablations of §1 to §5, and gives the ablations of §6, which are also stored in bench/results/I58/deform_ablation.md.

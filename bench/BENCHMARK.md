# Benchmark: the I58 brainstem pair

octreg registered the two original files as given (OCT 1457×2013×1595 at 20 µm, header LPI, and MRI crop 343×489×495 at 0.08 mm,
header RIA) with `octreg register OCT MRI -o OUT` and default parameters (Params hash 9148790ad6c39454). The numbers come from
result.json, eval.json, ablations.json and starts.json, stored in bench/results/I58/.

Commands: `python bench/run_i58.py` (see bench/README.md).

## Visual result

The pair has no reference, so the result is judged on the overlays in the
[README](../README.md#results-on-the-i58-brainstem-block).
Figures of the result and the ablations are in README.md, and of the baselines in bench/baselines/BASELINES.md.

## Main result

| | |
|---|---|
| raw-data frame check (Spearman) | 0.991 at the pose, axis flips <= 0.268, 2 mm shifts <= 0.475, pass |
| score S of the §4 pose = (2 polarity S_class + S_outline) / 3, with S_class, S_outline and the polarity | 0.2781, -0.1020, 0.6302, -1 |
| search top-1 / top-2 | 0.2740 / 0.2647 |
| score F of §5, at the §4 pose -> at the result | 0.0903 -> 0.1568 |
| pose change of §5 (block-corner mean) | 3.84 mm |
| scale per OCT array axis | 0.993 / 0.964 / 0.980 |
| smooth deformation (§6): status, model, interior matches and surface-edge offsets used by the fit | applied, control lattice 5 mm, λ 0.82, 510 and 18792 |
| §6 held-out median error, interior matches + surface-edge offsets: no deformation -> the field | 0.240 -> 0.165 mm + 0.311 -> 0.120 mm |
| §6 residuals affine -> deformed, measured again: interior matches, surface-edge offsets (within 0.3 mm) | 0.244 -> 0.117 mm, 0.311 -> 0.089 mm (49 -> 81 %) |
| §6 field over the MRI foreground, median / p95 / max, and max strain | 0.25 / 0.79 / 1.22 mm, 0.150 |
| flags | none |
| rim outline agreement forward / reverse, method masks | 1.10 / 1.69 mm |
| OCT specimen mask (base grid, intersected with the measured OCT) | 19.24 cm3 |
| registration time in process, peak RAM, peak GPU memory allocated by torch | 10.8 min, 5.2 GiB, 1.59 GiB |
| time per step (s) | mri 2, oct fine grid 130, oct mask 89, two class 2, search 31, refinement 316, fine-structure refinement 5, smooth deformation 21, outputs 50 |

## Ablations

Each variant is the method with one change, run from the same preprocessed grids (one per OCT mask source). L is the loss of §4,
1 - S plus the scale and shear penalty (docs/METHOD.md). A11 and A11b cut the specimen mask, so their mask column is empty. No
variant runs the smooth deformation (§6), which does not change the affine. Its ablation is in
bench/results/I58/deform_ablation.md.

| variant | change | S (§4 pose) | L (§4 pose) | polarity | scale per OCT axis | determinant | OCT mask cm3 | time (s) |
|---|---|---|---|---|---|---|---|---|
| base | the method | 0.2781 | 0.7287 | -1 | 0.993 / 0.964 / 0.980 | 0.938 | 19.24 | 627 |
| A0 | OCT intensity foreground (histogram valley) instead of the texture specimen mask | 0.0873 | 0.9129 | -1 | 1.014 / 1.000 / 1.000 | 1.013 | 29.52 | 469 |
| A4 | standardised intensity channels (z, -z) instead of two-class maps | 0.2967 | 0.7075 | -1 | 0.993 / 0.964 / 0.980 | 0.938 | 19.24 | 508 |
| A9 | no outline term: S = 2 \|S_class\| / 3 in the search and the refinement | 0.0975 | 0.9032 | -1 | 0.987 / 1.005 / 0.979 | 0.971 | 19.24 | 351 |
| A10 | two-sided outline: OCT embedding over MRI tissue or outside the crop counted as a mismatch | 0.2647 | 0.7398 | -1 | 0.993 / 0.964 / 0.980 | 0.938 | 19.24 | 444 |
| A12 | no fine-structure refinement (§5): the pose of §4 | 0.2781 | 0.7287 | -1 | 0.995 / 0.957 / 0.967 | 0.921 | 19.24 | 453 |
| A13 | no two-class term: S = S_outline / 3 in the search and the refinement, polarity +1 | 0.2228 | 0.7855 | 1 | 0.988 / 0.973 / 0.996 | 0.957 | 19.24 | 431 |
| A6p | no penalty, clamp kept: λ 0, clamp 0.15 | 0.3339 | 0.6661 | -1 | 0.861 / 0.861 / 0.896 | 0.660 | 19.24 | 594 |
| A14 | no orientation search: §4 from the image-centre start at both polarities, then §5 | 0.2781 | 0.7287 | -1 | 0.993 / 0.964 / 0.980 | 0.938 | 19.24 | 78 |
| A0c | texture specimen mask with holes filled in 3-D only (method: in every array plane) | 0.2793 | 0.7280 | -1 | 0.967 / 0.966 / 0.974 | 0.909 | 16.88 | 450 |
| A1 | MRI flattening off | 0.2883 | 0.7259 | -1 | 0.993 / 0.964 / 0.980 | 0.938 | 19.24 | 487 |
| A2 | OCT flattening off | 0.2740 | 0.7330 | -1 | 0.993 / 0.964 / 0.980 | 0.938 | 19.24 | 437 |
| A5+1 | polarity forced +1 | 0.2575 | 0.7505 | 1 | 0.982 / 1.022 / 1.005 | 1.008 | 19.24 | 440 |
| A5-1 | polarity forced -1 | 0.2781 | 0.7287 | -1 | 0.993 / 0.964 / 0.980 | 0.938 | 19.24 | 438 |
| A6 | no scale prior: λ 0 and clamp 1.0 (method: 2 and 0.15) | 0.5303 | 0.4697 | -1 | 0.550 / 0.696 / 0.391 | 0.139 | 19.24 | 437 |
| A6c | no clamp, penalty kept: λ 2, clamp 1.0 | 0.2781 | 0.7287 | -1 | 0.993 / 0.964 / 0.980 | 0.938 | 19.24 | 422 |
| A8 | the other handedness: OCT world mirrored (z negated) before the search | 0.3087 | 0.7073 | -1 | 0.984 / 0.980 / 0.964 | -0.929 | 19.24 | 363 |
| A11 | simulated cut face: specimen mask removed beyond 70 % of its extent along OCT axis 1 (data kept as embedding) | 0.3397 | 0.6658 | -1 | 0.972 / 0.954 / 0.973 | 0.902 |  | 401 |
| A11b | the same cut face with the two-sided outline | 0.3136 | 0.6906 | -1 | 0.972 / 0.954 / 0.973 | 0.902 |  | 361 |

Base, run through bench/ablate.py, lies 0.00 mm (corners max 0.00 mm) from the CLI run of the Main result.

## Start test

The OCT is turned by 90 and 180 degrees about each of its array axes, through the centre of its image box, and the method runs
from the image-centre start and from each turned start. Each cell gives the mean displacement from the base result (the method
from the file orientation) over the base specimen mask, and the angle of the rotation between the two poses. A14, the variant
without the orientation search, also runs from each start, and README.md shows it from the start turned by 90° about the
sectioning axis.

| start | start pose to base: mm / deg | method to base: mm / deg |
|---|---|---|
| image centres, header orientation | 2.46 / 11.4 | 0.00 / 0.0 |
| 90 deg about OCT array axis 0 | 15.26 / 83.2 | 0.09 / 0.4 |
| 180 deg about OCT array axis 0 | 22.77 / 172.8 | 0.00 / 0.0 |
| 90 deg about OCT array axis 1 | 14.33 / 85.6 | 0.00 / 0.0 |
| 180 deg about OCT array axis 1 | 20.45 / 175.1 | 0.09 / 0.4 |
| 90 deg about OCT array axis 2 | 15.19 / 97.6 | 0.00 / 0.0 |
| 180 deg about OCT array axis 2 | 20.18 / 172.7 | 0.09 / 0.4 |

## Runtime and memory of bench/ablate.py

| step | seconds | peak RAM (GiB) |
|---|---|---|
| OCT fine grid 728x1006x797 (streamed once) | 163 | 3.3 |
| prep intensity | 28 | 3.9 |
| prep texture | 82 | 5.0 |
| prep mri | 2 | 0.7 |
| prep texture3d | 95 | 4.9 |
| all variants and preprocessing | 8561 | 5.0 |

Peak GPU memory allocated by torch over the variants: 1.4 GiB.

<!-- reading: written by hand below this line, and bench/report.py keeps it when it rewrites the file -->

## Reading

Every variant except A12 ends with §5 on its own best pose. [README.md](../README.md#ablations) shows the main variants and
the start test in freeview, and [bench/i46](i46/README.md) scores seven of the variants (A4, A6p, A9, A10, A12, A13 and A14)
against the registration of Costantini et al. (2023).

# bench

The scripts that run and measure octreg, and their stored results. They are not part of the package. The I58 brainstem pair has
no reference registration, so it is shown in pictures. The test in [i46/](i46/README.md) scores octreg and standard tools on
block I46 of Costantini et al. (2023) against the authors' registration.

## Data layout

The scripts find their files through `bench/paths.py`. By default the runs go to `bench_runs/`, the baseline runs to
`baselines/` and the data are read from `data/` of the repository, all three in `.gitignore`. Four environment variables move
them.

| variable | meaning | default |
|---|---|---|
| `OCTREG_PROJECT_ROOT` | the runs go to `<project root>/bench_runs`, the baseline comparison to `<project root>/baselines` | the repository root |
| `OCTREG_DATA_ROOT` | the data root | `<repository>/data` |
| `OCTREG_I58_DIR` | the folder of the two I58 files | `<data root>/I58` |
| `OCTREG_DANDI000026_DIR` | the download of DANDI:000026 sub-I46 and the authors' registration files for `i46/` | `<data root>/dandi-000026` |

The I58 files are `I58_Brainstem_mus_Slice_full_20um_corr.nii.gz` (the OCT) and `I58_brainstem_MRI_cropped_to_OCT.nii.gz` (the
MRI crop). They are not part of the repository.

## I58

```bash
I58=${OCTREG_I58_DIR:-${OCTREG_DATA_ROOT:-data}/I58}
OCT=$I58/I58_Brainstem_mus_Slice_full_20um_corr.nii.gz
MRI=$I58/I58_brainstem_MRI_cropped_to_OCT.nii.gz
RUNS=${OCTREG_PROJECT_ROOT:-.}/bench_runs/I58
python -m octreg register $OCT $MRI -o $RUNS/octreg
python bench/ablate.py $OCT $MRI -o $RUNS/ablation
python bench/ablate.py $OCT $MRI -o $RUNS/ablation/turned --only base,no_search --turn 0,0,90
python bench/figures.py --octreg-run $RUNS/octreg --ablation $RUNS/ablation --deform-baselines DEFORM
```

`ablate.py` runs the method with one change per variant (its `--help` lists them), each through `octreg.register.register`. With
`--turn 0,0,90` it starts from the OCT turned by 90° about its sectioning axis, the z axis of the I58 OCT world.

| | | | |
|---|---|---|---|
| **no outline term** | **poses proposed by (F + O) / 2 only** | **no orientation search** | **no refinement** |
| <img src="../docs/figures/ablation/no_outline.png" width="175" alt="no outline term"> | <img src="../docs/figures/ablation/outline_decides.png" width="175" alt="poses proposed by (F + O) / 2 only"> | <img src="../docs/figures/ablation/no_search.png" width="175" alt="no orientation search"> | <img src="../docs/figures/ablation/no_refine.png" width="175" alt="no refinement"> |
| **only the first search pose refined** | **no scale and shear prior** | **signed gradients** | **signed gradients, OCT negated** |
| <img src="../docs/figures/ablation/first_choice.png" width="175" alt="only the first search pose refined"> | <img src="../docs/figures/ablation/no_prior.png" width="175" alt="no scale and shear prior"> | <img src="../docs/figures/ablation/signed.png" width="175" alt="signed gradients"> | <img src="../docs/figures/ablation/signed_neg.png" width="175" alt="signed gradients, OCT negated"> |
| **the trace-free tensor** | **search gradients over the whole specimen mask** | **no strain bound** | **turned 90°, the start** |
| <img src="../docs/figures/ablation/tracefree.png" width="175" alt="the trace-free tensor"> | <img src="../docs/figures/ablation/search_whole_mask.png" width="175" alt="search gradients over the whole specimen mask"> | <img src="../docs/figures/ablation/df_free.png" width="175" alt="no strain bound"> | <img src="../docs/figures/ablation/turned_start.png" width="175" alt="turned 90°, the start"> |
| **turned 90°, no orientation search** | **turned 90°, octreg** |  |  |
| <img src="../docs/figures/ablation/turned_no_search.png" width="175" alt="turned 90°, no orientation search"> | <img src="../docs/figures/ablation/turned_octreg.png" width="175" alt="turned 90°, octreg"> |  |  |

The sagittal plane of the I58 figure in the [README](../README.md) through each variant, with the MRI tissue boundary in red.

`figures.py` makes the freeview panels of `docs/figures`: the result, the affine and deformable baselines of
[baselines/](baselines/README.md) and the ablation. It runs after the baseline comparison, whose inputs give the display window
of every OCT panel (`--window-from`) and whose runs make the baseline panels. The deformable runs are read from
`--deform-baselines`, the root `DEFORM` of [baselines/](baselines/README.md). It needs FreeSurfer 8.2 (freeview, mri_mc,
mris_smooth) with its licence, in WSL on Windows.

## Files

| file | content |
|---|---|
| `paths.py` | the run and data roots and the two I58 files |
| `ablate.py` | the ablation: one change to the method per variant, on any pair |
| `figures.py` | the I58 panels in `docs/figures/` |
| `baselines/` | standard registration tools on the I58 pair, and the tool settings of the I46 test ([README](baselines/README.md), [BASELINES.md](baselines/BASELINES.md)) |
| `i46/` | the test on block I46 of Costantini et al. (2023) ([README](i46/README.md)) |
| `results/I58/` | `result.json` of the I58 run and the baseline tables in `baselines/` |
| `results/I46/` | the scores of the I46 test and their tables (`tables.md`) |

The scripts print their options with `--help`.

# bench

The scripts that run and measure octreg, and their stored results. They are not part of the package. The I58 brainstem pair has
no reference registration, so it is shown in pictures. The test in [i46/](i46/README.md) scores octreg and standard tools on
block I46 of Costantini et al. (2023) against the authors' registration.

## Data layout

The scripts find their files through `bench/paths.py`. By default the runs, the baseline runs and the data lie in folders of the
repository that `.gitignore` leaves out, and four environment variables move them.

| variable | meaning | default |
|---|---|---|
| `OCTREG_PROJECT_ROOT` | the runs go to `<project root>/bench_runs`, the baseline comparison to `<project root>/baselines` | the repository root |
| `OCTREG_DATA_ROOT` | the data root | `<repository>/data` |
| `OCTREG_I58_DIR` | the folder of the two I58 files | `<data root>/I58` |
| `OCTREG_DANDI000026_DIR` | the download of DANDI:000026 sub-I46 and the authors' registration files for `i46/` | `<data root>/dandi-000026` |

## I58

```bash
I58=${OCTREG_I58_DIR:-${OCTREG_DATA_ROOT:-data}/I58}
OCT=$I58/I58_Brainstem_mus_Slice_full_20um_corr.nii.gz
MRI=$I58/I58_brainstem_MRI_cropped_to_OCT.nii.gz
RUNS=${OCTREG_PROJECT_ROOT:-.}/bench_runs/I58
python -m octreg register $OCT $MRI -o $RUNS/octreg
python bench/ablate.py $OCT $MRI -o $RUNS/ablation
python bench/ablate.py $OCT $MRI -o $RUNS/ablation/turned --only base,no_search --turn 0,0,90
python bench/ablate.py $OCT $MRI -o $RUNS/ablation/fine --only base,no_outline --params '{"search_mm": 0.4, "search_sigma_mm": 0.8}'
python bench/repeat.py $OCT $MRI --run $RUNS/octreg -o $RUNS/repeat
```

With `--turn 0,0,90`, `ablate.py` starts from the OCT turned by 90° about its sectioning axis, the z axis of the I58 OCT world.
`baselines/summarise.py` writes `ablation.json` and `search.md` in `results/I58` from these runs.

| | | | |
|---|---|---|---|
| **no outline term** | **poses proposed by (F + O) / 2 only** | **no orientation search** | **no refinement** |
| <img src="../docs/figures/ablation/no_outline.png" width="175" alt="no outline term"> | <img src="../docs/figures/ablation/outline_decides.png" width="175" alt="poses proposed by (F + O) / 2 only"> | <img src="../docs/figures/ablation/no_search.png" width="175" alt="no orientation search"> | <img src="../docs/figures/ablation/no_refine.png" width="175" alt="no refinement"> |
| **only the first search pose refined** | **no scale and shear prior** | **signed gradients** | **signed gradients, OCT negated** |
| <img src="../docs/figures/ablation/first_choice.png" width="175" alt="only the first search pose refined"> | <img src="../docs/figures/ablation/no_prior.png" width="175" alt="no scale and shear prior"> | <img src="../docs/figures/ablation/signed.png" width="175" alt="signed gradients"> | <img src="../docs/figures/ablation/signed_neg.png" width="175" alt="signed gradients, OCT negated"> |
| **the trace-free tensor** | **search gradients over the whole specimen mask** | **turned 90°, the start** | **turned 90°, no orientation search** |
| <img src="../docs/figures/ablation/tracefree.png" width="175" alt="the trace-free tensor"> | <img src="../docs/figures/ablation/search_whole_mask.png" width="175" alt="search gradients over the whole specimen mask"> | <img src="../docs/figures/ablation/turned_start.png" width="175" alt="turned 90°, the start"> | <img src="../docs/figures/ablation/turned_no_search.png" width="175" alt="turned 90°, no orientation search"> |
| **turned 90°, octreg** | **scale and shear prior at a quarter of its weight** | **scale and shear prior at four times its weight** | **rigid refinement** |
| <img src="../docs/figures/ablation/turned_octreg.png" width="175" alt="turned 90°, octreg"> | <img src="../docs/figures/ablation/prior_0.5.png" width="175" alt="scale and shear prior at a quarter of its weight"> | <img src="../docs/figures/ablation/prior_8.png" width="175" alt="scale and shear prior at four times its weight"> | <img src="../docs/figures/ablation/rigid.png" width="175" alt="rigid refinement"> |
| **F without the subtraction of the means** | **MRI divided by its Gaussian blur, σ 5 mm** | **OCT specimen mask from the signal threshold alone** | **OCT voxel size 5 % too large** |
| <img src="../docs/figures/ablation/uncentred.png" width="175" alt="F without the subtraction of the means"> | <img src="../docs/figures/ablation/flatfield.png" width="175" alt="MRI divided by its Gaussian blur, σ 5 mm"> | <img src="../docs/figures/ablation/signal_only.png" width="175" alt="OCT specimen mask from the signal threshold alone"> | <img src="../docs/figures/ablation/spacing_p5.png" width="175" alt="OCT voxel size 5 % too large"> |
| **OCT voxel size 5 % too small** | **OCT array mirrored** | **search on a 0.4 mm grid with σ 0.8 mm, no outline term** | |
| <img src="../docs/figures/ablation/spacing_m5.png" width="175" alt="OCT voxel size 5 % too small"> | <img src="../docs/figures/ablation/mirrored.png" width="175" alt="OCT array mirrored"> | <img src="../docs/figures/ablation/fine_no_outline.png" width="175" alt="search on a 0.4 mm grid with σ 0.8 mm, no outline term"> | |

The sagittal plane of the I58 figure in the [README](../README.md) through each variant, with the MRI foreground boundary in
red.

`figures.py` makes the freeview panels of `docs/figures`: the result, the affine baselines of [baselines/](baselines/README.md)
and the ablation. It runs after the baseline comparison, whose inputs give the display window of every OCT panel
(`--window-from`) and whose runs make the baseline panels. It needs FreeSurfer 8.2 (freeview, mri_mc, mris_smooth) with its
licence, in WSL on Windows.

```bash
python bench/figures.py --octreg-run $RUNS/octreg --ablation $RUNS/ablation
```

## Files

| file | content |
|---|---|
| `paths.py` | the run and data roots, the two I58 files and the form of a path in WSL |
| `ablate.py` | the ablation: one change to the method or its input per variant, on any pair |
| `repeat.py` | octreg once more on a pair, its affine against a stored run of the same version, into `results/repeat.json` |
| `figures.py` | the I58 panels in `docs/figures/` |
| `baselines/` | standard registration tools on the I58 pair, and the tool settings of the I46 test ([README](baselines/README.md), [BASELINES.md](baselines/BASELINES.md)) |
| `i46/` | the test on block I46 of Costantini et al. (2023) ([README](i46/README.md)) |
| `results/I58/` | `result.json` of the I58 run with the input paths cut to file names, `ablation.json` with the pose and the search candidates of every ablation run, `search.md` with the candidates of the search in the I58 run, the two outline variants and the `fine` runs, and the baseline tables in `baselines/` |
| `results/I46/` | the scores of the I46 test and their tables (`tables.md`) |

The Python scripts print their options with `--help`.

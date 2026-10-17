# Changelog

## 3.0.0

Params hash c4963318cacfc67d.

One score in the search, the refinement and the deformation: the correlation of the trace-free orientation tensors of the
normalised gradients of the two volumes, which needs no intensity mapping and no contrast polarity
([docs/METHOD.md](docs/METHOD.md)). §1 to §5 instead of §1 to §6. The score is in the new module `octreg.features`.

- §2: the score F replaces the two-class maps. S = (F + O) / 2 with the one-sided outline O where both images show an outside,
  else S = F.
- §3: the search correlates the six tensor channels by FFT. The rotations are the identity and a Super-Fibonacci set.
- §4: every search pose is refined on F alone at σ 0.6, 0.4 and 0.3 mm, and the lowest final L wins. The clamp on scales and
  shears is removed, and the prior λ = 2 stays.
- §5: the smooth deformation is fitted to F, with the tissue surface in the score, under a strain bound of 0.08 on a 5 mm
  lattice, by L-BFGS. It runs on every pair (`--no-deform`, or `deform=False` of `register` in Python, skips it).
- Parameters: `erode_mm`, `search_sigma_mm`, `sigmas_mm`, `df_band_mm`, `df_sigmas_mm` and `df_iters` are new. `iters` counts
  the Adam iterations of one pass (150), and the learning rates are those of the passes (`lr_rot` 0.004, `lr_t` 0.06, `lr_ls`
  and `lr_sh` 0.002). `df_max_strain` is 0.08 (0.15 in 2.x).
- `result.json`: the pose holds S and the rank of the search pose it came from, the search pose, L, the passes on F and
  `display_polarity`, the sign used by the QC images. `candidates` holds the search rank, S and final L of every refined pose.
  `pose.F` and `shift_from_search_mm` take the place of `NGF` and `ngf_shift_mm`, and the pose keys `polarity`, `S_class`,
  `S_outline`, `T_before_ngf` and `NGF_start` are gone, and so are the `ngf` and `refine` blocks. `seconds` has `search` and
  `refine` in place of `two_class` and `search_refine_ngf`. `deform` holds the passes, `strain_max` and `seconds`, and it is
  null with `--no-deform`. `inputs.oct_mask` and `inputs.mri_mask` are absolute paths, so `octreg qc` finds them from any
  folder. The flags `clamp_saturated` and `deformation_not_supported` are gone. `octreg qc --run` reads 3.0 run directories.
- `bench/ablate.py`: one change to the method per variant, and starts from a turned OCT (`--turn`). OCT and MRI are positional
  arguments, `--oct-spacing-um` is new, and `--main`, `--starts`, `--random-starts`, `--seed`, `--force` and `--pair` are gone.
- `bench/i46`: the vessel check of the deformation (`common.vessel_field`), the ablation of 3.0 and the start test in
  `analyses.py`, whose `near` step and its options `--checkout` and `--device` are gone.

Removed

- The two-class maps, the polarity, the block matching and surface-edge fit of the old §6 with its held-out gate, the modules
  `octreg.ngf` and `octreg.blockmatch`, and the parameters `seed`, `flatten_sigma_mm`, `sigmoid_std`, `ngf_*` and the `df_*` of
  the old §6 except `df_grid_mm` and `df_max_strain`.
- `bench/run_i58.py`, `bench/evaluate.py`, `bench/report.py`, `bench/BENCHMARK.md`, `bench/compose_pair.py`,
  `bench/fig_registration.py`, `bench/ablate_deform.py`, `bench/ngf_lam.py`, and the stored results of the old ablations and
  evaluation (`bench/results/I58/ablations.json`, `starts.json`, `eval.json`, `deform_ablation.*`,
  `bench/results/I46/near.json`).

## 2.1.0

Params hash 9148790ad6c39454.

- §1: an MRI histogram valley is the tissue edge only when its darker peak lies below `medium_fraction` (0.5) of the brightest
  peak. A valley between two tissue peaks is passed over and flagged `mri_tissue_valley_passed`.
- §2: the outline is used only when both images have a measured outside, at least `outline_min_outside` (1 %) of the measured
  voxels each. Otherwise S_outline is 0 and the run is flagged `outline_not_used`.
- §3: candidates within 3 mm and 20° (`nms_deg`) count as one pose.
- §5: the score is the weighted correlation of the trace-free orientation tensors of the two normalised gradients, with a finite
  gradient where the OCT and the MRI points do not overlap.

Added

- `bench/i46/`: a test on block I46 of Costantini et al. (2023) against the authors' registration, with eight standard tools,
  a start test and ablations of the steps of the method. The results are in `bench/results/I46/` and the figures in
  `docs/figures/i46/`.
- `bench/baselines/`: a comparison with standard registration tools on the I58 pair, eight tools for the affine and five
  deformable tools started from octreg's affine, each at the setting its documentation gives. The tables are in
  `bench/results/I58/baselines/`.
- `bench/figures.py`: freeview figures of the I58 result, the baseline comparison and the ablation in `docs/figures/`.
- `bench/ablate.py`: two variants (the outline alone, no orientation search), start tests from turned starts (`--starts`) and
  from random starts (`--random-starts`), and runs on other pairs (`--oct`, `--mri`, `--oct-mask`).
  `bench/ablate_deform.py --save-fields` saves the fields of chosen §6 rows.
- Every bench script imports the octreg package of its own checkout, also when another copy is installed.

Removed

- `bench/dandi.py` and the stored results in `bench/results/I58/ngf_lam/`.

## 2.0.0

- Label-free registration of a serial-sectioning OCT block to an ex-vivo MRI crop in six steps: §1 specimen mask, §2 two-class
  maps and outline, §3 orientation search, §4 affine refinement, §5 fine-structure refinement and §6 smooth deformation
  ([docs/METHOD.md](docs/METHOD.md)).
- The command line `octreg register`, `octreg qc` and `octreg apply`, the same functions in `octreg.register`, and the method
  constants in `octreg.params.Params`, which `--params` overrides.
- Outputs: the affine as 4×4 text files, a FreeSurfer LTA and an ITK transform, the deformation field of §6 as a NIfTI vector
  image, the OCT resampled onto the MRI grid and the MRI into the OCT frame, QC images and `result.json`.
- `bench/`: the scripts of the I58 benchmark (registration, ablations, label-free evaluation, figures and report) and its
  stored results in `bench/results/I58/`.
- Tests on synthetic data that run on the CPU (`pytest`).

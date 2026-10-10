# Changelog

## Unreleased

- `bench/baselines/`: a comparison with standard registration tools on the I58 pair, eight tools for the affine and five
  deformable tools started from octreg's affine, each with its documented settings. The scripts that run and evaluate the
  comparison are included, and its tables are in `bench/results/I58/baselines/`.
- Figures of the I58 result and of the baseline comparison, freeview screenshots in `docs/figures/` rendered by
  `bench/figures.py`, in README.md and bench/baselines/BASELINES.md.
- An ablation of each step on the I58 pair in docs/METHOD.md, with freeview figures: two new variants in `bench/ablate.py`
  (the outline alone, no orientation search), a start test from turned starts (`--starts`), and the fields of chosen §6 rows
  saved by `bench/ablate_deform.py --save-fields`. The full tables are in bench/BENCHMARK.md and
  `bench/results/I58/deform_ablation.md`.
- Every bench script imports the octreg package of its own checkout, also when another copy is installed.

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

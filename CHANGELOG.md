# Changelog

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

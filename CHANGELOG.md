# Changelog

## 3.1.0

A specimen mask for OCT in bright, dark or clipped medium, a simpler MRI foreground, the full orientation tensor and a search in
which the outline only proposes poses ([docs/METHOD.md](docs/METHOD.md)).

- §1: the OCT specimen is the voxels with signal and isotropic texture, both decided by one threshold rule, the lower of Otsu's
  threshold and the low-side foot of the tallest histogram peak, each on its own histogram. There is no closing, and components
  holding at least 1 % of the mask are kept. The MRI foreground lies above the first histogram valley, walking down from the
  brightest peak, that holds less than half of its smaller peak and whose darker class lies mostly outside the convex hull of
  the voxels above it, and it is every positive voxel without such a valley. Its histogram peaks are found with an empty bin
  added at each end, and the MRI foreground has no closing. `--oct-spacing-um` applies to an `--oct-mask` of the OCT's array
  shape, and a mask of another shape keeps the spacing of its own file or, as a plain TIFF or NPY, is refused.
- §2: F uses the full orientation tensor n nᵀ (squared normalised gradient fields). The OCT points lie farther than 0.8 mm
  inside the specimen mask in Euclidean distance.
- §3: every rotation keeps its best translation by F and, where both images show medium around the tissue, by (F + O) / 2.
  The best 12 poses by (F + O) / 2 and the rest by F go to §4. The OCT gradients of the search are taken inside the eroded
  mask of its points, so a margin of medium inside the specimen mask adds no edge. The rotations are 8,000 Super-Fibonacci
  points.
- §4: one Adam step size for all parameters, in mm of motion of the OCT box corners (`lr_mm`), each pass keeps its last iterate,
  and among the poses whose final L lies within 10⁻⁵ of the lowest, the one proposed first wins.
- §5: the deformation is fitted to the F of §2 over the same points, without the tissue surface, and repeated runs give the same
  field.
- Parameters: `lr_mm` and `df_evals` are new. `medium_fraction`, `texture_close_mm`, `lr_rot`, `lr_t`, `lr_ls`, `lr_sh`,
  `df_band_mm` and `df_iters` are gone. `Params` refuses a value of the wrong type, a value that is not finite and a value out
  of its range, also when built in Python, and `search_mm` need not be a multiple of `base_mm`.
- Inputs: a mask file in another frame than its image (header or array) is refused before any output is written, and an
  OCT without positive voxels is refused with a message.
- Resampling: the interpolation positions of the isotropic grids are rounded to 10⁻⁹ voxel. A position on a pooled voxel
  centre took a residue weight from its neighbour, which left tiny values next to exact zeros and moved the §1 threshold on an
  OCT whose medium is clipped to zero.
- Outputs and commands: `T_mri2oct.txt`, the command `octreg qc` with its function `octreg.register.qc`, and `Params.hash()` are
  gone, and `octreg apply --inverse` inverts `T_oct2mri.txt`. In `result.json` every candidate holds its search rank, F and O at
  the search and its final L, and the pose has `search_F` and `search_O` in place of S and no `shift_from_search_mm`,
  `log_scales` or `shears`. `foreground.oct` has `signal_threshold` and `texture_threshold` in place of `threshold` and the
  source `octreg` in place of `texture`, and the run time of §5 is in `seconds.deform` only. The flags, the parameter hash, the
  search block, the display polarity, the strain read-out, the memory peaks, the component counts and the valley details of the
  MRI foreground are gone.
- bench: the I46 test gives every method the released OCT, and octreg runs with its own masks. Further runs give octreg the
  test's specimen mask. The deformable baselines run from `bench/baselines/deform_tools.py`. `bench/baselines/composites.py` and
  `zooms.py`, the `--shots` option of `evaluate_runs.py` and `evaluate_deform.py` and the `--stages` option of
  `bench/i46/figures.py` are gone. The ablation variants `tracefree`, `outline_decides` and `search_whole_mask` replace
  `full_tensor`, `outline_refine` and `df_interior`. A run of `bench/ablate.py` or `bench/i46/run.py` records the identity of
  its octreg code in `code.txt` and is reused only by the same code.

## 3.0.0

One score in the search, the refinement and the deformation: the correlation of the trace-free orientation tensors of the
normalised gradients of the two volumes, which needs no intensity mapping and no contrast polarity. §1 to §5 instead of §1 to
§6, and the score is in the module `octreg.features`.

- §2: the score F replaces the two-class maps. S = (F + O) / 2 with the one-sided outline O where both images show an outside,
  else S = F.
- §3: the search correlates the six tensor channels by FFT. The rotations are the identity and a Super-Fibonacci set.
- §4: every search pose is refined on F alone at σ 0.6, 0.4 and 0.3 mm, and the lowest final L wins. The clamp on scales and
  shears is removed, and the prior λ = 2 stays.
- §5: the smooth deformation is fitted to F, with the tissue surface in the score, under a strain bound of 0.08 on a 5 mm
  lattice, by L-BFGS. It runs on every pair (`--no-deform`, or `deform=False` of `register` in Python, skips it).
- `result.json` holds the search rank, S and final L of every refined pose, and `deform` is null with `--no-deform`. The mask
  paths in `inputs` are absolute.
- The two-class maps, the polarity, the block matching and surface-edge fit of the old §6, the modules `octreg.ngf` and
  `octreg.blockmatch`, and their parameters are gone.

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

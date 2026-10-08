# Changelog

## Unreleased

- §5 has its own prior weight `ngf_lam`, default 2.0, the value §4 uses, so §1-5 compute the same affine as 1.1.0 and the
  released run reproduces to the digits docs/METHOD.md prints. Until now the fine refinement inherited §4's `lam` through
  `refine.fit_adam` and the two could not be varied apart; `refine.fit_adam` now takes the weight as an argument, defaulting to
  `params.lam`. §1-4 and §6 are untouched.
- `bench/ngf_lam.py`: the §1 base grids cached once, then §5 alone from a run's §4 pose, once per weight, with the outline
  agreement of `bench/evaluate.py` at every pose. The measurement on the brainstem pair is in docs/METHOD.md: the penalty is
  load bearing in §5 as it is in §4, the clamp binds without it, and one weight cannot give the correction the shape the
  interior asks for. The default stays at 2.0.
- §5's prior splits into the size of the block and its shape, `ngf_lam` and `ngf_lam_shape`, equal at 2.0 by default, where
  together they are the penalty of §4. The default Params hash is `ab058ece8f3093ba`.
- `bench/ngf_lam.py` also sweeps the gradient scales, flat-fields the section stripes out of the OCT (`--destripe`) and
  caches the volumes on a grid of its own (`--grid-mm`), so §5 can be run finer than §1-4. All three are measured in
  docs/METHOD.md and none of them improves the pose.
- tests: the synthetic pair of `tests/test_ngf.py` takes an optional scale, and a new test builds it 6 % longer along one axis
  and checks that `ngf_lam` 0 takes that scale while a strong weight refuses it.
- The figures of the I58 pair are no longer in the repository, since the data are unpublished. The bench scripts draw them
  locally.
- The DANDI:000026 results are withdrawn until a re-run. `bench/dandi.py` built six of the eight MRI crops through label
  headers that do not match the MRI, and only sub-I46 and sub-I55 were placed correctly. Their table,
  `bench/results/dandi/summary.json`, is removed, and `bench/dandi.py --steps summary` now writes `summary.json` into its
  output root.
- The documents no longer report counts or measurements from visual readings of the QC images. The numbers the code computes
  are unchanged.
- The I58 pair is named by its subject alone, and the bench files follow: the runner is `bench/run_i58.py`, the stored results
  are in `bench/results/I58/`, the figures are `fig_*_I58.png` and the default run directory is `bench_runs/I58`.
- The bench paths default to the repository, runs under `bench_runs/` and data under `data/`, both ignored by git.
  `OCTREG_PROJECT_ROOT` and `OCTREG_DATA_ROOT` override the two, and `OCTREG_I58_DIR` the folder that holds the two I58 input
  files (default `<data root>/I58`). Stored result JSON keeps file names and run paths relative to the repository.

## 1.1.0

A new final stage, §6 smooth deformation (`octreg/deform.py`, docs/METHOD.md §6). §1-5 compute the same affine as 1.0, to
0.003 mm at the block corners on the I58 pair, and the transform files it writes are unchanged. Their code is untouched apart
from docstrings.

- After §5 a small smooth displacement field is fitted on the MRI base grid to two kinds of label-free evidence: block matches
  of a contrast-free structure feature of the interior (`octreg.blockmatch`, now part of the method) and the offset between the
  surface edges of the two volumes along the MRI boundary normals. The edge is the steepest fall of the intensity going
  outwards, found by one rule in the MRI and in the OCT, so that what the rule does to an edge cancels. A boundary point is
  used only when an interior match lies within 5 mm, which keeps the field out of regions without correspondence. The field
    lives on one 5 mm control lattice. Its membrane weight is chosen by held-out error over spatial folds, among the weights of
  a list that keep the strain below
  0.15 and the smallest weight that does, found by bisection. Without a held-out gain of 10 %, with too little evidence, or
    with no weight that keeps the strain limit, no field is applied (flag `deformation_not_supported`). Tried on I58 and
  dropped: the ridge of the bright OCT rim as the boundary
  evidence (the rim is a layer below the surface, so the ridge reads a bias that depends on the face), a search over lattice
  spacings, and further rounds of measuring and refitting (docs/METHOD.md, I58 brainstem pair).
- `octreg/blockmatch.py` now holds the interior evidence of §6 and nothing else: the prior-free robust affine of 1.0.1, tried
  in place of §5 and dropped, is gone from the package, and with it the ablation variant A13.
- New outputs: `oct2mri_warp.nii.gz` (the field, a NIfTI vector image on the MRI base grid, pull-back convention stated in
  `octreg/deform.py`) and `qc_deform.png`, both only when a field is applied, and `oct_in_mri_affine.nii.gz`, which is what
  `oct_in_mri.nii.gz` was before. `oct_in_mri.nii.gz` now goes through the affine and the field. `result.json` gains the block
  `deform` and the time `seconds.deform`. `T_oct2mri.txt`, `T_mri2oct.txt`, `oct2mri.lta`, `oct2mri_itk.txt`, `mri_in_oct.nii.gz`,
  `qc.png`, `qc_montage.png` and `pose` are as in 1.0.
- `octreg apply` uses the field of the run for OCT-frame volumes, `--affine-only` leaves it out, and `--inverse` stays affine
  (the field is not inverted). `geometry.resample_to` takes an optional pull-back field, and `io.save_field` / `io.load_field`
  write and read it.
- `bench/dandi.py` is one protocol instead of several. It crops the MRI to the subject's own Broca-area label box plus
  5 mm, with every side grown to at least the longest side of the OCT array, then calls `octreg register`, the released entry
  point, and evaluates the result against the cortical-layer label the method never reads. The location prior of the
  superseded pipeline, the whole-hemisphere coarse search, the per-subject voxel spacings and the hand-written verdicts are
  gone. Eight of the nine subjects with both modalities are used, and the crop step refuses sub-I56, whose label files have an
  affine that disagrees with their array.
- Every JSON octreg writes (`result.json` and the benchmark's tables) now has LF line ends on every OS, as the transform
  files have had since 1.0.1.
- The shell runner of the I58 pair is gone. It only ran on one server, and `bench/run_i58.py` does the same on any machine.
- The benchmark no longer needs the tree of the superseded pipeline. The ablation variant A0b fed that pipeline's OCT specimen
  mask to the method and is gone. So is the reference column `vs R5 corners max (mm)`, which held the distance to that
  pipeline's pose, and `V1_ROOT` in `bench/paths.py` went with them. `bench/evaluate.py` and `bench/ablate.py` now read the two
  input files and octreg's own runs and nothing else, so anyone who clones the repository can run them. Pose changes in the
  ablation table are measured over the points of the base specimen mask.
- docs/METHOD.md §1 now says that the texture specimen mask needs an embedding to separate, and what to do without one.
- The separable Gaussian and its derivative, which §5 and §6 each carried a byte-identical copy of, now live once in
  `octreg/geometry.py` as `gauss_1d` and `filter_sep`. Identical bytes in, identical bytes out, so no number moves.
- The scale prior of §4 was ablated as one thing and is two. New variants A6p and A6c take the penalty and the clamp apart:
  with λ 0 and the clamp kept the §4 pose moves 16.9 mm and every scale sits on the bound, and with the clamp removed and λ 2
  kept the pose does not move at all. All of the prior is the penalty. The clamp stays as a bound that has never bound on
  the I58 pair, and `clamp_saturated` is how a run says it did.
- `bench/ablate.py --only` used to overwrite the whole table with the rows it ran. It now merges into what is there, and
  refuses to merge rows written under other Params.
- `bench/dandi.py` gains `--tag`, which evaluates a variant run of a subject such as the `_allmeasured` diagnostic against the
  subject's own crop and the mask that run was given. `--steps summary` refuses to write the released table unless every
  subject has a run, so a partial run cannot overwrite it.
- `bench/report.py --store` copies the run's result.json, eval.json and ablations.json into bench/results/I58, and
  the figures step now writes every figure bench/BENCHMARK.md shows, not two of the four. The report is one command again.
- Two stored results of the I58 pair that no document or script named, `first_run` and `no_flattening`, are gone.
  Both were written under Params that no longer exist.
- Corrections to the documents, from reading them against the stored runs. Two steps, not one, sit inside the deletion
  threshold: MRI flattening, and the one-sided outline, which is what holds the block at a cut face.
- §4 now says that the winning pose is the third of the 24 by search score, so refining more than one is load bearing, and
  what that costs.
- The README figure shows the result as directly as it can be shown: per plane the MRI, the registered OCT, and those two
  panels cut into 8 mm squares and interleaved. The checkerboard is nothing but the two of them, on their own grey scales,
  neither inverted nor matched to each other nor masked, so every square can be checked against the panel it came from.
  Nothing is drawn on the data and nothing in the figure is derived from either mask; the script lost a third of its lines
  and one of its inputs with the machinery that used to do all that.
- The QC figures read better and nothing about the transforms changed. The MRI column of `qc.png`, `qc_montage.png` and
  `qc_deform.png` is now mapped by rank onto the grey scale of the OCT in the tissue both show, instead of being stretched
  between its own percentiles, where it used to clip to white over large parts of the specimen. The checkerboard is drawn
  wherever the specimen mask or the MRI foreground reaches and left out only where both call background, which takes the empty
  field out of the panel without hiding anything that could disagree: that support grows as the pose gets worse, and tissue in
  the tiles of one volume against nothing in the tiles of the other stays visible. `bench/fig_registration.py` follows, and
  `bench/dandi.py` cuts the label outlines of `qc_labels.png` to the measured OCT and names the labels in a legend. Two
  alternatives were tried on the I58 pair and dropped: local evening of the two volumes before the checkerboard, and drawing
  the checkerboard only where both maps reach, which is blind to a misplaced block by construction, since the visible
  disagreement inside that support is identically zero at 0, 2 and 5 mm of imposed error.
- The README figure gains the MRI tissue boundary on its registered-OCT panel. The file keeps one outline colour beside its
  greys instead of being flattened to grey.
- New Params `df_*` (METHOD.md, Parameters), so the default Params hash changes from `7d01b8a167a83631` to
  `892a1f3b4fd6f7ed` although §1-5 are untouched. `bench/ablate.py` therefore checks a prep against the §1 fields only
  (`prep_hash`) and accepts the preps of 1.0. It runs §6 for base and stores its read-outs, and `bench/report.py` prints them.

## 1.0.1

No change to the method or its results: same Params and hash, same `result.json` keys, same poses.

- Windows portability. The transform files (`T_oct2mri.txt`, `T_mri2oct.txt`, `oct2mri.lta`, `oct2mri_itk.txt`) are written with
  LF line ends on every OS. `peak_rss_gb` in `result.json` is filled on Windows too (the peak working set;
  `octreg.register.peak_rss_gb`). The benchmark reads its machine roots from `bench/paths.py`, and a Python runner of the I58 pair,
  now `bench/run_i58.py`, works on any OS next to the POSIX shell runner of this release.
- `octreg/blockmatch.py`, at this release a diagnostic the method does not call. It measures a label-free residual displacement
  field between the MRI and the registered OCT from local fine structure. A robust affine fitted to this field alone was tested
  in place of §5 on the I58 pair and not adopted. `bench/ablate.py` documents it as variant A13. Both the robust affine and
  A13 were removed in 1.1.0, where what is left of the module became the interior evidence of §6.

## 1.0.0

First public release: texture specimen mask, two-class score with the outline term, orientation search in the MRI crop,
prior-bounded affine refinement, fine-structure refinement by NGF descent, handedness of array frames.

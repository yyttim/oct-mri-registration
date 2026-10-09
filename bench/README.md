# bench

Benchmark of octreg 2.0 on the I58 brainstem pair, and a script for the DANDI:000026 Broca-area blocks. It lives in the
repository but is not part of the package, and the method never reads anything here. The I58 pair has no labels, so its
metrics are label-free.

The scripts find their files through `bench/paths.py`. By default the runs go to `bench_runs/` and the data are read from
`data/` of the repository, and both are in `.gitignore`. `OCTREG_PROJECT_ROOT` moves the runs to `<project root>/bench_runs`,
`OCTREG_DATA_ROOT` moves the data root, which holds the I58 folder and the DANDI tree (expected at
`<data root>/costantini/dandi-000026`, see `bench/paths.py`), and `OCTREG_I58_DIR` names the folder
that holds the two I58 input files (by default `<data root>/I58`). Nothing else here is machine-specific.

| file | what it does |
|---|---|
| `run_i58.py` | the runner for the I58 pair: the steps register, ablate, evaluate and report, their overrides and log files, on any OS |
| `paths.py` | the run and data roots and the input files derived from them |
| `evaluate.py` | one run measured against the data itself: boundary agreement, OCT mask volume, raw-data frame check, and the pose distance to an earlier octreg run |
| `ablate.py` | the preprocessing once, then the variants base and A0-A12, one JSON table |
| `ablate_deform.py` | §6 from the cached base grids of `ngf_lam.py cache`: the ablation table of METHOD.md §6 plus rows that widen the reach of the local evidence (df_reach_mm, the interior search range and the edge window), every row scored by the same measurement, and the boundary residual along the sectioning axis |
| `ngf_lam.py` | §5 alone from cached base grids: `cache` writes the §1 grids both volumes share, `sweep` refits §5 from the §4 pose of a run at several `lam` values or σ schedules (`--destripe` flat-fields the section stripes first) and scores each pose by the outline agreement of `evaluate.py`, `transforms` writes every pose of a sweep as a 4x4 text file for `octreg qc --T` |
| `dandi.py` | the DANDI blocks: MRI crop, `octreg register`, the cortical-layer read-out and its figure, a markdown summary (results withdrawn, see below) |
| `report.py` | writes `bench/BENCHMARK.md` from the outputs, and the qc figures it describes into `bench/figures/` |
| `compose_pair.py` | two `octreg qc` outputs of one run side by side (`bench/figures/fig_handedness_I58.png`) |
| `fig_registration.py` | the registration figure: an MRI plane, the OCT placed on it by the run, and those two panels cut into 8 mm squares and interleaved |
| `BENCHMARK.md` | the I58 report, written by `report.py`, which keeps its hand-written "Visual result" section and its closing reading |
| `results/` | the JSON copied out of the run directories |
| `figures/` | the figures of the I58 pair, which these scripts draw locally and which are not published |

## Running the I58 benchmark

`bench/run_i58.py` runs the steps from the repository root, with the interpreter that runs it. The whole benchmark is
`register ablate evaluate report`, with `evaluate` run twice: the mask and boundary metrics need the preprocessing that the
ablate step writes.

```bash
export OUT=/new/dir/bench_runs/I58/myrun/main               # a directory that holds no run yet
export ABL=/new/dir/bench_runs/I58/myrun/ablate
STEPS="register evaluate" python bench/run_i58.py           # the registration, then the metrics that need no masks
STEPS="ablate evaluate report" python bench/run_i58.py      # the ablations, then all metrics, then BENCHMARK.md
```

PowerShell sets the same variables with `$env:OUT = "...\myrun\main"`. To detach: `setsid nohup ... &` on POSIX,
`Start-Process -WindowStyle Hidden python bench/run_i58.py` on Windows.

`register` is `python -m octreg register OCT MRI -o OUT` on the two original files, with default parameters. The other
overrides are `PREV` (the `ablations.json` of the run that measured the removed steps A3 and A7, by default
`bench_runs/I58/ablate/ablations.json`, the first ablation run), `PREV_MAIN` (an earlier run for the pose distance,
by default `bench_runs/I58/rel7/main`, release 1.0, which the stored evaluation used), `DEVICE` (`cuda` or `cpu`) and
`CODE` (the repository to run from, by default the one this file is in). `OUT` and `ABL` default to the released run,
`v2/main` and `v2/ablate`, so a new run has to set both. Every step writes `NAME.log`, `NAME.time` (wall time and exit
status, the peak memory of a run is in its own result.json) and `NAME.gpu_mib` (nvidia-smi samples, when nvidia-smi is on
the PATH) into `${OUT}_logs`, and a start and a done line per step (and the last 5 log lines on failure) into `chain.log`
there.

Two rules. One registration job at a time: the runner refuses to start while another registration python is running. And
`register` and `ablate` refuse to write into their default `OUT` and `ABL` once those hold a run, so a new run needs a new
`OUT` and a new `ABL`. `report` has no such guard: it rewrites `bench/BENCHMARK.md` from whatever `OUT` and `ABL` name, which
without an override is the released run.

`python bench/evaluate.py --selftest` checks the frame check, the boundary agreement, the pose distance and the mask volume on
synthetic arrays in a few seconds, on the CPU and without any of the data.

The released run is `bench_runs/I58/v2`, evaluated against `rel7/main` (release 1.0, 0.001 mm away at the block corners,
with release 1.1 within 0.003 mm). The report step copies its `result.json`, `eval.json` and `ablations.json`
into `bench/results/I58/` (`report.py --store`, with every absolute path cut to a file name or a `bench_runs/` path) and
writes `BENCHMARK.md` and `bench/figures` from them, so the document and the numbers it quotes are one command and never
drift apart. `deform_ablation.json` and `.md` there are the output of `bench/ablate_deform.py`, copied by hand with the two
run paths cut as report.py cuts them. `ngf_lam/lam.json`, `sigmas.json` and `destripe.json` are the sweeps of
`bench/ngf_lam.py sweep` that docs/METHOD.md §5 quotes, with the run and cache paths cut the same way.

## The DANDI:000026 blocks

`bench/dandi.py` runs the method on the Broca-area blocks of DANDI:000026 (Costantini et al. 2023), public CC-BY 4.0 data.
Nine subjects have both an OCT block (OME-TIFF of the scattering coefficient, no orientation in its header, voxel spacing
from the BIDS sidecar) and an ex-vivo MRI of the hemisphere it was cut from. Eight are used. sub-I56 is refused: the affine
of every one of its label files disagrees with its array, so only 30 to 72 % of each label lands inside the MRI and neither
the crop nor the evaluation can be read from it.

The script was run on the eight subjects, and those results are withdrawn until a re-run: it built the MRI crops of six of
them through label headers that do not match the MRI, and only sub-I46 and sub-I55 were placed correctly.

```bash
python bench/dandi.py [SUBJ ...] [--steps crop register evaluate summary] [--device cuda] [--out DIR]
```

The default output root is `bench_runs/dandi`, one directory per subject.

The MRI is one EPIC `*_VFA.nii.gz` asset per subject, pinned in `MRI_FLIP` of `bench/dandi.py` and not chosen by a glob or by
flip angle, so every run reads the same file whatever else a download holds: `flip-2` for sub-I38, sub-I46, sub-I48 and
sub-I55, `flip-3` for sub-I58 and sub-I62, `flip-4` for sub-I57 and sub-I61. The sidecars give no rule for this choice
(sub-I48 has no sidecar, and sub-I58's pinned `flip-3` has FlipAngle 10 against 30 for its `flip-1`), so the map is explicit.

- `crop`: the method assumes an MRI already cropped around the block, so the MRI is cut to the bounding box of the subject's own
  Broca-area label plus 5 mm, with every side grown to at least the longest side of the OCT array, and written to
  `OUT/SUBJ/mri_crop.nii.gz`. That label says which part of the hemisphere the block was taken from, which is what a user of
  the method knows. It is not the registration: it fixes neither the orientation nor the position inside the crop.
- `register`: `octreg register` on the block and the crop, the released entry point. An OME-TIFF carries no orientation, so
  its array frame must have the handedness of the specimen (README, Usage), and §1-6 run as for any other pair.
- `evaluate`: a number the registration never sees, written to `OUT/SUBJ/eval.json`. The subject's cortical-layer label divides
  the cortex into layers. Its two ends, the classes with the lowest and the highest median MRI intensity, are brought into the
  OCT by the pose, and the two-class map of §2 is read there. The separation |AUC - 0.5| of the two sets of values says whether
  the registered OCT tells the two ends of the cortex apart, and does not depend on which modality is bright where. The same
  number at the pose of §4 and at 10 random poses of the block inside the crop says what it is worth on that subject. A random
  pose that lands fewer than 500 labelled points inside the specimen mask, or fewer than 100 in a class, is not scored, so the
  median and the max are over the scored ones, and `random_separation["n"]` of `eval.json` says how many.
  It also writes `qc_labels.png`, the registered OCT with the outlines of that label: where the pose is right they follow
  the bright and dark bands of the OCT.
- `summary`: prints one markdown row per subject that has a run: crop size, the scores of §4, F, the scale per OCT axis, §6,
  and the layer separation next to the random poses, whose column carries the number of poses actually scored, and writes
  the same as JSON to `OUT/summary.json`.

Visual inspection stays the evaluation (docs/METHOD.md): `qc.png`, `qc_montage.png` and `qc_labels.png` of every run are the
primary evidence and these numbers only support them.

A block can also be registered with the OCT given as its own mask, which is the mask a block without embedding should have.
The registration is the released command with one more argument, since a mask file is read as its positive voxels, and the
read-out is the evaluate step with a tag:

```bash
python -m octreg register OCT OUT/SUBJ/mri_crop.nii.gz -o OUT/SUBJ_allmeasured     --oct-spacing-um Z,Y,X --oct-mask OCT
python bench/dandi.py SUBJ --steps evaluate --tag _allmeasured --out OUT
```

The second command writes `OUT/SUBJ_allmeasured/eval.json`, the read-out of that run, taken over the mask the run was given.

## What is measured

The I58 pair has no labels, so `evaluate.py` measures the run against the data itself. It reads the two input files, the run
directory, the masks of an `ablate.py` prep directory (`--masks`) and, for the comparison, a second octreg run (`--previous`).
Nothing else. `T = RUN/T_oct2mri.txt` maps the OCT header world to the MRI header world.

- Boundary agreement: the OCT specimen-mask outline through the pose to the MRI foreground outline and back, median per face.
  "rim" leaves out the deep end of the block (raw axis 0, high index), which has no specimen rim. Cut faces of either field of
  view are excluded. With `--previous` the earlier run's rim medians are measured under the same masks, so the two runs are
  compared on the same footing.
- OCT specimen mask: the volume of `oct_mask.nii.gz` in cm3.
- Pose distance to an earlier octreg run (`--previous`, the runner passes `PREV_MAIN`): mean and max displacement over the
  points of the run's own OCT specimen mask, the same at the 8 block corners, and the rotation angle between the two poses.
  Without `--masks` the mean is over the 8 corners of the OCT array plus a uniform sample of its grid instead.
- Raw-data frame check: raw 20 um values streamed from the OCT `.nii.gz` through the header affine and the pose, against the
  exported affine overlay `oct_in_mri_affine.nii.gz`, or `oct_in_mri.nii.gz` for a run before 1.1 (Spearman, 7^3 voxel boxes).
  Controls flip each raw OCT axis or shift the pose by 2 mm. It passes when the pose gives at least 0.9 and every flip at most
  0.3.

Masks come from `ablate.py` (`ABL/prep/texture/`), so `evaluate` reports the mask and boundary metrics only after the ablate
step. The pose distance and the frame check need no masks.

`ablate.py` reads the same two input files and its own runs, and nothing else. Its pose changes are measured over the points of
the base specimen mask and over the 8 corners of the OCT array.

## Ablations

| name | change |
|---|---|
| base | the method (default Params) through the ablation driver |
| A0 | OCT intensity foreground (histogram valley) instead of the texture specimen mask |
| A0c | the texture mask with holes filled in 3-D only instead of in every array plane |
| A1, A2 | MRI flattening off, OCT flattening off (`two_class(..., flatten=False)`) |
| A4 | standardised intensity channels (z, -z) instead of two-class maps |
| A5+1, A5-1 | polarity forced to +1 or -1 instead of the sign of the two-class score (`align(..., polarity=...)`) |
| A6 | no scale prior: lambda 0 and clamp 1.0 instead of 2 and 0.15 |
| A6p, A6c | the penalty alone (lam 0, clamp kept) and the clamp alone (clamp 1.0, lam kept) |
| A8 | the other handedness: the OCT world mirrored (z negated) before search and refinement |
| A9 | no outline term: S = 2 S_class / 3 in the search (patched `search.combined`) and in the refinement (outline weight = the specimen mask, so S_outline = 0) |
| A10 | two-sided outline: OCT embedding over MRI tissue or outside the crop counted as a mismatch |
| A11, A11b | simulated cut face (specimen mask removed beyond 70 % of its extent along OCT axis 1, data kept as embedding), with the method and with the two-sided outline |
| A12 | no fine-structure refinement: the pose of §4 (every other variant ends with §5) |
| A3, A7 | removed steps (the section-stripe flat field, and the rigid, similarity and affine ladder), rows copied from the first ablation run through `PREV`, see below |

`Params` holds method constants only. Each variant is one explicit change made in the driver around the package functions the
method itself uses: an argument of `register.fine_mask`, `preprocess.two_class` or `register.align`, a Params override (lam 0
in A6 and A6p), or `refine.CLAMP` patched for the variant (clamp 1.0 in A6 and A6c: the clamp is a module constant, not a
Params field, so `ablate.py` sets it around the call and restores it). A0 takes its OCT mask from `preprocess.foreground`, A4
builds its channels in `ablate.py`, and A0c, A9 and A10 temporarily replace `preprocess._fill_planes` (with the 3-D
`ndimage.binary_fill_holes`), `search.combined` and the outline of the search and of the refinement. The method has no
ablation switches. The OCT is streamed once and each OCT mask source (texture, texture3d, intensity) is computed once and
cached under `ABL/prep/`. The driver's base run is compared with the CLI run ("driver check", expected at float precision).
Pose changes are measured against base. The boundary agreement of every variant uses the base masks, so it reflects the pose
only. A variant that makes the method refuse is recorded as failed with the message. Finished variants are reused on a rerun
unless `--force` is given.

The smooth deformation (§6) does not change the pose, so the driver does not run it and no variant is "no §6": its read-outs
are the residuals before and after the field in the run's own result.json, which report.py prints. The read-outs are the
lattice spacing, the membrane weight (the smallest in [0.3, 30] that keeps the strain under the limit) and the strain it
reaches, the interior matches and the boundary points of the fit, the held-out errors without and with the field, which gate
the field, and the residuals of both kinds (interior matches, surface-edge offsets) measured on the affine OCT and again on the
warped one. The ablations of §6 itself (one kind of evidence alone, the rim ridge in place of the edge, the support rule of
1.1, no Huber re-weighting, other lattice spacings, strain limits and reaches) are tabulated in docs/METHOD.md. They are made
by `bench/ablate_deform.py` from the base grids cached by `bench/ngf_lam.py cache`, at the affine of the released run of 1.1
(which 2.0 reproduces within 0.003 mm at the block corners), and the table is `bench/results/I58/deform_ablation.md`. Since 1.1 a
prep is checked against the Params fields that §1 reads (`prep_hash` in `prep.json`), so new Params of later stages leave it
valid, and the preps of 1.0 are accepted.

A3 and A7 were run in the first ablation run, with the earlier two-class score, the overlap gate and mirrored orientations in
the search (`bench_runs/I58/ablate/ablations.json`, the default `PREV`), against a base that still had the flat field
and the ladder. Neither moved the pose by more than 0.5 mm, so both steps were deleted. `ablate.py --previous` copies their rows
and reports how far the present base lies from that earlier base.

The rows appear only when `PREV` points at that first run, which is the default. A `--previous` file that holds neither step
makes `ablate.py` print a warning naming the variants it did find, so the rows cannot go missing unremarked.

The deletion rule: a step whose removal moves the pose by at most 0.5 mm (mean over the specimen-mask points, at the pose after
§4 for steps of §1-4 and at the final pose for §5) and changes no metric beyond noise is deleted before release, unless a test
outside this pair shows it load bearing. Three steps within that bound are kept: OCT flattening (A2, 0.15 mm) and per-plane
hole filling (A0c, 0.22 mm), for the reasons given in docs/METHOD.md after the ablation table, and the scale clamp (A6c,
0.00 mm), which is a guard rather than a parameter (docs/METHOD.md §4).

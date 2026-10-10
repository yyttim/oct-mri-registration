# bench

The benchmark of octreg on the I58 brainstem pair: the scripts that run and measure it, and its stored results. The
directory is not part of the package, and the method never reads anything here. The I58 pair has no labels, so every
metric is label-free.

## Data layout

The scripts find their files through `bench/paths.py`. By default the runs go to `bench_runs/` and the data are read from
`data/` of the repository, and both are in `.gitignore`. Three environment variables move them.

| variable | meaning | default |
|---|---|---|
| `OCTREG_PROJECT_ROOT` | the runs go to `<project root>/bench_runs` | the repository root |
| `OCTREG_DATA_ROOT` | the data root | `<repository>/data` |
| `OCTREG_I58_DIR` | the folder that holds the two I58 input files | `<data root>/I58` |

The two input files are `I58_Brainstem_mus_Slice_full_20um_corr.nii.gz` (the OCT) and
`I58_brainstem_MRI_cropped_to_OCT.nii.gz` (the MRI crop). They are not part of the repository. Nothing else here is
machine-specific.

The scripts read the pair through `OCT_I58` and `MRI_I58` of `bench/paths.py`, so another pair runs once those two names
point at its files. Two read-outs are specific to this block. The rim of the outline agreement leaves out the deep end of
raw OCT axis 0, and the simulated cut face of A11 lies along OCT axis 1.

## Scripts

| file | what it does |
|---|---|
| `run_i58.py` | runs the steps register, ablate, evaluate and report on the I58 pair, with a log per step, on any OS |
| `paths.py` | the run and data roots and the two input files |
| `evaluate.py` | measures one run against the data: raw-data frame check, outline agreement, OCT specimen mask volume, and on request the pose distance to a second run |
| `ablate.py` | preprocesses the pair once and runs base and the variants of the table below, written as one JSON table |
| `ablate_deform.py` | the §6 ablation table of docs/METHOD.md from the cached base grids of `ngf_lam.py cache`, every row scored by the same measurement |
| `ngf_lam.py` | §5 alone on cached base grids: `cache` writes the §1 grids, `sweep` refits §5 at several λ values or σ schedules, `transforms` writes every pose of a sweep as a 4x4 text file for `octreg qc --T` |
| `report.py` | writes `bench/BENCHMARK.md` from the run outputs, and on request the qc figures and the stored results |
| `compose_pair.py` | two `octreg qc` outputs of one run side by side (`bench/figures/fig_handedness_I58.png`) |
| `fig_registration.py` | the registration figure: an MRI plane, the OCT placed on it by the run, and the two cut into 8 mm squares and interleaved |
| `BENCHMARK.md` | the I58 report, written by `report.py` around two hand-written sections |
| `baselines/` | the comparison with standard registration tools, its scripts and [BASELINES.md](baselines/BASELINES.md) |
| `results/I58/` | the stored results of the benchmark (below) |
| `figures.py` | the freeview figures of the I58 pair in `docs/figures/`, shown in README.md, docs/METHOD.md and BASELINES.md |
| `figures/` | the default place for the qc figures of `report.py --figures`, not tracked by git |

## Running the I58 benchmark

`bench/run_i58.py` runs its steps from the repository root, with the interpreter that runs it. The whole benchmark is
`register ablate evaluate report`, with `evaluate` run twice, since the mask and outline metrics need the preprocessing
that the ablate step writes.

```bash
STEPS="register evaluate" python bench/run_i58.py           # the registration, then the metrics that need no masks
STEPS="ablate evaluate report" python bench/run_i58.py      # the ablations, then all metrics, then BENCHMARK.md
```

PowerShell sets the same variables with `$env:STEPS = "register evaluate"`. The steps are:

- `register`: `python -m octreg register OCT MRI -o OUT` on the two original files, with default parameters.
- `ablate`: `bench/ablate.py --out ABL`, which also compares its base with the run in `OUT`.
- `evaluate`: `bench/evaluate.py OUT`, with the masks of `ABL/prep/texture` once the ablate step has written them.
- `report`: `bench/report.py`, which rewrites `bench/BENCHMARK.md`, `bench/figures` and `bench/results/I58` from `OUT`
  and `ABL`.

The overrides are all optional. `OUT` and `ABL` name the run and ablation directories (by default `bench_runs/I58/octreg` and
`bench_runs/I58/octreg_ablate`). `PREV_MAIN` names a second octreg run, and when it is set the evaluate step also measures the
pose distance to it. `DEVICE` is `cuda` or `cpu`, and `CODE` is the repository to run from (by default the one that holds the
runner).

Every step writes `NAME.log`, `NAME.time` (wall time and exit status) and `NAME.gpu_mib` (nvidia-smi samples, when
nvidia-smi is on the PATH) into `${OUT}_logs`, and a start and a done line per step (and the last 5 log lines on failure)
into `chain.log` there. The peak memory of a run is in its own result.json.

The runner refuses to start while another `python -m octreg`, `bench/ablate.py` or `bench/evaluate.py` process runs, so that one
job holds the GPU at a time. `register` and `ablate` refuse a default `OUT` or `ABL` that already holds a run, so a second run
needs a new `OUT` and a new `ABL`. `report` has no such guard. It overwrites `bench/BENCHMARK.md` and `bench/results/I58` with
the run that `OUT` and `ABL` name.

`python bench/evaluate.py --selftest` checks the frame check, the outline agreement, the pose distance and the mask volume
on synthetic arrays in a few seconds, on the CPU and without any of the data.

The §5 sweeps and the §6 ablations run on base grids cached once:

```bash
python bench/ngf_lam.py cache --oct OCT --mri MRI -o CACHE
python bench/ngf_lam.py sweep --cache CACHE --run RUN -o OUT.json             # --vary sigmas
python bench/ablate_deform.py --cache CACHE --run RUN -o OUT.json             # writes OUT.json and OUT.md
```

## Stored results

`bench/results/I58/` holds the stored outputs of the benchmark, which the documents quote. Absolute paths in these files
are cut to a file name or a path under `bench_runs/`.

| file | content |
|---|---|
| `result.json`, `eval.json` | the output of `octreg register` and of `bench/evaluate.py` for the benchmark run |
| `ablations.json` | the ablation table of `bench/ablate.py` |
| `deform_ablation.json`, `.md` | the §6 ablation table of `bench/ablate_deform.py` |
| `baselines/` | the summaries of the comparison with standard registration tools |

The report step writes `result.json`, `eval.json` and `ablations.json` with `report.py --store` in the same command that
writes `BENCHMARK.md`, so the document and the stored numbers agree. `deform_ablation.json` and `.md` are the output of
`bench/ablate_deform.py`. The §6 ablation table was computed at the affine of a run whose pose lies within
0.003 mm of the benchmark run at the block corners.

## What is measured

The I58 pair has no labels, so `evaluate.py` measures the run against the data itself. It reads the two input files, the
run directory, the masks of an `ablate.py` prep directory (`--masks`) and, on request, a second octreg run (`--previous`).
Nothing else. `T = RUN/T_oct2mri.txt` maps the OCT header world to the MRI header world.

- Outline agreement: the outline of the OCT specimen mask through the pose to the outline of the MRI foreground, and back,
  as the median per face. "rim" leaves out the deep end of the block (raw axis 0, high index), which has no specimen rim.
  Cut faces of either field of view are excluded. With `--previous` the rim medians of the second run are measured under
  the same masks.
- OCT specimen mask: the volume of `oct_mask.nii.gz` in cm3.
- Pose distance to a second octreg run (`--previous`): mean and max displacement over the points of the run's own OCT
  specimen mask, the same at the 8 block corners, and the rotation angle between the two poses. Without `--masks` the mean
  is over the 8 corners of the OCT array plus a uniform sample of its grid.
- Raw-data frame check: raw 20 um values streamed from the OCT `.nii.gz` through the header affine and the pose, against
  the exported affine overlay `oct_in_mri_affine.nii.gz` (Spearman, 7^3 voxel boxes). Controls flip each raw OCT axis or
  shift the pose by 2 mm. It passes when the pose gives at least 0.9 and every flip at most 0.3.

The masks come from `ablate.py` (`ABL/prep/texture/`), so `evaluate` reports the mask and outline metrics only after the
ablate step. The pose distance and the frame check need no masks.

`ablate.py` reads the same two input files and its own runs, and nothing else. Its pose changes are measured over the
points of the base specimen mask and over the 8 corners of the OCT array.

## Ablations

| name | change |
|---|---|
| base | the method (default Params) run through `bench/ablate.py` |
| A0 | OCT intensity foreground (histogram valley) instead of the texture specimen mask |
| A0c | the texture mask with holes filled in 3-D only instead of in every array plane |
| A1, A2 | MRI flattening off, OCT flattening off (`two_class(..., flatten=False)`) |
| A4 | standardised intensity channels (z, -z) instead of two-class maps |
| A5+1, A5-1 | polarity forced to +1 or -1 instead of the sign of the two-class score (`align(..., polarity=...)`) |
| A6 | no scale prior: λ 0 and clamp 1.0 instead of 2 and 0.15 |
| A6p, A6c | no penalty, clamp kept (λ 0, clamp 0.15) and no clamp, penalty kept (λ 2, clamp 1.0) |
| A8 | the other handedness: the OCT world mirrored (z negated) before search and refinement |
| A9 | no outline term: S = 2 S_class / 3 in the search (patched `search.combined`) and in the refinement (outline weight = the specimen mask, so S_outline = 0) |
| A10 | two-sided outline: OCT embedding over MRI tissue or outside the crop counted as a mismatch |
| A11, A11b | simulated cut face (specimen mask removed beyond 70 % of its extent along OCT axis 1, data kept as embedding), with the method and with the two-sided outline |
| A12 | no fine-structure refinement: the pose of §4 (every other variant ends with §5) |

`Params` holds method constants only, and the method has no ablation switches. Each variant is one explicit change made in
`ablate.py` around the package functions the method itself uses: an argument of `register.fine_mask`,
`preprocess.two_class` or `register.align`, a Params override (`lam` 0 in A6 and A6p), or `refine.CLAMP` patched for the
variant (clamp 1.0 in A6 and A6c). The clamp is a module constant, not a Params field, so `ablate.py` sets it around the
call and restores it. A0 takes its OCT mask from `preprocess.foreground`, A4 builds its channels in `ablate.py`, and A0c,
A9 and A10 temporarily replace `preprocess._fill_planes` (with the 3-D `ndimage.binary_fill_holes`), `search.combined` and
the outline of the search and of the refinement.

The OCT is streamed once, and each OCT mask source (texture, texture3d, intensity) is computed once and cached under
`ABL/prep/`. A prep is checked against the Params fields that §1 reads (`prep_hash` in `prep.json`), so it stays valid
when the Params of later steps change. The pose distance of base to the CLI run (`--main`) is stored as a consistency
check. Pose changes are measured against base. The outline agreement of every variant uses the base masks, so it reflects
the pose only. A variant that makes the method refuse is recorded as failed with the message. Finished variants are reused
on a rerun unless `--force` is given.

The smooth deformation (§6) does not change the affine, so `ablate.py` does not run it. Its read-outs are in the run's
result.json, which `report.py` prints in the Main result of `BENCHMARK.md`. `bench/ablate_deform.py` ablates the elements
of §6, and docs/METHOD.md and `bench/results/I58/deform_ablation.md` give the table.

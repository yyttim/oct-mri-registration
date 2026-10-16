# bench

This directory holds the scripts that run and measure octreg, and their stored results. It is not part of the package. The
I58 brainstem pair has no labels and no reference registration, so it is shown in pictures and every number on it is a
label-free read-out. The test in [i46/](i46/README.md) scores octreg and the standard tools on block I46 of Costantini et al.
(2023), the public data of the work octreg extends, against the authors' registration.

## Data layout

The scripts find their files through `bench/paths.py`. By default the runs go to `bench_runs/`, the inputs, runs and summaries
of the baseline comparison to `baselines/`, and the data are read from `data/` of the repository. All three are in
`.gitignore`. Four environment variables move them.

| variable | meaning | default |
|---|---|---|
| `OCTREG_PROJECT_ROOT` | the runs go to `<project root>/bench_runs`, the baseline comparison to `<project root>/baselines` | the repository root |
| `OCTREG_DATA_ROOT` | the data root | `<repository>/data` |
| `OCTREG_I58_DIR` | the folder that holds the two I58 input files | `<data root>/I58` |
| `OCTREG_DANDI000026_DIR` | the download of DANDI:000026 sub-I46 and the authors' registration files for `i46/` | `<data root>/dandi-000026` |

The two input files are `I58_Brainstem_mus_Slice_full_20um_corr.nii.gz` (the OCT) and
`I58_brainstem_MRI_cropped_to_OCT.nii.gz` (the MRI crop). They are not part of the repository.

The scripts read the pair through `OCT_I58` and `MRI_I58` of `bench/paths.py`, so another pair runs once those two names
point at its files. Two read-outs are specific to this block. The rim of the outline agreement leaves out the deep end of
raw OCT axis 0, and the simulated cut face of A11 lies along OCT axis 1.

## Scripts

| file | what it does |
|---|---|
| `run_i58.py` | runs the steps register, ablate, evaluate and report on the I58 pair, with a log per step, on any OS |
| `paths.py` | the run and data roots and the two input files |
| `evaluate.py` | measures one run against the data: raw-data frame check, outline agreement, OCT specimen mask volume, and on request the pose distance to a second run |
| `ablate.py` | preprocesses the pair once and runs base and the variants into one JSON table, and with `--starts` and `--random-starts N` runs the method and the variant without the orientation search from turned starts |
| `ablate_deform.py` | the ablations of §6 on the cached base grids of `ngf_lam.py cache`, each row measured by the read-outs of §6 on the warped OCT, and on request the fields of chosen rows as run directories |
| `ngf_lam.py` | §5 alone on cached base grids: `cache` writes the §1 grids, `sweep` refits §5 at several λ values or σ schedules, `transforms` writes every pose of a sweep as a 4×4 text file for `octreg qc --T` |
| `report.py` | writes `bench/BENCHMARK.md` from the run outputs, and on request the qc figures and the stored results |
| `compose_pair.py` | a figure of two `octreg qc` outputs of one run side by side, for example the result and the best pose of the other handedness (A8) |
| `fig_registration.py` | a registration figure on three MRI planes: the MRI, the OCT placed on it by the run, and the two cut into 8 mm squares and interleaved |
| `figures.py` | the freeview figures of the I58 pair in `docs/figures/`, shown in README.md and BASELINES.md |
| `BENCHMARK.md` | the I58 report, written by `report.py` around two hand-written sections |
| `baselines/` | the comparison with standard registration tools, its scripts and [BASELINES.md](baselines/BASELINES.md) |
| `i46/` | the test on block I46 of Costantini et al. (2023), its scripts and [README](i46/README.md) |
| `results/I58/` | the stored results of the benchmark (below) |
| `figures/` | where the report step of `run_i58.py` writes the qc figures (`report.py --figures`), not tracked by git |

The scripts print their options with `--help`.

## Running the I58 benchmark

`bench/run_i58.py` runs the steps named in `STEPS` (default `register evaluate`) from the repository root, with the
interpreter that runs it. `STEPS="register ablate evaluate report"` runs the whole benchmark. The mask and outline metrics
need the preprocessing that the ablate step writes, so the two commands below run `evaluate` after the registration and again
after the ablations, when it rewrites `OUT/eval.json` with all the metrics.

```bash
STEPS="register evaluate" python bench/run_i58.py           # the registration, then the metrics that need no masks
STEPS="ablate evaluate report" python bench/run_i58.py      # the ablations, then all metrics, then BENCHMARK.md
```

PowerShell sets the same variables with `$env:STEPS = "register evaluate"`. The steps are:

- `register`: `python -m octreg register OCT MRI -o OUT` on the two original files, with default parameters.
- `ablate`: `bench/ablate.py --out ABL --starts`, which also runs the start test (below) and compares its base with the run in
  `OUT`.
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

The runner does not start while another `python -m octreg`, `bench/ablate.py` or `bench/evaluate.py` process runs, so that
one job holds the GPU at a time. `register` and `ablate` refuse a default `OUT` or `ABL` that already holds a run, so a
second run needs a new `OUT` and a new `ABL`. `report` overwrites `bench/BENCHMARK.md` and `bench/results/I58` with the run
that `OUT` and `ABL` name.

`python bench/evaluate.py --selftest` checks the frame check, the outline agreement, the pose distance and the mask volume
on synthetic arrays in a few seconds, on the CPU and without any of the data.

The §5 sweeps and the §6 ablations run on base grids cached once:

```bash
python bench/ngf_lam.py cache --oct OCT --mri MRI -o CACHE.npz
python bench/ngf_lam.py sweep --cache CACHE.npz --run RUN -o OUT.json         # --vary sigmas
python bench/ablate_deform.py --cache CACHE.npz --run RUN -o OUT.json         # writes OUT.json and OUT.md
```

The ablation figures in `docs/figures/ablation/` come from the output of `ablate.py --starts`, which `figures.py --ablation`
reads:

```bash
python bench/ablate.py --out ABL --main OUT --starts
python bench/figures.py --octreg-run OUT --ablation ABL
```

`figures.py` makes the I58 panels of `docs/figures` in one run (`--stages` makes some of them). It needs the baseline run
directories of `bench/baselines` (under `<project root>/baselines`, or `--baselines DIR`, and `--deform-baselines DIR` for
deformable runs that started from the affine of the octreg run shown) and FreeSurfer 8.2 (freeview, mri_mc, mris_smooth) with
its licence, in WSL on Windows.

## Stored results

`bench/results/I58/` holds the stored outputs of the benchmark, which the documents quote. Absolute paths in these files
are cut to a file name or a path under `bench_runs/`.

| file | content |
|---|---|
| `result.json`, `eval.json` | the output of `octreg register` and of `bench/evaluate.py` for the benchmark run |
| `ablations.json`, `starts.json` | the ablation table and the start test of `bench/ablate.py` |
| `deform_ablation.json`, `.md` | the §6 ablation table of `bench/ablate_deform.py`, computed at the affine of the benchmark run |
| `baselines/` | the summaries, run notes and tables of the comparison with standard registration tools |

`bench/results/I46/` holds the results of the I46 test and its tables (`tables.md`), written by the scripts of
[i46/](i46/README.md).

The report step writes `result.json`, `eval.json`, `ablations.json` and `starts.json` with `report.py --store` in the same
command that writes `BENCHMARK.md`. When the step logs in `${OUT}_logs` exist, the Main result of `BENCHMARK.md` also gives the
wall clock and the GPU memory in use on the whole device. These two are not stored.

## What is measured

`evaluate.py` measures a run against the data. `T = RUN/T_oct2mri.txt` maps the OCT header world to the MRI header world.

- Outline agreement: the outline of the OCT specimen mask through the pose to the outline of the MRI foreground, and back,
  as the median per face. "rim" leaves out the deep end of the block (raw axis 0, high index), which has no specimen rim.
  Cut faces of either field of view are excluded. With `--previous` the rim medians of the second run are measured under
  the same masks.
- OCT specimen mask: the volume of `oct_mask.nii.gz` in cm³.
- Pose distance to a second octreg run (`--previous`): mean and max displacement over the points of the run's own OCT
  specimen mask, the same at the 8 block corners, and the rotation angle between the two poses. Without `--masks` the mean
  is over the 8 corners of the OCT array plus a uniform sample of its grid.
- Raw-data frame check: raw 20 µm values streamed from the OCT `.nii.gz` through the header affine and the pose, against
  the exported affine overlay `oct_in_mri_affine.nii.gz` (Spearman, 7^3 voxel boxes). Controls flip each raw OCT axis or
  shift the pose by 2 mm. It passes when the pose gives at least 0.9 and every flip at most 0.3.

The outline agreement and the mask volume use the masks of `ablate.py` (`--masks ABL/prep/texture`), so `evaluate` reports
them only after the ablate step.

## Ablations

| name | change |
|---|---|
| base | the method (default Params) run through `bench/ablate.py` |
| A0 | OCT intensity foreground (histogram valley) instead of the texture specimen mask |
| A0c | the texture mask with holes filled in 3-D only instead of in every array plane |
| A1, A2 | MRI flattening off, OCT flattening off |
| A4 | standardised intensity channels (z, -z) instead of two-class maps |
| A5+1, A5-1 | polarity forced to +1 or -1 instead of the sign of the two-class score |
| A6 | no scale prior: λ 0 and clamp 1.0 instead of 2 and 0.15 |
| A6p, A6c | no penalty, clamp kept (λ 0, clamp 0.15) and no clamp, penalty kept (λ 2, clamp 1.0) |
| A8 | the other handedness: the OCT world mirrored (z negated) before search and refinement |
| A9 | no outline term: S = 2 \|S_class\| / 3 in the search and the refinement |
| A10 | two-sided outline: OCT embedding over MRI tissue or outside the crop counted as a mismatch |
| A11, A11b | simulated cut face (specimen mask removed beyond 70 % of its extent along OCT axis 1, data kept as embedding), with the method and with the two-sided outline |
| A12 | no fine-structure refinement: the pose of §4 |
| A13 | no two-class term: S = S_outline / 3 in the search and the refinement, polarity +1 |
| A14 | no orientation search: §4 from the image-centre start (header orientation, centres of the two image boxes aligned) at both polarities, then §5 |

`--starts` adds the start test: the OCT world turned by 90° and 180° about each of its array axes, through the centre
of its image box, with the method and A14 run from each turned start (`ABL/starts/`, `ABL/starts.json`). `--random-starts N`
does the same from N rotations drawn uniformly on SO(3) (scipy `Rotation.random`, `--seed`), into `ABL/starts/randNN/` and
`ABL/random_starts.json`.

Each variant is one change made in `ablate.py` around the package functions the method uses (`VARIANTS` lists them).
`--oct OCT --mri MRI [--oct-mask MASK] --pair NAME` runs the same variants on another pair, with the mask file in place of
the texture mask in base and in every variant that does not set its own.

`ablations.json` stores each variant's own values, its pose change against base over the points of the base specimen mask and
over the 8 corners of the OCT array, and its outline agreement under the base masks. `driver_check` holds the pose distance of
base to the CLI run (`--main`). A variant that makes the method refuse is recorded as failed with the message. Each OCT mask
source (texture, texture3d, intensity, or file with `--oct-mask`) is computed once and cached under `ABL/prep/`, checked
against the Params fields that §1 reads (`prep_hash` in `prep.json`). Finished variants are reused on a rerun unless
`--force` is given.

`ablate.py` does not run §6, which does not change the affine. The §6 read-outs of a run are in its result.json, and
`bench/ablate_deform.py` ablates the elements of §6 (`bench/results/I58/deform_ablation.md`).

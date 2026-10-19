# Baselines on the I58 brainstem pair, and the standard tools of the I46 test

This directory compares standard registration tools on the I58 pair, the two volumes octreg registers, at the settings their
documentation gives for two volumes of different contrast. Each tool runs from its standard starts, from the header alignment of
the two volumes up to its own global search where it has one. The results are in [BASELINES.md](BASELINES.md).
`standard_tools.py` holds the tools and settings of the I46 test ([bench/i46](../i46/README.md)).

## Inputs

Both volumes keep their NIfTI headers, which place them roughly together. The fixed image is the MRI crop as given (0.08 mm).
The moving image is the 20 µm OCT box-averaged to 0.08 mm isotropic, the voxel size of the MRI (`oct_0.08mm.nii.gz`). The only
mask a tool receives is a field-of-view (FOV) mask of the OCT (voxels > 0), and only in separate runs.
The antsAI rotation search of one ANTs run works on copies at 0.15 mm, `oct_0.15mm.nii.gz` and the MRI resampled by ANTs'
ResampleImageBySpacing in that run's command. `prepare_inputs.py` writes these files from the two original files. The affine
runs read them from `<project root>/baselines/inputs`, the project root of [bench/README.md](../README.md).

## Tools and settings

The main run of each tool uses the setting its documentation gives for this kind of registration, and this run is the one in the
figures and tables of BASELINES.md. There is one exception to the documented settings: FLIRT's headers are scaled x10, because
its schedule resamples the images to fixed voxel sizes in mm. The other affine runs are in the stored table
[affine.md](../results/I58/baselines/affine.md).

| tool | version | setting | starts | main run |
|---|---|---|---|---|
| ANTs antsRegistration | 2.6.5 | antsRegistrationSyN.sh -t a: rigid then affine, Mattes MI, the script's large-image schedule | header, centre of mass, antsAI rotation search, centre of mass + FOV mask | `ants_com` |
| FSL FLIRT | 6.0.7.23 | normmi, dof 6 then 12, default and full (±180°) search, headers scaled x10, and one run on the unscaled headers | default search, full search, -usesqform, -inweight FOV, unscaled headers | `flirt_search_x10` |
| elastix | 5.3.1 | model-zoo rigid then affine, AdvancedMattesMutualInformation | header (stops with an error, no transform), geometric centre, centre of gravity, + FOV mask | `elastix_center` |
| NiftyReg reg_aladin | 2.1.2 | defaults: symmetric block matching, rigid then affine | image centres, header (-nac), + FOV mask (ignored by this version) | `niftyreg_center` |
| FreeSurfer mri_robust_register | 8.2.0 | --cost NMI, rigid then --affine --ixform | centre of mass, --noinit (rigid step only), + --maskmov | `mri_robust_register_default_com` |
| FreeSurfer mri_coreg | 8.2.0 | --dof 6 then --dof 12 --init-reg, brute-force start | centre, --regheader, + --mov-mask | `mri_coreg_default` |
| greedy | 1.4.0 | NMI, dof 6 then 12, -n 100x50x10 | header, image centres, -search 1000, + FOV mask (as the moving mask) | `greedy_centers` |
| FreeSurfer mri_synthmorph | 8.2.0 | mri_synthmorph -m affine as documented (a 1 mm brain model) | none (the network predicts the affine) | `synthmorph_affine` |

[INSTALL.md](INSTALL.md) gives the version and installation route of each tool. The command of every affine run (`cmd.sh`)
and its output stay in its run directory under `<project root>/baselines/affine`, which holds image data and is not in the
repository.

## Scripts

| script | role |
|---|---|
| `prepare_inputs.py` | the shared inputs, from the two original files |
| `convert_affines.py` | every tool's affine transform file in one convention (OCT world to MRI world, NIfTI RAS mm) |
| `rundir.py` | `init` makes a run directory in octreg's layout from a converted 4×4 matrix. `check` compares the OCT resampled by octreg through the converted transform with the image the tool itself warped, and passes when the correlation and the Dice of the two supports both exceed 0.95 |
| `evaluate_runs.py` | affine runs: the 20 µm OCT through the transform by octreg's resampler (the overlay of the figures), the scale per OCT axis and the determinant, the wall time and the hardware, and with `--octreg-run` the same row for octreg and for the start (the header orientation with the image box centres aligned), whose run directory `reference/centres` it writes |
| `summarise.py` | the table `affine.md`, the main rows and, where there are other runs, a supplement with every run, from the summary JSON and the tool, setting and start of each run in `notes.json` |
| `standard_tools.py` | the tools and settings of the I46 test: their inputs and moving masks, run scripts and queue, the conversion of each transform and its check against the tool's own resampling (used by `bench/i46/baselines.py`) |
| `elastix_defaults/` | the elastix model-zoo rigid and affine parameter files that `standard_tools.py` uses |

The stored results (summary JSON, `notes.json` and `affine.md`) are in `bench/results/I58/baselines/`, and the figures of
BASELINES.md in `docs/figures/` (`bench/figures.py`).

The inputs and the table come from these commands, with `OCTREG_PROJECT_ROOT` set to the project root:

```bash
python bench/baselines/prepare_inputs.py --oct OCT --mri MRI -o $OCTREG_PROJECT_ROOT/baselines/inputs
python bench/baselines/evaluate_runs.py $OCTREG_PROJECT_ROOT/baselines/affine/*/* --octreg-run RUN
python bench/baselines/summarise.py
```

`OCT` and `MRI` are the two I58 files and `RUN` the octreg run of [bench/README.md](../README.md), whose `result.json` is stored
in `bench/results/I58`. Run directories without a passing `check.json` are skipped.

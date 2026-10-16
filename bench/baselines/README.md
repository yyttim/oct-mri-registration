# Baselines on the I58 brainstem pair, and the standard tools of the I46 test

This directory runs standard registration tools on the I58 pair, the two volumes octreg registers, with the settings their
documentation gives for two volumes of different contrast. No setting is tuned on this pair. The affine stage asks what each
tool does from its standard starts, from the header alignment of the two volumes up to its own global search. The deformable
stage starts every tool from octreg's affine and shows the field each tool adds. I58 has no ground truth, so the result is
the overlays, and the tables hold only properties that need no reference. The results are in [BASELINES.md](BASELINES.md).
`standard_tools.py` runs the standard tools of the I46 test ([bench/i46](../i46/README.md)).

## Inputs

Both volumes keep their NIfTI headers, which place them roughly together. The fixed image is the MRI crop as given (0.08 mm).
The moving image is the 20 µm OCT box-averaged to 0.08 mm isotropic, the resolution of the MRI (`oct_0.08mm.nii.gz`). This is
how microscopy volumes are usually passed to these tools. The only mask a tool receives is a field-of-view (FOV) mask of the
OCT (voxels > 0), and only in a separate variant. In the deformable stage the moving image is the OCT resampled onto the MRI
grid through octreg's affine (`oct_affine_0.08mm.nii.gz`, with its FOV mask `oct_affine_fov_0.08mm.nii.gz`). Every tool then
starts from the same alignment, with the identity as initial transform, and writes its field in the MRI world.
`prepare_inputs.py` writes these files from the two original files and an octreg run. The affine runs read them from
`<project root>/baselines/inputs`, the project root of [bench/README.md](../README.md). The deformable inputs depend on the
octreg run, so they and the deformable runs have a root of their own (`DEFORM` below, with `inputs/` and `deform/`).

## Tools and settings

The main run of each tool is the one at the setting its documentation gives for this kind of registration, with FLIRT's
headers scaled x10 because its schedule works in mm. It is the run in
the figures and the main rows of the tables of BASELINES.md. Every other run is in the supplement tables.

| stage | tool | version | setting | starts | main run |
|---|---|---|---|---|---|
| affine | ANTs antsRegistration | 2.6.5 | antsRegistrationSyN.sh -t a: rigid then affine, Mattes MI, the script's large-image schedule | header, centre of mass, antsAI rotation search, centre of mass + FOV mask | `ants_com` |
| affine | FSL FLIRT | 6.0.7.23 | normmi, dof 6 then 12, default and full (±180°) search, headers scaled x10 because its pyramid works in mm, and one run on the unscaled headers | default search, full search, -usesqform, -inweight FOV, unscaled headers | `flirt_search_x10` |
| affine | elastix | 5.3.1 | model-zoo rigid then affine, AdvancedMattesMutualInformation | header (stops before the first iteration: too many samples outside the moving image), geometric centre, centre of gravity, + FOV mask | `elastix_center` |
| affine | NiftyReg reg_aladin | 2.1.2 | defaults: symmetric block matching, rigid then affine | image centres, header (-nac), + FOV mask (ignored by this version) | `niftyreg_center` |
| affine | FreeSurfer mri_robust_register | 8.2.0 | --cost NMI, rigid then --affine --ixform | centre of mass, --noinit (the affine step finds no overlap, the rigid step is kept), + --maskmov | `mri_robust_register_default_com` |
| affine | FreeSurfer mri_coreg | 8.2.0 | --dof 6 then --dof 12 --init-reg, brute-force start | centre, --regheader, + --mov-mask | `mri_coreg_default` |
| affine | greedy | 1.4.0 | NMI, dof 6 then 12, -n 100x50x10 | header, image centres, -search 1000, + FOV mask (as the moving mask) | `greedy_centers` |
| affine | SynthMorph | FS 8.2.0 | mri_synthmorph -m affine as documented (a 1 mm brain model, outside its domain on this block) | as is | `synthmorph_affine` |
| deformable | ANTs SyN | 2.6.5 | antsRegistrationSyN.sh -t so (SyN, CC) and the MI form of the Quick script | identity, + FOV mask | `deform_ants_syn_cc` |
| deformable | elastix B-spline | 5.3.1 | model-zoo B-spline, Mattes MI, 16 mm grid, and a 5 mm grid | identity, + FOV mask | `deform_elastix_default` |
| deformable | NiftyReg reg_f3d | 2.1.2 | defaults (NMI, -5 voxel grid), -sx 5 mm, and -vel with the mask (the one form that honours it) | identity | `deform_niftyreg_default` |
| deformable | ConvexAdam | 0.2.0 | MIND-SSC, shipped defaults, GPU | identity, + FOV mask | `deform_convexadam_default` |
| deformable | greedy | 1.4.0 | WNCC 2x2x2 (quick start), smoothing in voxels and in mm, NMI | identity, + FOV mask (as the moving mask) | `deform_greedy_wncc` |

[INSTALL.md](INSTALL.md) gives the version and installation route of each tool. The exact commands, the conversion of each
deformable field to octreg's convention (`oct2mri_warp.nii.gz`) and the output of every run (`cmd.sh`, `log.txt`, `notes.md`)
stay in the run directories under the two roots above, which hold image data and are not in the repository. The same holds
for the freeview script and the list of sections behind `--shots`, `composites.py` and `zooms.py` (`shots.sh` and
`sections.txt` in `<project root>/baselines/shots/`).

## Scripts

| script | role |
|---|---|
| `prepare_inputs.py` | the shared inputs, from the two original files and an octreg run |
| `convert_affines.py` | every tool's affine transform file in one convention (OCT world to MRI world, NIfTI RAS mm) |
| `rundir.py` | `init` makes a run directory in octreg's layout from a converted 4×4 matrix. `check` and `check-field` compare it with the image the tool itself warped (correlation and support Dice above 0.95), so that no result rests on a wrong convention |
| `evaluate_runs.py` | affine runs: the 20 µm OCT through the transform by octreg's resampler (the overlay of the figures), the scale per OCT axis and the determinant, the wall time and the hardware, and with `--octreg-run` the same row for octreg. With `--shots`, freeview section shots by the local script above |
| `evaluate_deform.py` | deformable runs: the same resampling through affine and field, the Jacobian determinant of the field over the MRI foreground (minimum, share of folded voxels, SD of log J), the field grid, the wall time and the hardware, and with `--octreg-run` the rows of octreg's affine and §6. With `--shots`, the same section shots |
| `composites.py`, `zooms.py` | side-by-side panels per section and 2x zooms of the superior end from the `--shots` screenshots, for looking at the runs |
| `summarise.py` | the tables `affine.md` and `deform.md`, main rows and supplement, from the summary JSON and the tool, setting and start of each run in `notes.json` |
| `standard_tools.py` | the tools and settings of the I46 test: their inputs, run scripts and queue, the conversion of each transform and its check against the tool's own resampling (used by `bench/i46/baselines.py`) |
| `elastix_defaults/` | the elastix model-zoo rigid and affine parameter files that `standard_tools.py` uses |

No script computes a distance to octreg's affine, or uses octreg's OCT mask or read-outs. The Jacobian is taken over octreg's
MRI foreground, a mask of the MRI alone. The stored results (summary JSON, `notes.json` and the tables) are in
`bench/results/I58/baselines/`, and the figures of BASELINES.md in `docs/figures/` (`bench/figures.py`).

The tables come from these commands, with `OCTREG_PROJECT_ROOT` set to the project root:

```bash
python bench/baselines/evaluate_runs.py $OCTREG_PROJECT_ROOT/baselines/affine/*/* --octreg-run RUN \
    --summary DEFORM/summary_affine.json
python bench/baselines/evaluate_deform.py DEFORM/deform/*/* --octreg-run RUN --cache CACHE --summary DEFORM/summary_deform.json
python bench/baselines/summarise.py --affine DEFORM/summary_affine.json --deform DEFORM/summary_deform.json \
    --notes DEFORM/notes.json
```

`RUN` is the octreg run of `bench/results/I58`, `DEFORM` the root of the deformable runs started from its affine, and
`CACHE` the `cache.npz` of `bench/ngf_lam.py cache` for that run. Run directories without a passing `check.json` or
`check_field.json` are skipped.

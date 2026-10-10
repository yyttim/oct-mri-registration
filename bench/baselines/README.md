# Baselines on the I58 brainstem pair

Standard registration tools, each with its documented settings for two volumes of different contrast, run on the same
inputs as octreg and judged the same way. Nothing was tuned on this pair. Two questions: does a standard tool find the
affine pose of an embedded OCT block inside a cropped MRI from the header alignment, and, from octreg's affine, does a
standard deformable tool fit the OCT better than octreg's §6 field. The results are in [BASELINES.md](BASELINES.md).

## Inputs

Both volumes keep their NIfTI headers, which put them roughly together (the true pose is within about 10 degrees and 2 cm of
the header alignment). The fixed image is the MRI crop as given (0.08 mm). The moving image is the 20 um OCT box-averaged to
0.08 mm isotropic (`oct_0.08mm.nii.gz`), the resolution of the MRI, which is how microscopy volumes are usually handed to
these tools. A field-of-view mask (voxels > 0 of the OCT) is the only mask a tool receives, as a declared variant. For the
deformable stage the moving image is the OCT resampled onto the MRI grid through octreg's affine (`oct_affine_0.08mm.nii.gz`),
so every tool starts from the same alignment and writes its field in the MRI world. The method's own masks are used only by
the evaluation. `prepare_inputs.py` writes these files.

## Tools and settings

| stage | tool | version | setting | starts |
|---|---|---|---|---|
| affine | ANTs antsRegistration | 2.6.5 | antsRegistrationSyN.sh -t a: rigid then affine, Mattes MI, the script's large-image schedule | header, centre of mass, antsAI rotation search, centre of mass + FOV mask |
| affine | FSL FLIRT | 6.0.7.23 | normmi, dof 6 then 12, default and full (+-180 degrees) search, headers scaled x10 because its pyramid works in mm (declared workaround), and one run on the native header | default search, full search, -usesqform, -inweight FOV, native |
| affine | elastix | 5.3.1 | model-zoo rigid then affine, AdvancedMattesMutualInformation | header (fails: too few samples inside), geometric centre, centre of gravity, + FOV mask |
| affine | NiftyReg reg_aladin | 2.1.2 | defaults: symmetric block matching, rigid then affine | image centres, header (-nac), + FOV mask (ignored by this version) |
| affine | FreeSurfer mri_robust_register | 8.2.0 | --cost NMI, rigid then --affine --ixform | centre of mass, --noinit (diverges), + --maskmov |
| affine | FreeSurfer mri_coreg | 8.2.0 | --dof 6 then --dof 12 --init-reg, brute-force start | centre, --regheader, + --mov-mask |
| affine | greedy | 1.4.0 | NMI, dof 6 then 12, -n 100x50x10 | header, image centres, -search 1000, + moving mask |
| affine | SynthMorph | FS 8.2.0 | mri_synthmorph -m affine as documented (a 1 mm brain model: out of domain, reported for completeness) | as is |
| deformable | ANTs SyN | 2.6.5 | antsRegistrationSyN.sh -t so (SyN, CC) and the MI form of the Quick script | identity, + FOV mask |
| deformable | elastix B-spline | 5.3.1 | model-zoo B-spline, Mattes MI, 16 mm grid, and a 5 mm grid | identity, + FOV mask |
| deformable | NiftyReg reg_f3d | 2.1.2 | defaults (NMI, -5 voxel grid), -sx 5 mm, and -vel with the mask (the one form that honours it) | identity |
| deformable | ConvexAdam | 0.2.0 | MIND-SSC, shipped defaults, GPU | identity, + FOV mask |
| deformable | greedy | 1.4.0 | WNCC 2x2x2 (quick start), smoothing in voxels and in mm, NMI | identity, + moving mask |

Install routes, exact commands and every tool's output are in the run directories (`cmd.sh`, `log.txt`, `notes.md`), which
hold the data and are not in the repository. [INSTALL.md](INSTALL.md) lists the versions and how each tool was installed.

## Scripts

| script | role |
|---|---|
| `prepare_inputs.py` | the shared inputs from the two original files and an octreg run |
| `convert_affines.py` | every tool's transform file into one convention (OCT world to MRI world, NIfTI RAS mm) |
| `rundir.py` | `init` makes an octreg-style run directory from a converted 4x4, and `check` / `check-field` verify the conversion against the image the tool itself warped (correlation and support Dice above 0.95), so no result rests on a wrong convention |
| `evaluate_runs.py` | affine runs: the 20 um OCT through the transform with octreg's resampler, `bench/evaluate.py` (rim boundary agreement, pose distance to the octreg affine, frame check), Dice against the MRI foreground, the scale per OCT axis, freeview section shots |
| `evaluate_deform.py` | deformable runs: the same resampling through affine and field, octreg's §6 read-outs on the warped OCT, the field over the MRI foreground (magnitude, Jacobian, folding), section shots |
| `composites.py`, `zooms.py` | side-by-side panels per section and 2x zooms of the superior end for the visual check |
| `summarise.py` | the tables of BASELINES.md from the summary JSON |

The stored results are in `bench/results/I58/baselines/` (summary JSON and tables). The images of the pair are drawn locally
and are not published.

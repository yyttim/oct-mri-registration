# Held-out evaluation on DANDI:000874

octreg was developed on one block, I58. These scripts test it, with its default parameters, on public data that played no part
in its development: the four OCT blocks of [DANDI:000874](https://dandiarchive.org/dandiset/000874/draft) ("Multimodal
Postmortem Infant Brainstem Imaging", C. Magnain and L. Zöllei, CC BY 4.0, draft last modified on 14 August 2026), described in
[Magnain et al., bioRxiv 2026](https://doi.org/10.64898/2026.08.28.744504). Every block comes with the authors' own OCT→MRI
affine, which serves as the reference. The results are in [bench/results/SP002/tables.md](../results/SP002/tables.md), and the
README shows the main table and the freeview panels.

## Data

| block | tissue | OCT grid (20 µm) |
|---|---|---|
| 01 | caudal medulla | 877 × 725 × 451 |
| 02 | rostral medulla and caudal pons | 1161 × 1148 × 617 |
| 03 | rostral pons | 1163 × 1425 × 417 |
| 04 | midbrain | 1163 × 1709 × 550 |

The specimen (sub-SP002) is one infant brainstem. Per the preprint, each block was embedded in agarose for serial-sectioning OCT,
and blocks 03 and 04 are one piece of tissue imaged in two sessions. The OCT input is the authors' 20 µm scattering-coefficient
volume, the same kind of input as I58. The MRI is the 150 µm chunk-2 scan of the dissected brainstem and cerebellum: 3D FLASH at
seven flip angles (3° to 40°) and the T1 and PD maps fitted from them, all on one 448³ grid. The main input is the 5° FLASH, the
image the authors registered the OCT to. In this scan the brainstem still lies against the cerebellum. octreg's MRI foreground
covers this share of each C3 k = 0 crop and of the I58 crop (`prepare.py`, stored in
[context.json](../results/SP002/context.json)):

| crop | MRI foreground |
|---|---|
| block 01 | 69 % |
| block 02 | 74 % |
| block 03 | 52 % |
| block 04 | 30 % |
| I58 crop | 32 % |

## Reference

The authors registered each block with a manual rigid start in FreeView followed by a robust NMI affine (the preprint), made with
mri_robust_register per the dataset's sidecar files, and checked it visually. It is an expert registration, not ground truth.
Two checks (`reference.py`) show how well it fits the images. The first runs on the C3 k = 0 crop of each block, the second on
the whole MRI, as the authors registered it.

- **Local refinements started at the reference.** ANTs (MI, the two finest levels) moves it by a mean of 0.12 to 0.24 mm over
  the tissue, NiftyReg reg_aladin (two levels) by 0.37 to 0.60 mm, and octreg's own §4 and §5 by 0.20 to 0.33 mm on blocks 01
  to 03 and by 8.8 mm on block 04. Differences of about 0.2 mm cannot be resolved against this reference.
- **Restarting mri_robust_register does not test the reference.** The restarts begin from the reference moved by 2° and 0.5 mm.
  The tool's coarsest level has 28³ voxels of 2.4 mm, on which a block spans a few voxels. On blocks 01 to 03 every restart
  that finished ends 6.2 to 28.9 mm from the reference. On block 04, 4 of 10 restarts end within 1 mm. Two of the 40 restarts
  stopped on a numerical error of the tool.

## Inputs and protocol

- **OCT.** Each block's OME-Zarr level 0 becomes a NIfTI in the authors' frame. The zarr stores their NIfTI header, and its
  affine equals the source geometry of their LTA. The destination geometry of every LTA equals the header of every chunk-2 MRI
  image. `prepare.py` checks both.
- **MRI crops.** octreg expects the MRI cropped around the block, as the I58 MRI is. Each crop here is an axis-aligned box on
  the MRI grid around the 0.5 and 99.5 percentiles of the OCT tissue placed by the reference. It gets a margin m on every face
  and a seeded random shift of up to s mm per axis, so the block is not centred. The same percentiles on I58, with its OCT placed
  by octreg, lie −2.33 to 4.40 mm inside the faces of its crop (negative where the crop cuts the tissue), against 1.11 to
  5.14 mm for C3 ([context.json](../results/SP002/context.json)). The conditions:
  - **C3:** m = 3 mm, s = 2 mm, the I58-like crop and the main condition.
  - **C6:** m = 6 mm, s = 5 mm.
  - **C12:** m = 12 mm, s = 10 mm.
  - Each with ten crops per block (seeds 100 · block + k).
  - The C3 k = 0 box also on the eight other chunk-2 images.
  - The whole 448³ MRI, which is outside octreg's input assumption.
- **octreg.** Version 2.0.0 with its default parameters (hash da914d8ccc555207) and no masks: `octreg register OCT CROP -o RUN`.
- **Standard tools.** The tools of [bench/baselines](../baselines/BASELINES.md) run at the setting reported there, with the same
  commands, inputs and conversions as on I58, on the C3 k = 0 crop of each block. NiftyReg reg_aladin, the only tool that
  matched octreg on I58, also runs on all C3 crops. On the C3 k = 0 crops each converted transform is checked against the
  tool's own resampling of the OCT, as `bench/baselines/rundir.py check` does on I58. The correlation is 0.98 or more on blocks
  02 to 04 and 0.91 to 1.00 on block 01, where SynthMorph's own resampling and the converted one overlap with a Dice of 0.865.
  FLIRT's own resampling holds no OCT tissue inside the crop, so its check is inconclusive.

## Metrics

Every estimate T is compared with the reference R. Both map the OCT world to the MRI world.

- **Error:** |T x − R x| over the OCT tissue points (the Otsu-thresholded OCT at 80 µm, every fourth 20 µm voxel), reported as
  mean, median, p95 and maximum.
- **Upper bound:** the largest error over the eight corners of the tissue box.
- **Rotation:** the rotation angle of T R⁻¹.
- **Classes:** a run is *accurate* when its mean error is at most 0.30 mm (two MRI voxels), and has the *correct pose* when it
  is at most 1 mm.

The reference is affine, so octreg's smooth deformation (§6) is not scored against it.

## Running

```
export OCTREG_DANDI874_DIR=/path/to/dandi-000874            # bench/paths.py gives the other roots
bash bench/dandi874/fetch.sh                                 # about 10.5 GB, needs the DANDI client (pip install dandi)
python bench/dandi874/prepare.py                             # OCT NIfTI, reference affines, crops
python bench/dandi874/run.py                                 # 156 octreg runs, one at a time, scored -> octreg.json
python bench/dandi874/baselines.py prepare
bash "${OCTREG_PROJECT_ROOT:-.}/bench_runs/SP002/baselines/queue.sh"
python bench/dandi874/baselines.py collect                   # -> baselines.json
python bench/dandi874/reference.py local
python bench/dandi874/reference.py restarts
python bench/dandi874/reference.py score                     # -> reference.json
python bench/dandi874/summarize.py                           # -> tables.md
python bench/dandi874/figures.py                             # -> docs/figures/sp002
```

The standard tools and FreeSurfer run as described in [bench/baselines/INSTALL.md](../baselines/INSTALL.md). On Windows the
commands run in Git Bash, and the scripts start the Linux tools in WSL. Elsewhere they run in the same shell. The run
directories hold image data and stay under `<project root>/bench_runs/SP002`, which is not in the repository.

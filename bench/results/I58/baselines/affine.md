| tool | variant | wall s | scale per OCT axis | rim OCT to MRI / MRI to OCT (mm) | OCT boundary p90 (mm) | Dice | to octreg: mean mm / deg | frame check | note |
|---|---|---|---|---|---|---|---|---|---|
| ants | antsai | 17670 | 1.031 / 1.004 / 1.032 | 2.88 / 4.29 | 5.37 | 0.000 | 40.8 / 165.2 | fail | antsAI global rotation search (0.15 mm), then the same stages: the search picks a flipped pose |
| ants | com | 45 | 0.864 / 0.618 / 0.921 | 1.92 / 2.30 | 4.66 | 0.704 | 3.9 / 9.3 | pass | antsRegistrationSyN.sh -t a (rigid+affine, Mattes MI), centre-of-mass start, OCT shrunk to half its volume |
| ants | com_fov | 68 | 1.283 / 1.008 / 1.101 | 1.62 / 2.22 | 3.92 | 0.697 | 2.9 / 6.6 | pass | centre-of-mass start with the OCT FOV mask, OCT inflated by 39 % |
| ants | identity | 25 | 0.915 / 1.032 / 0.989 | 2.09 / 2.10 | 6.16 | 0.020 | 40.3 / 23.2 | fail | same stages from the header alignment, ends 17.8 deg from that start, far from the block |
| elastix | center | 26 | 0.916 / 0.835 / 0.974 | 0.86 / 1.28 | 3.25 | 0.809 | 2.6 / 12.1 | pass | model-zoo rigid+affine (Mattes MI), geometric-centre start, stays 1 to 2 deg from the header orientation, shrunk |
| elastix | cog | 32 | 0.899 / 0.818 / 0.970 | 0.87 / 1.24 | 3.26 | 0.815 | 2.8 / 12.3 | pass | same with the centre-of-gravity start, same pose as center |
| elastix | cog_fov | 29 | 1.028 / 0.995 / 1.012 | 1.56 / 2.00 | 3.65 | 0.724 | 2.4 / 10.6 | pass | centre-of-gravity start with the OCT FOV mask, no shrink, same wrong orientation |
| flirt | fullsearch_x10 | 8768 | 0.732 / 0.731 / 0.727 | 1.41 / 2.12 | 4.41 | 0.603 | 19.8 / 168.6 | pass | FLIRT with the full +-180 deg search, headers scaled x10 |
| flirt | search_native | 3761 | 0.755 / 0.779 / 0.615 | 1.93 / 2.62 | 4.99 | 0.590 | 18.1 / 176.5 | pass | FLIRT as documented on the unscaled 0.08 mm inputs |
| flirt | search_x10 | 3193 | 0.764 / 0.758 / 0.673 | 1.46 / 2.18 | 4.19 | 0.597 | 19.6 / 170.5 | pass | FLIRT normmi, dof 6 then 12, default search, headers scaled x10 |
| flirt | search_x10_fov | 4425 | 0.858 / 0.807 / 0.775 | 1.63 / 2.43 | 4.70 | 0.653 | 4.9 / 6.0 | fail | FLIRT default search with -inweight (FOV weighting), headers scaled x10 |
| flirt | sqform_x10 | 1409 | 1.174 / 1.180 / 1.171 | 1.95 / 6.10 | 2.74 | 0.000 | 43.3 / 68.9 | fail | FLIRT -usesqform -nosearch (header start), headers scaled x10 |
| greedy | centers | 63 | 1.186 / 0.934 / 0.896 | 1.21 / 1.97 | 3.78 | 0.727 | 3.7 / 14.4 | pass | greedy NMI from the image centres |
| greedy | centers_fov | 73 | 1.000 / 1.000 / 1.000 | 1.42 / 1.80 | 3.61 | 0.733 | 2.4 / 11.0 | pass | greedy NMI from the image centres with the moving FOV mask, stalled at the start |
| greedy | identity | 55 | 1.283 / 1.973 / 1.018 | 2.53 / 2.74 | 5.04 | 0.025 | 66.6 / 14.6 | fail | greedy NMI rigid then affine from the header alignment, the affine stage stretched the OCT 2.2x |
| greedy | search | 53 | 1.000 / 1.000 / 1.000 | n/a / n/a | n/a | 0.000 | 42.5 / 32.3 | fail | greedy -search 1000 global search, the best candidate has almost no overlap with the MRI |
| mri_coreg | default | 1329 | 0.915 / 0.808 / 0.998 | 1.01 / 1.70 | 3.28 | 0.788 | 4.1 / 16.6 | pass | mri_coreg --dof 6 then --dof 12 --init-reg, brute-force initial search, the OCT shrunk by 19 % along one axis |
| mri_coreg | default_movmask | 1066 | 0.915 / 0.808 / 0.998 | 1.01 / 1.70 | 3.28 | 0.788 | 4.1 / 16.6 | pass | the same with --mov-mask oct_fov: identical to default |
| mri_coreg | regheader | 1166 | 1.054 / 1.057 / 1.025 | 1.72 / 2.05 | 4.09 | 0.696 | 2.0 / 10.2 | pass | mri_coreg --regheader (header start) |
| mri_robust_register | default_com | 842 | 0.990 / 0.997 / 1.071 | 1.61 / 1.94 | 3.54 | 0.746 | 3.0 / 11.7 | pass | mri_robust_register --cost NMI, rigid then --affine --ixform, centre-of-mass start |
| mri_robust_register | default_com_maskmov | 1097 | 0.990 / 0.997 / 1.071 | 1.61 / 1.94 | 3.54 | 0.746 | 3.0 / 11.7 | pass | the same with --maskmov oct_fov: identical to default_com (the OCT is already zero outside its field of view) |
| mri_robust_register | noinit_rigid_only | 314 | 1.000 / 1.000 / 1.000 | n/a / n/a | n/a | 0.000 | 42.4 / 10.7 | fail | mri_robust_register --noinit (header start): the affine step diverged, the rigid result is recorded |
| niftyreg | center | 309 | 0.966 / 0.958 / 0.983 | 1.06 / 1.59 | 3.77 | 0.760 | 0.3 / 1.0 | pass | reg_aladin defaults (block matching, rigid then affine), image-centre start, lands near octreg's affine |
| niftyreg | center_fov | 238 | 0.966 / 0.958 / 0.983 | 1.06 / 1.59 | 3.77 | 0.760 | 0.3 / 1.0 | pass | the floating mask is silently ignored by this NiftyReg version: identical to center |
| niftyreg | nac | 206 | 1.071 / 0.947 / 1.035 | 1.62 / 2.76 | 4.68 | 0.714 | 10.6 / 32.3 | pass | reg_aladin -nac (header start), a different, wrong pose |
| reference | centres | n/a | 1.000 / 1.000 / 1.000 | 1.54 / 1.85 | 3.57 | 0.743 | 2.4 / 11.1 | pass | reference, not a registration: image centres aligned with the header orientation, the start most tools use |
| synthmorph | affine | 38 | 0.821 / 0.617 / 1.197 | 3.88 / n/a | 10.36 | 0.000 | 28.3 / 37.8 | pass | mri_synthmorph -m affine as documented (1 mm brain model): out of domain, fails |

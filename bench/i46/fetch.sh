#!/bin/bash
# The files the I46 test reads, about 11 GB: from DANDI:000026 (sub-I46) the OCT, its vessel segmentation, the 10, 20 and
# 30 deg FLASH, the T1 map and the manual vessel and layer labels, and from github.com/neuroscales/biccn-broca-I46 at the commit
# below the registration files of Costantini et al. 2023, into DEST/sub-I46/{oct,mri,labels,costantini_repo/results}.
# Needs the DANDI client (pip install dandi) and curl.
# Usage: bash bench/i46/fetch.sh [DEST]  (default: $OCTREG_DANDI000026_DIR)
set -e
D=${1:-$OCTREG_DANDI000026_DIR}
[ -n "$D" ] || { echo "give DEST or set OCTREG_DANDI000026_DIR"; exit 1; }
S=$D/sub-I46
get() { mkdir -p "$S/$1"; dandi download --existing refresh -o "$S/$1" "dandi://dandi/000026@draft/$2"; }
get oct sub-I46/ses-OCT/micr/sub-I46_ses-OCT_sample-BrocaAreaS01_OCT.ome.tiff
get oct sub-I46/ses-OCT/micr/sub-I46_ses-OCT_sample-BrocaAreaS01_OCT.json
get oct derivatives/sub-I46/ses-OCT/vessel/ves_seg.tif
M=derivatives/EPIC/sub-I46/ses-MRI/anat
for f in 1 2 3; do get mri "$M/sub-I46_ses-MRI_flip-${f}_VFA.nii.gz"; get mri "$M/sub-I46_ses-MRI_flip-${f}_VFA.json"; done
get mri "$M/sub-I46_T1map.nii.gz"
get mri "$M/sub-I46_T1map.json"
L=derivatives/Labels/sub-I46/ses-MRI/anat
for l in vessels infrasupra; do
  get labels "$L/sub-I46_ses-MRI_space-EPIC_label-${l}_dseg.nii.gz"
  get labels "$L/sub-I46_ses-MRI_space-EPIC_label-${l}_dseg.json"
done
G=https://raw.githubusercontent.com/neuroscales/biccn-broca-I46/b36e398cc42db4ca89ecc121818255bd96fc67ad
F=(results/mri2oct/affine.lta results/mri/recon2raw.lta)
for i in $(seq 8 23); do F+=("$(printf 'results/oct2lsfm/svf.%02d.nii.gz' "$i")"); done
for f in "${F[@]}"; do
  mkdir -p "$S/costantini_repo/$(dirname "$f")"
  curl -fsSL --retry 5 -o "$S/costantini_repo/$f" "$G/$f"
done

#!/bin/bash
# The files of DANDI:000874 (sub-SP002) that the evaluation reads, about 10.5 GB, with the DANDI paths kept:
# the four 20 um scattering-coefficient OCT blocks with the authors' OCT -> MRI LTAs, and every chunk-2 MRI image.
# Needs the DANDI client (pip install dandi).  Usage: bash bench/dandi874/fetch.sh [DEST]  (default: $OCTREG_DANDI874_DIR)
set -e
D=${1:-$OCTREG_DANDI874_DIR}
[ -n "$D" ] || { echo "give DEST or set OCTREG_DANDI874_DIR"; exit 1; }
O=derivatives/OCT-pipeline/sub-SP002/micr/sub-SP002_ses-OCT
M=derivatives/MRI-pipeline/sub-SP002/anat/sub-SP002_ses-MRI
P=()
for n in 01 02 03 04; do
  P+=("${O}_sample-${n}_res-20um_OCT.ome.zarr" "${O}_sample-${n}_res-20um_OCT.json" "${O}_sample-${n}_to-SP002_xfm.lta" "${O}_sample-${n}_to-SP002_xfm.json")
done
for f in 1 2 3 4 5 6 7; do P+=("${M}_flip-${f}_chunk-2_VFA.nii.gz" "${M}_flip-${f}_chunk-2_VFA.json"); done
for m in T1map PDmap; do P+=("${M}_chunk-2_${m}_VFA.nii.gz" "${M}_chunk-2_${m}_VFA.json"); done
P+=(dataset_description.json participants.tsv samples.tsv)
for p in "${P[@]}"; do
  mkdir -p "$D/$(dirname "$p")"
  dandi download --existing refresh -o "$D/$(dirname "$p")" "dandi://dandi/000874@draft/$p"
done

"""The block's MRI as the reference of the registered volume (docs/TILESTACK.md, MRI reference): the tissue mask and the
levelled copy that octreg reads, and octreg's parameters for the MRI. Needs only the written volume and its source map.
"""
import dataclasses
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu

from octreg import geometry as G, io
from octreg.params import Params
from octreg.preprocess import _fill_planes
from octreg.register import apply, register

from . import BELOW

SMOOTH_MM = 0.16                # Gaussian sigma of log10 mu_s before the tissue threshold


def tissue(vol, src, spacing_mm):
    """Tissue mask of vol (x, y, depth), 0 without data: the voxels where log10 mu_s, smoothed over the reliable voxels
    of the source map src (normalised Gaussian, sigma SMOOTH_MM), lies above Otsu's threshold, with holes filled in
    every array plane. The voxels below the reliable planes take the tissue of those around them."""
    meas = vol > 0
    w = (meas & (src < BELOW)).astype(np.float32)
    s = SMOOTH_MM / np.asarray(spacing_mm, float)
    S = ndimage.gaussian_filter(np.log10(vol, out=np.zeros_like(vol), where=w > 0), s)
    ndimage.gaussian_filter(w, s, output=w)
    seen = meas & (w > 1e-3)
    np.divide(S, w, out=S, where=seen)
    del w
    return _fill_planes(seen & (S > threshold_otsu(S[seen]))) & meas


def depth_under_cut(src):
    """Depth of every voxel under the cut: the planes from where its slab begins in its column of the source map src."""
    k = np.arange(src.shape[2], dtype=np.int16)
    slab = np.where(src >= 0, src % BELOW, -1)
    first = np.ones(src.shape, bool)
    first[:, :, 1:] = slab[:, :, 1:] != slab[:, :, :-1]
    return k - np.maximum.accumulate(np.where(first, k, 0), axis=2)


def levelled(vol, mask, src):
    """vol with every voxel divided by the median of the tissue (mask) of its slab at its depth under that slab's cut. A
    slab's depth without tissue takes the median of all tissue."""
    key = (np.maximum(src, 0) % BELOW).astype(np.int32) * vol.shape[2] + depth_under_cut(src)
    lt, kt = np.log(vol[mask]), key[mask]
    groups = np.unique(kt)
    level = np.full(int(key.max()) + 1, np.median(lt), np.float32)
    level[groups] = ndimage.median(lt, kt, groups)
    out = np.exp(level)[key]
    return np.divide(vol, out, out=out)


def octreg_params(mri):
    """octreg's parameters for the MRI file: a base grid of one MRI voxel (its largest side, within octreg's own fine
    and base grids) and, after octreg's passes, passes each 1.5 times finer, the last at two voxels of that grid."""
    P = Params()
    v = min(max(G.mm6(io.load_volume(mri).spacing_mm.max()), P.fine_mm), P.base_mm)
    sigmas = list(P.sigmas_mm)
    while sigmas[-1] > G.mm6(2 * v):
        sigmas.append(max(G.mm6(sigmas[-1] / 1.5), G.mm6(2 * v)))
    return dataclasses.replace(P, base_mm=v, sigmas_mm=tuple(sigmas))


def register_mri(volume, mri, out_dir):
    """octreg of the volume file, with its source map *_source.nii* next to it, to the MRI file. out_dir gets octreg's
    outputs for the levelled copy, the copy (levelled.nii.gz), the tissue mask (tissue.nii.gz) and the volume itself on
    the MRI grid (*_in_mri.nii.gz)."""
    out, vo = Path(out_dir), io.load_volume(volume)
    vol = np.asarray(nib.load(str(volume)).dataobj, np.float32)
    src = np.asarray(nib.load(str(volume).replace(".nii", "_source.nii")).dataobj)
    mask = tissue(vol, src, vo.spacing_mm)
    io.save_nifti(mask.astype(np.uint8), vo.affine, out / "tissue.nii.gz")
    io.save_nifti(levelled(vol, mask, src), vo.affine, out / "levelled.nii.gz")
    del vol, src, mask
    register(out / "levelled.nii.gz", mri, out, oct_mask=out / "tissue.nii.gz", params=octreg_params(mri))
    apply(out, volume, mri, out / Path(volume).name.replace(".nii", "_in_mri.nii"))

"""Write the registered volume and, for every slab, its placement as an ITK displacement field, both as NIfTI. Voxel
(i, j, k) of a slab file, unplaced, lies at world (i vx, j vy, -k vz) mm, and the common frame (X, Y, Z), in working
voxels w, at (X w + (w - vx) / 2, Y w + (w - vy) / 2, -Z w - (w - vz) / 2).
"""
import os

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt

from . import compose, data


def _nifti(a, aff):
    img = nib.Nifti1Image(a, aff)
    img.header.set_xyzt_units("mm")
    img.set_sform(aff, 1)
    img.set_qform(aff, 1)
    return img


def _affine(z0):
    """Voxel to world of a volume at the voxel of the slab files whose plane 0 begins at common working plane z0."""
    vz, vy, vx = data.RES_MM
    aff = np.diag([vx, vy, -vz, 1.0])
    aff[:3, 3] = -data.PAD * data.VOX, -data.PAD * data.VOX, -z0 * data.VOX
    return aff


def slab_affine():
    """Voxel (x, y, depth) of a slab file to the world in which the warps place it."""
    vz, vy, vx = data.RES_MM
    return np.diag([vx, vy, -vz, 1.0])


def name():
    """registered_<voxel>um, from the voxel of the slab files (x, y, depth)."""
    um = [f"{v * 1000:g}" for v in data.RES_MM[::-1]]
    return f"registered_{um[0] if len(set(um)) == 1 else 'x'.join(um)}um"


def save_volume(vol, z0, path, src=None):
    """vol (Z, Y, X) at the voxel of the slab files -> NIfTI (X, Y, Z), and src, the source map, next to it."""
    aff = _affine(z0)
    nib.save(_nifti(np.transpose(np.asarray(vol, np.float32), (2, 1, 0)), aff), path)
    if src is not None:
        nib.save(_nifti(np.transpose(np.asarray(src, np.int16), (2, 1, 0)), aff), "_source.nii".join(path.rsplit(".nii", 1)))
    return path


def save_warps(sol, lay, out, z0, nz):
    """warps/slab_NNN.nii.gz: the placement of slab NNN as an ITK displacement field (LPS mm) on the in-plane
    grid of the registered volume of nz planes, in two planes at its top and bottom, and between moved tiles
    that of the nearest tile."""
    os.makedirs(f"{out}/warps", exist_ok=True)
    f = data.FACTOR[1]
    ny, nx = (n + 2 * data.PAD * f for n in data.SHAPE[1:])
    aff = _affine(z0)
    aff[2, 2] = -max(nz - 1, 1) * data.RES_MM[0]
    Y, X = (np.mgrid[0:ny, 0:nx] + 0.5) / f - 0.5 - data.PAD                # working pixels, as compose places them
    for s in sol["slabs"]:
        y, x, cell, zo = (v.cpu().numpy() for v in compose.inverse_map(sol, lay, s, ny, nx, 1 / f))
        d = np.stack([X - x, Y - y, zo]) * data.VOX                   # (L, P, S) mm: RAS (x - X, y - Y, zo)
        iy, ix = distance_transform_edt(cell < 0, return_distances=False, return_indices=True)
        d = d[:, iy, ix].transpose(2, 1, 0)[:, :, None, None]
        img = _nifti(np.repeat(d, 2, axis=2).astype(np.float32), aff)
        img.header.set_intent("vector")
        nib.save(img, f"{out}/warps/slab_{s:03d}.nii.gz")

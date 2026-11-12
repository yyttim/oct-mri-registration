"""Write a solution as physical-unit transforms (JSON) and the registered volume as NIfTI."""
import json

import nibabel as nib
import numpy as np

from . import data
from .solve import Spline

DOC = f"""Raw slab coordinates in mm, from 0-based voxel indices of the {data.VOX * 1000:g} um maps (shape_zyx): x = i_x * {data.VOX:g} with
i_x the index along the {data.NX}-voxel axis (the first MATLAB dimension), y = i_y * {data.VOX:g} along the {data.NY}-voxel axis,
z = i_z * {data.VOX:g} along depth. If the maps are n x n x n block means of finer data, the fine voxel j sits at
(j - (n - 1) / 2) * {data.VOX:g} / n mm on the same axes. A point of slab s in mosaic tile k maps to the common frame (mm) by
    X = x + ax + fx(x, y) + ex[k]
    Y = y + ay + fy(x, y) + ey[k]
    Z = z + az + fz(x, y) + ez[k]
fx, fy, fz: cubic B-splines, control (i, j) at x = (j - 1) G, y = (i - 1) G, G = bspline.G_mm, with x and y clipped to the
raw field (0 to {data.NX * data.VOX:g} and 0 to {data.NY * data.VOX:g} mm). A slab's fx_mm, fy_mm and fz_mm list the n_rows x n_cols
control values row by row (control (i, j) at index i * n_cols + j).
Tile k = i * cols + j: solve x = X0 + j PX + i SX + W / 2, y = Y0 + i PY + j SY + H / 2 (layout, mm) for (i, j), round
both and clip them to the lattice. If the slab has no tile there, use the present tile with the nearest (i, j).
Tiles not listed have e = 0."""


def save_transforms(sol, lay, path):
    assert sol["field"] is not None and not np.abs(sol["aff"][:, :, 1:]).any(), "the transforms are written for the method's model"
    v = data.VOX
    L = lay.to_dict()
    layout = {k: (L[k] * v if k in ("X0", "Y0", "PX", "PY", "SX", "SY", "W", "H") else L[k]) for k in L}
    spl = Spline(sol["G"])
    out = dict(description=DOC, voxel_mm_of_fit=v, shape_zyx=[data.NP, data.NY, data.NX], slabs=sol["slabs"], layout=layout,
               bspline=dict(G_mm=sol["G"] * v, n_rows=spl.gy, n_cols=spl.gx), settings=sol["settings"], per_slab={})
    for k, s in enumerate(sol["slabs"]):
        e = dict(ax_mm=float(sol["aff"][k, 0, 0]) * v, ay_mm=float(sol["aff"][k, 1, 0]) * v, az_mm=float(sol["aff"][k, 2, 0]) * v,
                 tiles={str(t): [round(float(x) * v, 6) for x in sol["tiles"][k, t]] for t in range(lay.n)
                        if np.abs(sol["tiles"][k, t]).max() > 1e-3})
        for i, a in enumerate("xyz"):
            e[f"f{a}_mm"] = (sol["field"][k, i] * v).round(8).tolist()
        out["per_slab"][str(s)] = e
    json.dump(out, open(path, "w"))
    return path


def _affine(z0, level_um):
    s = level_um / 1000.0
    aff = np.diag([s, s, data.VOX, 1.0])
    aff[2, 3] = z0 * data.VOX
    aff[0, 3] = aff[1, 3] = ((s / data.VOX - 1) / 2 - data.PAD) * data.VOX
    return aff


def save_volume(vol, z0, level_um, path, src=None):
    """vol (Z, Y, X) -> NIfTI (X, Y, Z), voxel level_um in-plane and the raw voxel in depth, origin at the common frame.
    The grid has a margin of data.PAD raw pixels: voxel 0 of a working-grid volume is centred at raw pixel
    (DOWN - 1) / 2 - PAD. src: the source map, written next to it."""
    aff = _affine(z0, level_um)
    for a, p in ((np.asarray(vol, np.float32), path), (None if src is None else np.asarray(src, np.int16), path.replace(".nii", "_source.nii"))):
        if a is not None:
            img = nib.Nifti1Image(np.transpose(a, (2, 1, 0)), aff)
            img.header.set_xyzt_units("mm")
            nib.save(img, p)
    return path

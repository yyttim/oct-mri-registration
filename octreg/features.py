"""The structure similarity F (docs/METHOD.md §2): the correlation of the gradient-orientation tensors of OCT and MRI.
The tensor n n^T of the normalised gradient n = g / sqrt(|g|^2 + eps^2) ignores the sign of g, so F needs no intensity
mapping."""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy import ndimage

from . import geometry as G

MIN_POINTS = 10


def to_torch(x, device):
    return torch.as_tensor(x if torch.is_tensor(x) else np.asarray(x, np.float32)).to(device=device, dtype=torch.float32)


def orientation_tensor(n):
    """Orientation tensor n n^T of vectors n [3, ...] as six channels whose dot product is (n . n')^2. -> [6, ...]"""
    r2 = math.sqrt(2.0)
    return torch.stack([n[0] * n[0], n[1] * n[1], n[2] * n[2], r2 * n[0] * n[1], r2 * n[0] * n[2], r2 * n[1] * n[2]])


def gradient_field(arr, mask, affine, sigma_mm, device="cuda"):
    """World-frame gradient [3, D, H, W] per mm of arr smoothed inside mask by normalised convolution, zero outside it.
    arr and mask lie on an isotropic grid, affine voxel -> world."""
    h = float(np.linalg.norm(np.asarray(affine)[:3, 0]))
    x, m = to_torch(arr, device), to_torch(mask, device)
    g = G.nc_gradient(x * m, m, sigma_mm / h)
    Ainv = torch.as_tensor(np.linalg.inv(np.asarray(affine, float)[:3, :3]).T, dtype=torch.float32, device=device)
    return torch.einsum("ij,jklm->iklm", Ainv, g)


def median_norm(g, mask):
    """eps: the median gradient magnitude over the mask."""
    return g.norm(dim=0)[to_torch(mask, g.device) > 0].median()


def tensors(g, eps):
    return orientation_tensor(g / torch.sqrt((g * g).sum(0) + eps * eps))


class MRITensors:
    """The MRI tensor field on its grid, zero outside the mask."""

    def __init__(self, arr, mask, affine, sigma_mm, device="cuda"):
        g = gradient_field(arr, mask, affine, sigma_mm, device)
        self.field = tensors(g, median_norm(g, mask))
        self.affine = np.asarray(affine, float)

    def at(self, y):
        """Tensors at MRI world points y [N, 3], zero outside the grid. -> [6, N]"""
        return G.sample_world(self.field, self.affine, y)


class OCTTemplate:
    """OCT side: world positions of every second voxel of `points` and their gradients smoothed inside `mask`."""

    def __init__(self, arr, mask, affine, sigma_mm, points, device="cuda"):
        sel = np.zeros(np.shape(points), bool)
        sel[::2, ::2, ::2] = True
        idx = torch.as_tensor(np.argwhere(np.asarray(points, bool) & sel), device=device)
        if len(idx) < MIN_POINTS:
            raise ValueError("OCT template has too few points")
        g = gradient_field(arr, mask, affine, sigma_mm, device)
        self.eps = median_norm(g, mask)
        self.g = g[(slice(None),) + tuple(idx.T)]
        self.pts = G.apply_affine(np.asarray(affine, float), idx.to(torch.float32))


def correlation(a, b):
    """Centred, normalised correlation of the tensor sets a, b [6, N]."""
    da, db = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    return (da * db).sum() / torch.sqrt((da * da).sum() * (db * db).sum() + 1e-24)


def score(T, tpl: OCTTemplate, mri: MRITensors):
    """F of the pose T (torch 4x4). The OCT gradients go through A^-T, A the linear part of T."""
    return correlation(tensors(torch.linalg.inv(T[:3, :3]).T @ tpl.g, tpl.eps), mri.at(G.apply_affine(T, tpl.pts)))


def eroded(mask, mm, h):
    """Voxels of mask farther than mm from its outside on a grid of spacing h. Beyond the array counts as outside."""
    return ndimage.distance_transform_edt(np.pad(np.asarray(mask, bool), 1), sampling=h)[1:-1, 1:-1, 1:-1] > mm

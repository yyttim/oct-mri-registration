"""The structure score of octreg (docs/METHOD.md §2): the correlation of gradient-orientation tensors.

Each volume I with mask m is smoothed and differentiated inside the mask by normalised convolution, g = grad(G_s(I m) / G_s(m)),
so the mask edge adds no gradient, and g is taken to world coordinates. The normalised gradient is n = g / sqrt(|g|^2 + eps^2)
(Haber and Modersitzki 2007), eps the median |g| over the mask, and its orientation tensor P(n) = n n^T is stored as six
channels whose dot product is <P(n), P(n')> = (n . n')^2, the squared product of normalised gradient fields. The tensor does not
change when the gradient changes sign, so the score needs neither an intensity mapping nor the contrast polarity.

The score is computed in template form, as the orientation search computes it: over the OCT points x_i, the correlation of the
OCT tensors, their gradients taken through the pose (g' = A^-T g for the linear part A, and (I - grad u^T) g' under a small
displacement u), with the MRI tensors sampled at the mapped points, zero outside the MRI mask and grid:
    F = sum_i <a_i - abar, b_i - bbar> / sqrt(sum_i |a_i - abar|^2  sum_i |b_i - bbar|^2).
OCT tissue mapped onto MRI background therefore lowers F, as it lowers the search score.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy import ndimage

from . import geometry as G


def to_torch(x, device):
    """numpy or torch -> float32 tensor on device."""
    return torch.as_tensor(x if torch.is_tensor(x) else np.asarray(x, np.float32)).to(device=device, dtype=torch.float32)


def orientation_tensor(n):
    """Orientation tensor n n^T of vectors n [3, ...] (torch) as six channels (xx, yy, zz, sqrt2 xy, sqrt2 xz, sqrt2 yz), whose
    dot product is (n . n')^2. -> torch [6, ...]."""
    r2 = math.sqrt(2.0)
    return torch.stack([n[0] * n[0], n[1] * n[1], n[2] * n[2], r2 * n[0] * n[1], r2 * n[0] * n[2], r2 * n[1] * n[2]])


def gradient_field(arr, mask, affine, sigma_mm, device="cuda"):
    """World-frame gradient [3, D, H, W] (torch, per mm) of arr smoothed with sigma_mm inside mask by normalised convolution;
    zero outside the mask. arr, mask on an isotropic grid (affine voxel -> world)."""
    h = float(np.linalg.norm(np.asarray(affine)[:3, 0]))
    x, m = to_torch(arr, device), to_torch(mask, device)
    g = G.nc_gradient(x * m, m, sigma_mm / h)
    Ainv = torch.as_tensor(np.linalg.inv(np.asarray(affine, float)[:3, :3]).T, dtype=torch.float32, device=device)
    return torch.einsum("ij,jklm->iklm", Ainv, g)


def median_norm(g, mask):
    """eps: the median gradient magnitude over the mask (torch scalar)."""
    return g.norm(dim=0)[to_torch(mask, g.device) > 0].median()


def tensors(g, eps):
    """Orientation tensors of the normalised gradients g / sqrt(|g|^2 + eps^2). -> torch [6, ...]."""
    return orientation_tensor(g / torch.sqrt((g * g).sum(0) + eps * eps))


class MRITensors:
    """The MRI side: the tensor field on its grid, zero outside the mask (or over the whole grid with mask=None)."""

    def __init__(self, arr, mask, affine, sigma_mm, device="cuda", eps_mask=None):
        m = np.ones(np.shape(arr), np.float32) if mask is None else mask
        g = gradient_field(arr, m, affine, sigma_mm, device)
        self.field = tensors(g, median_norm(g, m if eps_mask is None else eps_mask))
        self.affine = np.asarray(affine, float)

    def at(self, y):
        """Tensors at MRI world points y [N, 3] (zero outside the grid). -> [6, N]."""
        return G.sample_world(self.field, self.affine, y)


class OCTTemplate:
    """The OCT side: points of `points` (bool, every `stride`-th voxel per axis) in the world and their world-frame gradients of
    arr smoothed inside `mask`; eps the median |g| over `eps_mask` (default `mask`)."""

    def __init__(self, arr, mask, affine, sigma_mm, points, stride=2, device="cuda", eps_mask=None):
        sel = np.zeros(np.shape(points), bool)
        sel[::stride, ::stride, ::stride] = True
        idx = torch.as_tensor(np.argwhere(np.asarray(points, bool) & sel), device=device)
        if len(idx) < 10:
            raise ValueError("OCT template has too few points")
        g = gradient_field(arr, mask, affine, sigma_mm, device)
        self.eps = median_norm(g, mask if eps_mask is None else eps_mask)
        self.g = g[(slice(None),) + tuple(idx.T)]
        del g
        self.pts = G.apply_affine(np.asarray(affine, float), idx.to(torch.float32))
        self.c = G.box(affine, np.shape(arr))[0]


def correlation(a, b):
    """The centred, normalised correlation of tensor sets a, b [6, N]."""
    da, db = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    return (da * db).sum() / torch.sqrt((da * da).sum() * (db * db).sum() + 1e-24)


def score(T, tpl: OCTTemplate, mri: MRITensors, field=None):
    """F of the pose T (torch 4x4) and, optionally, a displacement field u (callable: MRI world points [N, 3] -> [N, 3]) acting
    after T: the OCT point x goes to y = T x + u(T x). -> torch scalar."""
    y = G.apply_affine(T, tpl.pts)
    g = torch.linalg.inv(T[:3, :3]).T @ tpl.g
    if field is not None:
        E = torch.eye(3, device=y.device)
        h = 0.5                                       # mm, the step of the central differences of u
        J = torch.stack([(field(y + h * E[k]) - field(y - h * E[k])) / (2 * h) for k in range(3)], 2)   # [N, i, k] du_i/dy_k
        g = g - torch.einsum("nki,kn->in", J, g)
        y = y + field(y)
    return correlation(tensors(g, tpl.eps), mri.at(y))


def eroded(mask, mm, h):
    """The voxels of mask farther than mm from its outside (Euclidean, on a grid of spacing h; beyond the array is outside)."""
    return ndimage.distance_transform_edt(np.pad(np.asarray(mask, bool), 1), sampling=h)[1:-1, 1:-1, 1:-1] > mm

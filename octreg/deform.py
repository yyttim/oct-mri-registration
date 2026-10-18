"""Smooth deformation (docs/METHOD.md §5): a displacement d on top of the affine T, fitted to F under a strain bound.

The OCT point x goes to y = T x + d(T x), d in mm along the MRI world axes, trilinear on a lattice of spacing Params.df_grid_mm
over the MRI grid (two spacings of margin); its gradient goes through the linear part of the map, (I - grad d^T) A^-T g. F is
taken over the OCT points within Params.df_band_mm of the specimen mask, with the OCT gradients over all measured voxels and
the MRI gradients over the whole crop, so the tissue surface is an edge in both volumes. Minimised by L-BFGS: -F + STRAIN_WEIGHT
sum max(0, |dc| / df_grid_mm - df_max_strain)^2 over the components dc of the first differences of the lattice along its axes.
Output: the pull-back field u on the MRI base grid, x + u(x) = (id + d)^-1(x), so the registered OCT at the MRI point x is
OCT(T^-1 (x + u(x))).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy import ndimage

from . import geometry as G
from .features import MRITensors, OCTTemplate, score
from .params import Params

STRAIN_WEIGHT = 1e3       # weight of the strain hinge (a soft form of the bound)
INVERSE_ITERS = 20        # fixed-point iterations of the pull-back field
HISTORY = 10              # L-BFGS memory


class _Trilinear(torch.autograd.Function):
    """d = sum over the 8 lattice corners of w c[:, idx]. Its gradient is a segment sum over the corners sorted by node, which is
    deterministic on the GPU (an atomic scatter, as in the backward pass of grid_sample, is not)."""

    @staticmethod
    def forward(ctx, c, idx, w):
        ctx.save_for_backward(idx, w)
        ctx.nodes = c.shape[1]
        return (c[:, idx] * w).sum(-1).T

    @staticmethod
    def backward(ctx, g):
        idx, w = ctx.saved_tensors
        key, order = torch.sort(idx.reshape(-1), stable=True)
        counts = torch.bincount(key, minlength=ctx.nodes)
        v = (g.T[:, :, None] * w).reshape(3, -1)[:, order]
        return torch.stack([torch.segment_reduce(x, "sum", lengths=counts) for x in v]), None, None


class LatticeField(torch.nn.Module):
    """Displacement d(y) (mm, MRI world axes) at world points y [N, 3], trilinear on a lattice of spacing h over lo..hi
    (constant beyond it)."""

    def __init__(self, lo, hi, h, device):
        super().__init__()
        self.lo = torch.tensor(np.asarray(lo, float), dtype=torch.float32, device=device)
        self.h = float(h)
        self.n = [int(math.ceil((b - a) / h)) + 1 for a, b in zip(lo, hi)]
        self.c = torch.nn.Parameter(torch.zeros(3, math.prod(self.n), device=device))

    def forward(self, y):
        n = torch.tensor(self.n, device=y.device)
        q = torch.minimum(((y - self.lo) / self.h).clamp(min=0), n - 1)
        i0 = torch.minimum(q.floor().long(), n - 2)
        f = q - i0
        idx, w = [], []
        for a in (0, 1):
            for b in (0, 1):
                for e in (0, 1):
                    idx.append(((i0[:, 0] + a) * n[1] + i0[:, 1] + b) * n[2] + i0[:, 2] + e)
                    w.append((f[:, 0] if a else 1 - f[:, 0]) * (f[:, 1] if b else 1 - f[:, 1]) * (f[:, 2] if e else 1 - f[:, 2]))
        return _Trilinear.apply(self.c, torch.stack(idx, 1), torch.stack(w, 1))

    def strains(self):
        c = self.c.reshape(3, *self.n)
        return [torch.diff(c, dim=a + 1) / self.h for a in range(3)]


def smooth_deformation(mri, oct, T, params: Params = Params(), device="cuda"):
    """mri = (arr, foreground, affine), oct = (arr, specimen mask, measured fraction, affine) on the base grids, T the affine.
    -> (pull-back field u numpy [3, D, H, W] on the MRI base grid, info)."""
    P = params
    (m_arr, m_mask, A_M), (o_arr, o_mask, o_valid, A_O) = mri, oct
    h_o = float(np.linalg.norm(np.asarray(A_O)[:3, 0]))
    valid = np.asarray(o_valid) > 0.5
    band = (ndimage.distance_transform_edt(~np.asarray(o_mask, bool), sampling=h_o) <= P.df_band_mm) & valid
    n = np.array(np.shape(m_arr)) - 1
    corners = G.apply_affine(np.asarray(A_M, float), np.array([[i, j, k] for i in (0, n[0]) for j in (0, n[1]) for k in (0, n[2])],
                                                              float))
    lo, hi = corners.min(0) - 2 * P.df_grid_mm, corners.max(0) + 2 * P.df_grid_mm
    field = LatticeField(lo, hi, P.df_grid_mm, device)
    Tt = torch.tensor(np.asarray(T, float), dtype=torch.float32, device=device)
    passes = []
    for s in P.df_sigmas_mm:
        tpl = OCTTemplate(o_arr, valid.astype(np.float32), A_O, s, band, device=device, eps_mask=o_mask)
        mt = MRITensors(m_arr, None, A_M, s, device, eps_mask=m_mask)
        opt = torch.optim.LBFGS(field.parameters(), lr=1.0, max_iter=P.df_evals, max_eval=P.df_evals, history_size=HISTORY,
                                line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-12)

        def objective(tpl=tpl, mt=mt):
            field.zero_grad(set_to_none=True)
            loss = -score(Tt, tpl, mt, field) + STRAIN_WEIGHT * sum((torch.relu(x.abs() - P.df_max_strain) ** 2).sum()
                                                                    for x in field.strains())
            loss.backward()
            return loss

        with torch.no_grad():
            f0 = float(score(Tt, tpl, mt, field))
        opt.step(objective)
        with torch.no_grad():
            Fv = score(Tt, tpl, mt, field)
            d = field(G.apply_affine(Tt, tpl.pts)).norm(dim=1)
        passes.append({"sigma_mm": s, "F_start": round(f0, 5), "F_end": round(float(Fv), 5),
                       "displacement_median_mm": round(float(d.median()), 4), "displacement_max_mm": round(float(d.max()), 4)})
        del tpl, mt, objective
    shape = tuple(np.shape(m_arr))
    with torch.no_grad():
        u = np.zeros((3,) + shape, np.float32)
        ijk = torch.stack(torch.meshgrid(*[torch.arange(n, device=device, dtype=torch.float32) for n in shape[1:]], indexing="ij"), -1)
        for i in range(shape[0]):                                    # plane by plane: memory independent of the grid
            p = torch.cat([torch.full_like(ijk[..., :1], float(i)), ijk], -1).reshape(-1, 3)
            x = G.apply_affine(torch.tensor(np.asarray(A_M, float), dtype=torch.float32, device=device), p)
            z = x.clone()
            for _ in range(INVERSE_ITERS):
                z = x - field(z)
            u[:, i] = (z - x).T.reshape(3, *shape[1:]).cpu().numpy()
    return u, {"passes": passes}

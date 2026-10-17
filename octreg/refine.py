"""Affine refinement (docs/METHOD.md §4).

Model x_mri = R(r) Sh(sh) diag(exp(ls)) (x_oct - c) + t, c = OCT grid box centre (mm); proper poses only. Every search
candidate is refined on the structure score F alone, coarse to fine (one Adam pass per sigma of Params.sigmas_mm), under the
prior L = 1 - F + lam (sum ls^2 + sum sh^2), and the lowest final L wins: the search proposes the poses, and the finest scale,
where the internal structure is most specific, decides between them.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from . import geometry as G
from .features import MRITensors, OCTTemplate, eroded, score
from .params import Params


def compose(r, t, ls, sh, c):
    """Torch parameters (rotation vector r rad, t = image of c in mm, log-scales, shears Sh[0,1], Sh[0,2], Sh[1,2]; each [3])
    and c (mm, [3]) -> 4x4 OCT world -> MRI world (torch, differentiable; the rotation is exp([r]x), also at r = 0)."""
    z, one = torch.zeros_like(r[0]), torch.ones_like(r[0])
    K = torch.stack([torch.stack([z, -r[2], r[1]]), torch.stack([r[2], z, -r[0]]), torch.stack([-r[1], r[0], z])])
    Sh = torch.stack([torch.stack([one, sh[0], sh[1]]), torch.stack([z, one, sh[2]]), torch.stack([z, z, one])])
    A = torch.linalg.matrix_exp(K) @ Sh @ torch.diag(torch.exp(ls))
    return torch.cat([torch.cat([A, (t - A @ c)[:, None]], 1), torch.eye(4, dtype=r.dtype, device=r.device)[3:]])


def decompose(T, c):
    """Inverse of compose for a 4x4 numpy affine: A = Q U (QR, diag U > 0) -> (r, t, ls, sh numpy [3]); refuses a mirror."""
    A = np.asarray(T, float)[:3, :3]
    if np.linalg.det(A) <= 0:
        raise ValueError(f"mirrored or singular pose (det <= 0)\n{T}")
    Q, U = np.linalg.qr(A)
    d = np.where(np.diag(U) < 0, -1.0, 1.0)
    Q, U = Q * d, d[:, None] * U
    s = np.diag(U)
    if not s.min() > 1e-9 * s.max():
        raise ValueError(f"singular pose\n{T}")
    Sh = U / s
    return (Rotation.from_matrix(Q).as_rotvec(), A @ np.asarray(c, float) + np.asarray(T, float)[:3, 3], np.log(s),
            np.array([Sh[0, 1], Sh[0, 2], Sh[1, 2]]))


def fit_adam(objective, c, T0, iters, params: Params = Params(), device="cuda"):
    """Adam on L = 1 - objective(T) + lam (sum ls^2 + sum sh^2) from pose T0 (numpy 4x4) over all twelve affine parameters (r,
    t, 3 log-scales, 3 shears), cosine schedule to zero over iters, learning rates lr_rot rad, lr_t mm, lr_ls and lr_sh.
    -> (T numpy 4x4, objective, L) of the iterate with the lowest L."""
    P = params
    f = lambda x: torch.tensor(np.asarray(x, float), dtype=torch.float32, device=device, requires_grad=True)
    r, t, ls, sh = (f(x) for x in decompose(T0, c))
    cc = torch.tensor(np.asarray(c, float), dtype=torch.float32, device=device)
    opt = torch.optim.Adam([{"params": [r], "lr": P.lr_rot}, {"params": [t], "lr": P.lr_t},
                            {"params": [ls], "lr": P.lr_ls}, {"params": [sh], "lr": P.lr_sh}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=0.0)
    best = (math.inf, None, None)
    for i in range(iters + 1):
        opt.zero_grad(set_to_none=True)
        T = compose(r, t, ls, sh, cc)
        v = objective(T)
        L = 1.0 - v + P.lam * ((ls ** 2).sum() + (sh ** 2).sum())
        l, s = float(L.detach()), float(v.detach())
        if l < best[0]:
            best = (l, T.detach().cpu().double().numpy(), s)
        if i == iters:
            break
        L.backward()
        opt.step()
        sched.step()
    return best[1], best[2], best[0]


def refine(starts, mri, oct, params: Params = Params(), device="cuda"):
    """Refine every pose of starts (numpy 4x4, the search candidates in rank order) on F alone, one pass per sigma in
    params.sigmas_mm, coarse to fine. mri = (arr, foreground, affine), oct = (arr, specimen mask, measured fraction, affine) on
    the base grid. The OCT points are the specimen mask eroded by round(erode_mm / h) 6-neighbourhood steps.
    -> [{T, L, F: [{sigma_mm, start, end}], log_scales, shears, search_rank}], lowest final L first."""
    P = params
    (m_arr, m_mask, A_M), (o_arr, o_mask, _, A_O) = mri, oct
    h_o = float(np.linalg.norm(np.asarray(A_O)[:3, 0]))
    inner = eroded(o_mask, P.erode_mm, h_o)
    poses = [{"T": np.asarray(T, float), "F": [], "search_rank": i} for i, T in enumerate(starts)]
    for s in P.sigmas_mm:
        tpl = OCTTemplate(o_arr, o_mask, A_O, s, inner, device=device)
        mt = MRITensors(m_arr, m_mask, A_M, s, device)
        F = lambda x, tpl=tpl, mt=mt: score(x, tpl, mt)
        for p in poses:
            with torch.no_grad():
                f0 = float(F(torch.tensor(p["T"], dtype=torch.float32, device=device)))
            p["T"], f1, p["L"] = fit_adam(F, tpl.c, p["T"], P.iters, P, device)
            p["F"].append({"sigma_mm": s, "start": round(f0, 5), "end": round(f1, 5)})
        del tpl, mt, F
    for p in poses:
        _, _, ls, sh = decompose(p["T"], G.box(A_O, np.shape(o_arr))[0])
        p.update(log_scales=ls.tolist(), shears=sh.tolist())
    return sorted(poses, key=lambda p: p["L"])

"""Affine refinement (docs/METHOD.md §4): every search candidate refined on F under a scale and shear prior, coarse to fine.
The lowest final L wins, a tie (within TIE) going to the better search rank. Model x_mri = R(r) Sh(sh) diag(exp(ls)) (x_oct - c)
+ t, c the OCT grid box centre (mm).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from . import geometry as G
from .features import MRITensors, OCTTemplate, eroded, score
from .params import Params

TIE = 1e-5          # final L closer than this to the lowest count as equal (the round-off of the fit)


def compose(r, t, ls, sh, c):
    """Torch parameters (rotation vector r rad, t = image of c in mm, log-scales, shears Sh[0,1], Sh[0,2], Sh[1,2], each [3])
    and c (mm, [3]) -> 4x4 OCT world -> MRI world (torch, differentiable). The rotation is exp([r]x), also at r = 0."""
    z, one = torch.zeros_like(r[0]), torch.ones_like(r[0])
    K = torch.stack([torch.stack([z, -r[2], r[1]]), torch.stack([r[2], z, -r[0]]), torch.stack([-r[1], r[0], z])])
    Sh = torch.stack([torch.stack([one, sh[0], sh[1]]), torch.stack([z, one, sh[2]]), torch.stack([z, z, one])])
    A = torch.linalg.matrix_exp(K) @ Sh @ torch.diag(torch.exp(ls))
    return torch.cat([torch.cat([A, (t - A @ c)[:, None]], 1), torch.eye(4, dtype=r.dtype, device=r.device)[3:]])


def decompose(T, c):
    """Inverse of compose for a proper 4x4 numpy affine: A = Q U (QR, diag U > 0) -> (r, t, ls, sh numpy [3])."""
    A = np.asarray(T, float)[:3, :3]
    Q, U = np.linalg.qr(A)
    d = np.where(np.diag(U) < 0, -1.0, 1.0)
    Q, U = Q * d, d[:, None] * U
    s = np.diag(U)
    Sh = U / s
    return (Rotation.from_matrix(Q).as_rotvec(), A @ np.asarray(c, float) + np.asarray(T, float)[:3, 3], np.log(s),
            np.array([Sh[0, 1], Sh[0, 2], Sh[1, 2]]))


def fit_adam(objective, c, rho, T0, params: Params = Params(), device="cuda"):
    """Adam on L = 1 - objective(T) + lam (sum ls^2 + sum sh^2) from pose T0 (numpy 4x4) over all twelve affine parameters,
    cosine schedule to zero over params.iters. One step size in mm of corner motion: lr_mm for t, lr_mm / rho for r, ls and sh
    (rho the radius of the OCT box). -> (T numpy 4x4, objective, L) of the last iterate, and the objective at T0."""
    P = params
    f = lambda x: torch.tensor(np.asarray(x, float), dtype=torch.float32, device=device, requires_grad=True)
    r, t, ls, sh = (f(x) for x in decompose(T0, c))
    cc = torch.tensor(np.asarray(c, float), dtype=torch.float32, device=device)
    opt = torch.optim.Adam([{"params": [r, ls, sh], "lr": P.lr_mm / rho}, {"params": [t], "lr": P.lr_mm}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=P.iters, eta_min=0.0)
    for i in range(P.iters + 1):
        opt.zero_grad(set_to_none=True)
        T = compose(r, t, ls, sh, cc)
        v = objective(T)
        L = 1.0 - v + P.lam * ((ls ** 2).sum() + (sh ** 2).sum())
        if i == 0:
            start = float(v.detach())
        if i == P.iters:
            break
        L.backward()
        opt.step()
        sched.step()
    return T.detach().cpu().double().numpy(), float(v.detach()), float(L.detach()), start


def refine(starts, mri, oct, params: Params = Params(), device="cuda"):
    """Refine every pose of starts (numpy 4x4, the search candidates in rank order) on L (F and the scale and shear prior, no
    outline), one pass per sigma in params.sigmas_mm, coarse to fine. mri = (arr, foreground, affine), oct = (arr, specimen
    mask, affine) on the base grid. The OCT points lie farther than erode_mm inside the specimen mask.
    -> [{T, L, F: [{sigma_mm, start, end}], search_rank}]: the poses within TIE of the lowest final L first, by search rank,
    then the others by final L."""
    P = params
    (m_arr, m_mask, A_M), (o_arr, o_mask, A_O) = mri, oct
    h_o = float(np.linalg.norm(np.asarray(A_O)[:3, 0]))
    inner = eroded(o_mask, P.erode_mm, h_o)
    rho = G.box(A_O, np.shape(o_arr))[1]
    poses = [{"T": np.asarray(T, float), "F": [], "search_rank": i} for i, T in enumerate(starts)]
    for s in P.sigmas_mm:
        tpl = OCTTemplate(o_arr, o_mask, A_O, s, inner, device=device)
        mt = MRITensors(m_arr, m_mask, A_M, s, device)
        F = lambda x, tpl=tpl, mt=mt: score(x, tpl, mt)
        for p in poses:
            p["T"], f1, p["L"], f0 = fit_adam(F, tpl.c, rho, p["T"], P, device)
            p["F"].append({"sigma_mm": s, "start": round(f0, 5), "end": round(f1, 5)})
        del tpl, mt, F
    lo = min(p["L"] for p in poses)                 # within TIE of the lowest L the better search rank wins
    return sorted(poses, key=lambda p: (0, p["search_rank"]) if p["L"] <= lo + TIE else (1, p["L"]))

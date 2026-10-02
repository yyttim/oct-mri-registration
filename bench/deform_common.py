"""Shared helpers for the §6 (smooth deformation) bench scripts: ablate_deform.py, ablate_deform_multiscale.py,
fig_deform_boundary.py. Not part of the package; like the rest of bench/, the method never reads this.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F_

from octreg import geometry as G
from octreg.ngf import NGFGrid


def stripe_axis(arr, mask):
    """Array axis along which the OCT has section stripes: the plane-mean intensity is a sawtooth there (largest first differences
    relative to its spread). -> (axis, [score per axis])."""
    scores = []
    for a in range(3):
        other = tuple(x for x in range(3) if x != a)
        cnt = mask.sum(axis=other)
        prof = ((arr * mask).sum(axis=other) / np.maximum(cnt, 1))[cnt > 50]
        scores.append(float(np.std(np.diff(prof)) / (np.std(prof) + 1e-9)))
    return int(np.argmax(scores)), scores


def match_diag(feat_m, feat_o, core, base_mm, *, block_mm, step_mm, range_mm, z_min):
    """octreg.blockmatch.match, returning EVERY block tried with its status, not only the confident ones. -> dict of numpy
    arrays: centre (voxel idx), disp (voxels), z, border (best peak on the border of the range), confident. Same arithmetic
    as the original; kept separate because octreg.blockmatch.match only returns the confident matches."""
    B, R, S = max(2, round(block_mm / base_mm)), max(1, round(range_mm / base_mm)), max(1, round(step_mm / base_mm))
    core = core.to(torch.float32)
    starts = torch.nonzero(F_.avg_pool3d(core[None, None], B, S)[0, 0] >= 0.7) * S
    fo, n, m = F_.pad(feat_o, (R,) * 6), B + 2 * R, 2 * R + 1
    keep = {k: [] for k in ("centre", "disp", "z", "border", "confident")}
    for batch in starts.split(32):
        N, cut = len(batch), lambda x, i, j, k, w: x[..., i:i + w, j:j + w, k:k + w]
        msk = torch.stack([cut(core, *c, B) for c in batch.tolist()])[:, None]
        t = torch.stack([cut(feat_m, *c, B) for c in batch.tolist()])
        f = torch.stack([cut(fo, *c, n) for c in batch.tolist()])
        cnt = msk.sum((1, 2, 3, 4))
        t = (t - (t * msk).sum((2, 3, 4), keepdim=True) / cnt.view(N, 1, 1, 1, 1)) * msk
        num = F_.conv3d(f.reshape(1, N * 6, n, n, n), t, groups=N)[0]
        sf = F_.conv3d(f.reshape(1, N * 6, n, n, n), msk.repeat(1, 6, 1, 1, 1).reshape(N * 6, 1, B, B, B), groups=N * 6)
        sff = F_.conv3d((f * f).sum(1).reshape(1, N, n, n, n), msk, groups=N)[0]
        var = (sff - (sf.reshape(N, 6, m, m, m) ** 2).sum(1) / cnt.view(N, 1, 1, 1)).clamp_min(1e-3 * cnt.view(N, 1, 1, 1))
        sc = num / ((t * t).sum((1, 2, 3, 4)).clamp_min(1e-12).sqrt().view(N, 1, 1, 1) * var.sqrt())
        flat = sc.reshape(N, -1)
        peak, arg = flat.max(1)
        z = (peak - flat.mean(1)) / flat.std(1).clamp_min(1e-12)
        p = torch.stack([arg // (m * m), arg // m % m, arg % m], 1)
        inner = (p > 0).all(1) & (p < 2 * R).all(1)
        keep["centre"].append((batch + (B - 1) / 2).double().cpu().numpy())
        keep["disp"].append((p.to(torch.float32) - R).double().cpu().numpy())
        keep["z"].append(z.double().cpu().numpy())
        keep["border"].append((~inner).cpu().numpy())
        keep["confident"].append(((z >= z_min) & inner).cpu().numpy())
    return {k: np.concatenate(v) for k, v in keep.items()}


def binned(s, val, edges, fn=np.median):
    """fn of val over 2 mm bins of s (edges), None where fewer than 5 points fall in a bin."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (s >= lo) & (s < hi)
        out.append(float(fn(val[sel])) if sel.sum() >= 5 else None)
    return out


def contributions(grid, T):
    Ti = torch.linalg.inv(T)
    x = G.apply_affine(Ti, grid.pts)
    g = Ti[:3, :3].T @ G.sample_world(grid.gO, grid.A_O, x)
    w = G.sample_world(grid.w, grid.A_O, x)[0]
    n = g / torch.sqrt((g * g).sum(0) + grid.eps_O ** 2)
    return (w * ((n * grid.nM).sum(0)) ** 2).cpu().numpy(), w.cpu().numpy()


def f_vs_chance(mri, arr, mask, A_m, P, device, keep_pts):
    """F (fine-structure NGF agreement, sigma 0.3 mm) of an OCT volume on the MRI base grid against the MRI (identity pose),
    overall and over the MRI points in keep_pts(points)->bool, each against the mean F of six 10 mm-shifted poses (chance)."""
    grid = NGFGrid(mri, (arr, mask, A_m), 0.3, P, device)
    I4 = torch.eye(4, device=device)
    pts = grid.pts.cpu().numpy()
    sel = keep_pts(pts)
    res = {}
    with torch.no_grad():
        runs = [("actual", *contributions(grid, I4))]
        for d in [s * e for e in np.eye(3) for s in (10.0, -10.0)]:
            Tn = I4.clone()
            Tn[:3, 3] += torch.tensor(d, dtype=torch.float32, device=device)
            runs.append(("null", *contributions(grid, Tn)))
    for name, s_ in (("all", np.ones(len(pts), bool)), ("excl_folia", sel)):
        F = [c[s_].sum() / max(w[s_].sum(), 1.0) for kind, c, w in runs if kind == "actual"][0]
        Fn = float(np.mean([c[s_].sum() / max(w[s_].sum(), 1.0) for kind, c, w in runs if kind == "null"]))
        res[name] = {"F": float(F), "chance": Fn, "ratio": float(F / Fn)}
    del grid
    return res


def load_grids(out):
    """The §1 prep grids of bench/ablate.py's `prepare(out, ["texture"])`: mri = (arr, mask, affine), oct = (arr, mask, affine),
    both on the base grid (Params.base_mm), in the format octreg.deform.Evidence and octreg.register.register expect."""
    with np.load(out / "prep" / "texture" / "oct_h.npz") as z, np.load(out / "prep" / "mri" / "mri_h.npz") as y:
        oct_grid = (z["arr"], z["mask"], z["affine"])
        mri_grid = (y["arr"], y["mask"], y["affine"])
    return mri_grid, oct_grid

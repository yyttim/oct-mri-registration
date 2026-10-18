"""Orientation search (docs/METHOD.md §3): for each rotation of a near-uniform set, F and the outline term O at every
translation by FFT, in the masked form of Padfield (IEEE Trans. Image Process. 21:2706, 2012)."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from . import geometry as G
from .features import eroded, orientation_tensor, to_torch
from .params import Params

EPS = 1e-6
MIN_POINTS = 10
DIMS = (-3, -2, -1)


def angle_deg(R1, R2):
    """Rotation angle (deg) between two rotation matrices."""
    return math.degrees(math.acos(np.clip((np.trace(np.asarray(R1) @ np.asarray(R2).T) - 1) / 2, -1.0, 1.0)))


class Searcher:
    """Score maps of one orientation at a time against the zero-padded MRI grid."""

    def __init__(self, mri, mri_mask, mri_affine, oct_, oct_w, oct_valid, oct_affine, sigma_mm, erode_mm, device="cuda"):
        """mri and mri_mask (foreground fraction M) on the isotropic grid mri_affine, oct_, oct_w (specimen mask) and
        oct_valid (measured fraction) on oct_affine. The OCT points lie farther than erode_mm inside the specimen mask."""
        self.dev = device
        I, M, o, w, q = (to_torch(x, device) for x in (mri, mri_mask, oct_, oct_w, oct_valid))
        A_M, A_O = np.asarray(mri_affine, float), np.asarray(oct_affine, float)
        if not float(w.sum()) > 0:
            raise ValueError("OCT specimen mask is empty on the search grid")
        if not bool((M > 0.5).any()):
            raise ValueError("no MRI foreground on the search grid")
        h = float(np.linalg.norm(A_M[:3, :3], axis=0).mean())
        self.sig = sigma_mm / h
        self.to_world = torch.as_tensor(np.linalg.inv(A_M[:3, :3]).T, dtype=torch.float32, device=device)
        h_o = float(np.linalg.norm(A_O[:3, :3], axis=0).mean())
        inner = torch.as_tensor(eroded(oct_w > 0.5, erode_mm, h_o), dtype=torch.float32, device=device)
        if int(inner.sum()) < MIN_POINTS or int((M > 0.5).sum()) < MIN_POINTS:
            raise ValueError(f"fewer than {MIN_POINTS} OCT points ({erode_mm} mm inside the specimen mask) or MRI foreground "
                             "voxels on the search grid")
        Mb = (M > 0.5).float()  # gradients inside the binary mask, as in features
        gM = torch.einsum("ij,jklm->iklm", self.to_world, G.nc_gradient(I * Mb, Mb, self.sig))
        eps_M = gM.norm(dim=0)[Mb > 0].median()
        TM = orientation_tensor(gM / torch.sqrt((gM * gM).sum(0) + eps_M ** 2))
        # gradients inside the erosion, unlike §2: at this scale medium inside the mask adds an edge reaching the points
        gO = G.nc_gradient(o * inner, inner, sigma_mm / h_o)
        self.eps_O = gO.norm(dim=0)[inner > 0].median() / h_o         # per mm, as the template gradients
        self.c_o, radius = G.box(A_O, w.shape)
        self.n = n = math.ceil(2 * radius / h) + 3
        pad = n // 2 + 1                                               # so a block over the edge of the crop is still scored
        self.A_M = A_M.copy()
        self.A_M[:3, 3] -= A_M[:3, :3] @ np.full(3, float(pad))
        TMp = F.pad(TM[None], [pad] * 6)[0]
        Mp = F.pad(M[None], [pad] * 6)[0]
        self.shape = sh = tuple(Mp.shape)
        self.FT, self.FT2 = torch.fft.rfftn(TMp, dim=DIMS), torch.fft.rfftn((TMp * TMp).sum(0), dim=DIMS)
        self.FM, self.FM2, self.FM3 = (torch.fft.rfftn(Mp ** k, dim=DIMS) for k in (1, 2, 3))
        self.FE = torch.fft.rfftn(F.pad(M[None], [pad] * 6, value=1.0)[0], dim=DIMS)             # E: M in the crop, 1 outside
        # variance floor of M per unit weight: O stays near 0 where M is almost constant under the template
        self.var_floor = float(Mp.var()) * 0.02
        del TMp, Mp, TM, gM
        self.src = F.pad(torch.stack([o, w, q, inner])[None], [1] * 6)[0]
        self.A_src = A_O.copy()
        self.A_src[:3, 3] -= A_O[:3, :3] @ np.ones(3)
        k = torch.arange(n, dtype=torch.float32, device=device) - (n - 1) / 2.0
        L_M = torch.as_tensor(self.A_M[:3, :3], dtype=torch.float32, device=device)
        self.tgrid = torch.stack(torch.meshgrid(k, k, k, indexing="ij"), -1).reshape(-1, 3) @ L_M.T  # offsets on MRI axes, mm
        self.c_t = torch.as_tensor(self.c_o, dtype=torch.float32, device=device)
        self.valid = torch.zeros(sh, dtype=torch.bool, device=device)                                 # template inside the grid
        self.valid[:sh[0] - n + 1, :sh[1] - n + 1, :sh[2] - n + 1] = True

    def _ft(self, x):
        """Conjugate FFT of a template x [..., n, n, n] placed at the grid origin."""
        xp = x.new_zeros(x.shape[:-3] + self.shape)
        xp[..., :self.n, :self.n, :self.n] = x
        return torch.fft.rfftn(xp, dim=DIMS).conj()

    def _inv(self, X):
        return torch.fft.irfftn(X, s=self.shape, dim=DIMS)

    @torch.no_grad()
    def score(self, R, outline=True):
        """Orientation R (numpy 3x3) -> (F, O) maps [D', H', W'], O None without the outline.
        Index u means x_mri = R (x_oct - c_o) + t_u. O weights the OCT voxels by q (1 - (1 - w) E), so medium over MRI
        tissue costs nothing, as the crop may hold tissue beyond a cut face (docs/METHOD.md §2)."""
        n = self.n
        pts = self.c_t + self.tgrid @ torch.as_tensor(np.asarray(R), dtype=torch.float32, device=self.dev)   # x_o = c_o + R^T y
        v, w, q, inner = G.sample_world(self.src, self.A_src, pts).reshape(4, n, n, n)
        wt = (inner > 0.5).float()                                    # the OCT points, also the support of their gradients
        g = torch.einsum("ij,jklm->iklm", self.to_world, G.nc_gradient(v * wt, wt, self.sig))
        T = orientation_tensor(g / torch.sqrt((g * g).sum(0) + self.eps_O ** 2))
        N = wt.sum() + EPS
        Fw, FwT = self._ft(wt), self._ft(wt[None] * T)
        ST, STT = (wt[None] * T).sum(DIMS), (wt[None] * T * T).sum()
        SI = self._inv(Fw[None] * self.FT)
        num = self._inv((FwT * self.FT).sum(0)) - (ST.reshape(-1, 1, 1, 1) * SI).sum(0) / N
        varI = (self._inv(Fw * self.FT2) - (SI * SI).sum(0) / N).clamp(min=EPS * N)
        Fmap = num / torch.sqrt(((STT - (ST * ST).sum() / N) * varI).clamp(min=EPS))
        if not outline:
            return Fmap, None
        b = q * (1 - w)                       # weight q - b E: with E M = M^2 and E M^2 = M^3 every sum is one correlation
        Fq, Fb, Fqw, Fbw, Fbww = (self._ft(x) for x in (q, b, q * w, b * w, b * w * w))
        Nq = q.sum() - self._inv(Fb * self.FE) + EPS
        Sw, Sww = (q * w).sum() - self._inv(Fbw * self.FE), (q * w * w).sum() - self._inv(Fbww * self.FE)
        SM, SMM = self._inv(Fq * self.FM - Fb * self.FM2), self._inv(Fq * self.FM2 - Fb * self.FM3)
        SwM = self._inv(Fqw * self.FM - Fbw * self.FM2)
        varM = torch.maximum(SMM - SM * SM / Nq, self.var_floor * Nq)
        Omap = (SwM - Sw * SM / Nq) / torch.sqrt(((Sww - Sw * Sw / Nq) * varM).clamp(min=EPS))
        return Fmap, Omap

    def pose(self, R, index):
        """4x4 OCT world -> MRI world (mm) of orientation R at the flat translation index."""
        T = np.eye(4)
        T[:3, :3] = np.asarray(R, float)
        centre = np.array(np.unravel_index(int(index), self.shape), float) + (self.n - 1) / 2.0  # template centre, MRI voxels
        T[:3, 3] = G.apply_affine(self.A_M, centre) - T[:3, :3] @ self.c_o
        return T


def search(mri, mri_mask, mri_affine, oct_, oct_w, oct_valid, oct_affine, params: Params = Params(), device="cuda", outline=True):
    """Orientation search over geometry.rotations(n_rot), arguments as Searcher. -> candidates [{T 4x4, F, O}]
    After non-maximum suppression, with the outline the best topk // 2 by S = (F + O) / 2 come first, then the best by F up
    to topk. Without the outline O is None."""
    P = params
    s = Searcher(mri, mri_mask, mri_affine, oct_, oct_w, oct_valid, oct_affine, P.search_sigma_mm, P.erode_mm, device)
    Rs = G.rotations(P.n_rot)
    lists = ([], []) if outline else ([],)
    for i, R in enumerate(Rs):
        Fm, Om = s.score(R, outline)
        for found, X in zip(lists, (((Fm + Om) / 2) if outline else Fm, Fm)):
            j = int(torch.argmax(X.masked_fill(~s.valid, float("-inf"))))
            o = None if Om is None else float(Om.reshape(-1)[j])
            found.append((float(X.reshape(-1)[j]), i, j, float(Fm.reshape(-1)[j]), o))
    kept = []
    for found, n in zip(lists, (P.topk // 2, P.topk) if outline else (P.topk,)):
        for x, i, j, f, o in sorted(found, key=lambda e: -e[0]):
            if len(kept) >= n:
                break
            T = s.pose(Rs[i], j)
            centre = G.apply_affine(T, s.c_o)
            if not any(np.linalg.norm(centre - c) < P.nms_mm and angle_deg(Rs[i], R) < P.nms_deg for _, c, R in kept):
                kept.append((dict(T=T, F=f, O=o), centre, Rs[i]))
    return [k[0] for k in kept]

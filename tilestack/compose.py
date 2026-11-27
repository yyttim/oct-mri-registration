"""The registered volume in the common frame from all slabs and their tile placements.

Every output voxel takes the newest slab whose cut surface lies at least MARGIN planes above it and that holds it in
its window, raw planes down to data.DEEP_A: the slab that imaged it nearest its cut, after a deep cut the deep planes
of the slab above. The voxel holds that slab's raw mu_s resampled at the registered position, trilinear in mu_s. Each
slab is resampled tile by tile: an output pixel takes the tile whose moved cell contains it (the one deepest inside its
cell where two moved cells overlap), and none where moved cells leave a gap, where the volume takes another slab. Along
depth the raw plane of a common-frame voxel follows the slab's depth offset. Nothing else changes a value: no brightness correction, no blending at the seams, no
filling of empty voxels, which stay 0. A source map records the slab of every voxel, +1000 where its raw plane lies
below the reliable planes (data.DEEP), -1 for no data. The output grid has the voxel of the slab files and is their
field with a margin of data.PAD working pixels on every side. The cut, the margin and the planes are taken on the
working grid, where the placement is solved; the coverage and the values are read from the slab files.
"""
import numpy as np
import torch
import torch.nn.functional as Fnn

from . import data
from .solve import Spline, displacement

dev = "cuda"
BELOW = 1000                                           # source-map offset of the planes below the reliable ones


def _spline(c, y, x, G):
    """A cubic B-spline with control values c (gy, gx) at raw points (torch tensors), as solve.Spline."""
    fy, fx = y.clamp(0, data.NY - 1e-3) / G, x.clamp(0, data.NX - 1e-3) / G
    jy, jx = fy.floor().long(), fx.floor().long()
    out = torch.zeros_like(y)
    for a, wy in enumerate(_b3(fy - jy)):
        for b, wx in enumerate(_b3(fx - jx)):
            out += wy * wx * c[jy + a, jx + b]
    return out


def _b3(u):
    u2, u3 = u * u, u * u * u
    return ((1 - u) ** 3 / 6, (3 * u3 - 6 * u2 + 4) / 6, (-3 * u3 + 3 * u2 + 3 * u + 1) / 6, u3 / 6)


def inverse_map(sol, lay, s, ny, nx, scale):
    """Slab (y, x) in working pixels, the tile and the common-frame depth offset (working planes) for every output pixel
    of slab s, the output pixel being scale working pixels;
    tile -1 where no tile of the slab lands. Per tile the forward map is the translation plus
    the smooth field, inverted by fixed-point iteration (the field sampled on a 16 px lattice); the depth offset is
    the slab's at the raw position found."""
    k = sol["slabs"].index(s)
    a = sol["aff"][k]
    M = np.eye(2) + np.array([[a[0, 1], a[0, 2]], [a[1, 1], a[1, 2]]]) / 1000
    Mi = torch.tensor(np.linalg.inv(M), dtype=torch.float32, device=dev)
    t0 = torch.tensor([a[0, 0] - (a[0, 1] * data.XC + a[0, 2] * data.YC) / 1000, a[1, 0] - (a[1, 1] * data.XC + a[1, 2] * data.YC) / 1000],
                      dtype=torch.float32, device=dev)
    Li = torch.tensor(np.linalg.inv(lay._A), dtype=torch.float32, device=dev)
    E = torch.tensor(sol["tiles"][k], dtype=torch.float32, device=dev)            # (n, 3)
    ok_grid = torch.zeros((lay.rows + 2, lay.cols + 2), dtype=torch.bool, device=dev)    # present tiles, padded
    ok_grid[1:-1, 1:-1] = torch.as_tensor(lay.present(s), device=dev)
    cmap = torch.as_tensor(lay.cellmap(s), device=dev)
    Y = (torch.arange(ny, device=dev).float() + 0.5) * scale - 0.5 - data.PAD
    X = (torch.arange(nx, device=dev).float() + 0.5) * scale - 0.5 - data.PAD
    Y, X = torch.meshgrid(Y, X, indexing="ij")
    field = None
    if sol["field"] is not None:              # the in-plane field sampled on a 16 px lattice, bilinear in between
        gy, gx = np.arange(0, data.NY + 16, 16.0), np.arange(0, data.NX + 16, 16.0)
        GY, GX = np.meshgrid(gy, gx, indexing="ij")
        spl = Spline(sol["G"])
        idx, w = spl.weights(GY.ravel(), GX.ravel())
        Fd = torch.tensor(np.stack([(sol["field"][k, i][idx] * w).sum(1).reshape(GY.shape) for i in range(2)]), dtype=torch.float32, device=dev)

        def sampled(px, py):
            g = torch.stack([px / (gx[-1]) * 2 - 1, py / (gy[-1]) * 2 - 1], -1)[None]
            return Fnn.grid_sample(Fd[None], g, align_corners=True, padding_mode="border")[0]
        field = sampled

    def lattice(p):
        c = torch.stack([p[0] - lay.X0 - lay.W / 2, p[1] - lay.Y0 - lay.H / 2])
        ji = torch.einsum("ab,bhw->ahw", Li, c)
        return ji[1], ji[0]
    q = torch.stack([X - t0[0], Y - t0[1]])
    p0 = torch.einsum("ab,bhw->ahw", Mi, q)
    if field is not None:
        for _ in range(3):
            p0 = torch.einsum("ab,bhw->ahw", Mi, q - field(p0[0], p0[1]))
    i0, j0 = (torch.round(v) for v in lattice(p0))
    best = torch.full_like(X, -1e9); cell = torch.full(X.shape, -1, dtype=torch.long, device=dev)
    py, px = Y.clone(), X.clone()
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ii = (i0 + di).long(); jj = (j0 + dj).long()
            inside = (ii >= 0) & (ii < lay.rows) & (jj >= 0) & (jj < lay.cols)
            ic, jc = ii.clamp(0, lay.rows - 1), jj.clamp(0, lay.cols - 1)
            ok = inside & ok_grid[ic + 1, jc + 1]
            kc = torch.where(ok, ic * lay.cols + jc, torch.zeros_like(ii))
            e = E[kc][..., :2].permute(2, 0, 1)
            p = torch.einsum("ab,bhw->ahw", Mi, q - e - (field(p0[0], p0[1]) if field is not None else 0))
            if field is not None:
                for _ in range(2):
                    p = torch.einsum("ab,bhw->ahw", Mi, q - e - field(p[0], p[1]))
            fi, fj = lattice(p)
            ci = torch.round(fi).long().clamp(0, lay.rows - 1); cj = torch.round(fj).long().clamp(0, lay.cols - 1)
            ok &= cmap[ci, cj] == kc
            dI = fi - ii; dJ = fj - jj                      # sides without a present neighbour tile are open
            dI = torch.where(~ok_grid[ic, jc + 1], torch.clamp(dI, min=0), dI)
            dI = torch.where(~ok_grid[ic + 2, jc + 1], torch.clamp(dI, max=0), dI)
            dJ = torch.where(~ok_grid[ic + 1, jc], torch.clamp(dJ, min=0), dJ)
            dJ = torch.where(~ok_grid[ic + 1, jc + 2], torch.clamp(dJ, max=0), dJ)
            margin = torch.where(ok, 0.5 - torch.maximum(dI.abs(), dJ.abs()), torch.full_like(dI, -1e9))
            take = margin > best
            best = torch.where(take, margin, best); cell = torch.where(take, kc, cell)
            px = torch.where(take, p[0], px); py = torch.where(take, p[1], py)
    cell = torch.where(best > -1e8, cell, torch.full_like(cell, -1))
    zo = a[2, 0] + a[2, 1] * (px - data.XC) / 1000 + a[2, 2] * (py - data.YC) / 1000 + E[torch.clamp(cell, min=0), 2]
    if sol["field"] is not None:
        c = torch.tensor(sol["field"][k, 2].reshape(spl.gy, spl.gx), dtype=torch.float32, device=dev)
        zo = zo + _spline(c, py, px, sol["G"])
    return py, px, cell, zo


def compose(sol, lay, out_path):
    """Returns (volume mu_s float32 (Z, Y, X) at the voxel of the slab files, written to the .npy out_path, 0 where no
    data, z0 = the common working plane at which plane 0 begins, source map int16)."""
    slabs = sol["slabs"]
    fz, fy, fx = data.FACTOR
    nz_r, ny_r, nx_r = data.SHAPE
    reach = 0.0
    for s in slabs:                                  # how far the placement moves the raw field: it must stay in PAD
        yy, xx = (v.ravel().astype(float) for v in np.mgrid[0:data.NY:16, 0:data.NX:16])
        u, v, _ = displacement(sol, lay, s, xx, yy)
        reach = max(reach, float(np.abs(u).max()), float(np.abs(v).max()))
    assert reach < data.PAD, f"a slab is placed {reach:.0f} px beyond the raw field, more than data.PAD ({data.PAD})"
    ny, nx = ny_r + 2 * data.PAD * fy, nx_r + 2 * data.PAD * fx
    lo, hi = np.inf, -np.inf
    for s in slabs:                                  # the depth range of the volume, from the placement on a 16 px grid
        surf, valid = data.geometry(s)
        yy, xx = (v[valid[::16, ::16]].astype(float) for v in np.mgrid[0:data.NY:16, 0:data.NX:16])
        w = displacement(sol, lay, s, xx, yy)[2]
        lo = min(lo, float(w.min()) + float(surf.min()) - 4)
        hi = max(hi, float(w.max()) + data.NP + 3)
    z0 = int(np.floor(lo)); NZ = int(np.ceil(hi)) - z0                 # working planes
    vol = np.lib.format.open_memmap(out_path, "w+", np.float32, (NZ * fz, ny, nx))
    src = np.full((NZ * fz, ny, nx), -1, np.int16)
    for s in slabs:                                                  # newer slabs overwrite older ones
        Mt = torch.from_numpy(data.load(s)).to(dev)
        St = torch.from_numpy(np.ascontiguousarray(data.geometry(s)[0])).float().to(dev)
        Vt = (Mt[nz_r // 2] > 0).float()                                # coverage: the middle plane holds data
        y, x, cell, zo = inverse_map(sol, lay, s, ny, nx, 1 / fy)      # working pixels of every output pixel
        g2 = torch.stack([x / (data.NX - 1) * 2 - 1, y / (data.NY - 1) * 2 - 1], -1)[None]
        gy = ((y + 0.5) * fy - 0.5) / (ny_r - 1) * 2 - 1               # the same points on the files' grid
        gx = ((x + 0.5) * fx - 0.5) / (nx_r - 1) * 2 - 1
        surf_s = Fnn.grid_sample(St[None, None], g2, align_corners=True, padding_mode="border")[0, 0]
        val_s = (Fnn.grid_sample(Vt[None, None], torch.stack([gx, gy], -1)[None], align_corners=True,
                                 padding_mode="zeros")[0, 0] > 0.99) & (cell >= 0)
        Zlo = int(np.floor(float((zo + surf_s)[val_s].min()))) if val_s.any() else z0
        Zhi = int(np.ceil(float(zo[val_s].max()))) + data.DEEP_A + 2 if val_s.any() else z0
        for i0 in range((max(Zlo, z0) - z0) * fz, (min(Zhi, z0 + NZ) - z0) * fz, 16):
            i = torch.arange(i0, min(i0 + 16, (min(Zhi, z0 + NZ) - z0) * fz), device=dev).float()
            rz = (z0 + (i + 0.5) / fz - 0.5)[:, None, None] - zo[None]   # the working raw plane of every output plane
            gz = (rz * fz + (fz - 1) / 2) / (nz_r - 1) * 2 - 1
            g3 = torch.stack([gx[None].expand_as(rz), gy[None].expand_as(rz), gz], -1)[None]
            val = Fnn.grid_sample(Mt[None, None], g3, align_corners=True, padding_mode="border")[0, 0]
            ok = (val_s[None] & (rz - surf_s[None] >= data.MARGIN) & (rz <= data.DEEP_A)).cpu().numpy()
            i1 = i0 + len(i)
            vol[i0:i1][ok] = val.cpu().numpy()[ok]
            src[i0:i1][ok] = (s + BELOW * (rz > data.DEEP)).to(torch.int16).cpu().numpy()[ok]
        del Mt
        torch.cuda.empty_cache()
    vol.flush()
    return vol, z0, src

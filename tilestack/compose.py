"""The registered volume in the common frame from all slabs and their tile placements (docs/TILESTACK.md, Volume).

Each slab is resampled tile by tile: an output pixel takes the tile whose moved cell holds it, the one it lies farthest
inside where moved cells overlap, and none where moved cells leave a gap, where the volume takes another slab.
Along depth the raw plane of a common-frame voxel follows the slab's depth offset. The output grid is the raw field
with a margin of data.PAD pixels on every side.
"""
import numpy as np
import torch
import torch.nn.functional as Fnn

from . import BELOW, data, dev
from .solve import Spline, _b3, displacement


def _spline(c, y, x, G):
    """A cubic B-spline with control values c (gy, gx) at raw points (torch tensors), as solve.Spline."""
    fy, fx = y.clamp(0, data.NY - 1e-3) / G, x.clamp(0, data.NX - 1e-3) / G
    jy, jx = fy.floor().long(), fx.floor().long()
    out = torch.zeros_like(y)
    for a, wy in enumerate(_b3(fy - jy)):
        for b, wx in enumerate(_b3(fx - jx)):
            out += wy * wx * c[jy + a, jx + b]
    return out


def inverse_map(sol, lay, s, ny, nx, scale):
    """Raw (y, x) in raw pixels, the tile and the common-frame depth offset (planes) for every output pixel of slab s,
    tile -1 where no tile of the slab lands. Per tile the forward map is the translations plus the smooth field,
    inverted by fixed-point iteration with the field sampled on a 16 px lattice, bilinear in between. The depth offset
    is the slab's at the raw position found."""
    k = sol["slabs"].index(s)
    a = torch.tensor(sol["shift"][k], dtype=torch.float32, device=dev)
    Li = torch.tensor(np.linalg.inv(lay._A), dtype=torch.float32, device=dev)
    E = torch.tensor(sol["tiles"][k], dtype=torch.float32, device=dev)            # (n, 3)
    cpad = torch.as_tensor(np.pad(lay.cellmap(s), 1, mode="edge"), device=dev)    # tile of every cell, padded
    cmap = torch.as_tensor(lay.cellmap(s), device=dev)
    Y = (torch.arange(ny, device=dev).float() + 0.5) * scale - 0.5 - data.PAD
    X = (torch.arange(nx, device=dev).float() + 0.5) * scale - 0.5 - data.PAD
    Y, X = torch.meshgrid(Y, X, indexing="ij")
    gy, gx = np.arange(0, data.NY + 16, 16.0), np.arange(0, data.NX + 16, 16.0)
    GY, GX = np.meshgrid(gy, gx, indexing="ij")
    spl = Spline(sol["G"])
    idx, w = spl.weights(GY.ravel(), GX.ravel())
    Fd = torch.tensor(np.stack([(sol["field"][k, i][idx] * w).sum(1).reshape(GY.shape) for i in range(2)]), dtype=torch.float32, device=dev)

    def field(px, py):
        g = torch.stack([px / (gx[-1]) * 2 - 1, py / (gy[-1]) * 2 - 1], -1)[None]
        return Fnn.grid_sample(Fd[None], g, align_corners=True, padding_mode="border")[0]

    def lattice(p):
        c = torch.stack([p[0] - lay.X0 - lay.W / 2, p[1] - lay.Y0 - lay.H / 2])
        ji = torch.einsum("ab,bhw->ahw", Li, c)
        return ji[1], ji[0]
    q = torch.stack([X - a[0], Y - a[1]])
    p0 = q
    for _ in range(3):
        p0 = q - field(p0[0], p0[1])
    i0, j0 = lattice(p0)
    i0, j0 = torch.round(i0).clamp(0, lay.rows - 1), torch.round(j0).clamp(0, lay.cols - 1)
    best = torch.full_like(X, -1e9); cell = torch.full(X.shape, -1, dtype=torch.long, device=dev)
    py, px = Y.clone(), X.clone()
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ii = (i0 + di).long(); jj = (j0 + dj).long()
            inside = (ii >= 0) & (ii < lay.rows) & (jj >= 0) & (jj < lay.cols)
            ic, jc = ii.clamp(0, lay.rows - 1), jj.clamp(0, lay.cols - 1)
            kc, ok = cmap[ic, jc], inside                       # the tile that holds that cell
            e = E[kc][..., :2].permute(2, 0, 1)
            p = p0
            for _ in range(3):
                p = q - e - field(p[0], p[1])
            fi, fj = lattice(p)
            ci = torch.round(fi).long().clamp(0, lay.rows - 1); cj = torch.round(fj).long().clamp(0, lay.cols - 1)
            ok &= cmap[ci, cj] == kc
            u, v = fi - ci, fj - cj                         # the distance to the nearest cell of another tile
            margin = torch.full_like(u, 0.5)
            for si, sj in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                d = torch.hypot(torch.clamp(0.5 - si * u, min=0) if si else torch.zeros_like(u),
                                torch.clamp(0.5 - sj * v, min=0) if sj else torch.zeros_like(v))
                margin = torch.where(cpad[ci + 1 + si, cj + 1 + sj] == kc, margin, torch.minimum(margin, d))
            margin = torch.where(ok, margin, torch.full_like(u, -1e9))
            take = margin > best
            best = torch.where(take, margin, best); cell = torch.where(take, kc, cell)
            px = torch.where(take, p[0], px); py = torch.where(take, p[1], py)
    cell = torch.where(best > -1e8, cell, torch.full_like(cell, -1))
    c = torch.tensor(sol["field"][k, 2].reshape(spl.gy, spl.gx), dtype=torch.float32, device=dev)
    return py, px, cell, a[2] + E[torch.clamp(cell, min=0), 2] + _spline(c, py, px, sol["G"])


def compose(sol, lay, full=False, out_path=None):
    """Returns (volume mu_s float32 (Z, Y, X), 0 where no data, z0 = common plane of index 0, source map int16), on the
    working grid, or at the raw voxel with full (written to the .npy out_path when given)."""
    slabs = sol["slabs"]
    scale = 1 if full else data.DOWN
    reach = 0.0
    for s in slabs:                                  # how far the placement moves the raw field: it must stay in PAD
        yy, xx = (v.ravel().astype(float) for v in np.mgrid[0:data.NY:16, 0:data.NX:16])
        u, v, _ = displacement(sol, lay, s, xx, yy)
        reach = max(reach, float(np.abs(u).max()), float(np.abs(v).max()))
    assert reach < data.PAD, f"a slab is placed {reach:.0f} px beyond the raw field, more than data.PAD ({data.PAD})"
    ny, nx = (data.NY + 2 * data.PAD) // scale, (data.NX + 2 * data.PAD) // scale
    lo, hi = np.inf, -np.inf
    for s in slabs:                                  # the depth range of the volume, from the placement on a 16 px grid
        surf, valid = data.geometry(s)
        yy, xx = (v[valid[::16, ::16]].astype(float) for v in np.mgrid[0:data.NY:16, 0:data.NX:16])
        w = displacement(sol, lay, s, xx, yy)[2]
        lo = min(lo, float(w.min()) + float(surf.min()) - 4)
        hi = max(hi, float(w.max()) + data.NP + 3)
    z0 = int(np.floor(lo)); NZ = int(np.ceil(hi)) - z0
    vol = np.lib.format.open_memmap(out_path, "w+", np.float32, (NZ, ny, nx)) if out_path else np.zeros((NZ, ny, nx), np.float32)
    src = np.full((NZ, ny, nx), -1, np.int16)
    for s in slabs:                                                  # newer slabs overwrite older ones
        Mv = data.load(s) if full else data.load40(s)
        surf, valid = data.geometry(s) if full else data.geometry40(s)
        Mt = torch.from_numpy(Mv).to(dev)
        nzr, my, mx = Mv.shape
        del Mv
        St = torch.from_numpy(np.ascontiguousarray(surf)).float().to(dev)
        Vt = torch.from_numpy(valid.astype(np.float32)).to(dev)
        y, x, cell, zo = inverse_map(sol, lay, s, ny, nx, scale)
        ly = (y + 0.5) / scale - 0.5; lx = (x + 0.5) / scale - 0.5                  # raw pixels at this grid
        gy = ly / (my - 1) * 2 - 1; gx = lx / (mx - 1) * 2 - 1
        g2 = torch.stack([gx, gy], -1)[None]
        surf_s = Fnn.grid_sample(St[None, None], g2, align_corners=True, padding_mode="border")[0, 0]
        val_s = (Fnn.grid_sample(Vt[None, None], g2, align_corners=True, padding_mode="zeros")[0, 0] > 0.99) & (cell >= 0)
        Zlo = int(np.floor(float((zo + surf_s)[val_s].min()))) if val_s.any() else z0
        Zhi = int(np.ceil(float(zo[val_s].max()))) + data.DEEP_A + 2 if val_s.any() else z0
        for Zc in range(max(Zlo, z0), min(Zhi, z0 + NZ), 16):
            Zs = torch.arange(Zc, min(Zc + 16, Zhi, z0 + NZ), device=dev).float()
            rz = Zs[:, None, None] - zo[None]                                # the raw plane of every common plane
            gz = rz / (nzr - 1) * 2 - 1
            g3 = torch.stack([gx[None].expand_as(rz), gy[None].expand_as(rz), gz], -1)[None]
            val = Fnn.grid_sample(Mt[None, None], g3, align_corners=True, padding_mode="border")[0, 0]
            ok = (val_s[None] & (rz - surf_s[None] >= data.MARGIN) & (rz <= data.DEEP_A)).cpu().numpy()
            i0, i1 = int(Zs[0]) - z0, int(Zs[-1]) - z0 + 1
            vol[i0:i1][ok] = val.cpu().numpy()[ok]
            src[i0:i1][ok] = (s + BELOW * (rz > data.DEEP)).to(torch.int16).cpu().numpy()[ok]
        del Mt
        torch.cuda.empty_cache()
    if out_path:
        vol.flush()
    return vol, z0, src

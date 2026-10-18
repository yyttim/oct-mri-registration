"""Grids, rotations, world-space sampling and separable Gaussian filtering.
An affine maps voxel index (i, j, k) -> world mm, and a transform T is a 4x4 world -> world matrix (mm)."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.spatial.transform import Rotation

from . import io

CHUNK = 1 << 22                                  # destination voxels per resample_to chunk, memory only


def _np(A) -> np.ndarray:
    return A.detach().cpu().double().numpy() if isinstance(A, torch.Tensor) else np.asarray(A, float)


def rotations(n) -> np.ndarray:
    """n Super-Fibonacci rotations (Alexa, CVPR 2022), a deterministic near-uniform set on SO(3). -> [n, 3, 3]"""
    s = np.arange(n) + 0.5
    phi, psi = np.sqrt(2.0), 1.533751168755204288118041
    r, R_ = np.sqrt(s / n), np.sqrt(1.0 - s / n)
    a, b = 2 * np.pi * s / phi, 2 * np.pi * s / psi
    q = np.stack([r * np.sin(a), r * np.cos(a), R_ * np.sin(b), R_ * np.cos(b)], 1)        # (x, y, z, w)
    return Rotation.from_quat(q).as_matrix()


def box(affine, shape):
    """World centre and bounding-sphere radius (mm) of the voxel centres of a grid."""
    ijk = np.array([[i, j, k] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1) for k in (0, shape[2] - 1)], float)
    xyz = apply_affine(np.asarray(affine, float), ijk)
    return xyz.mean(0), float(np.linalg.norm(xyz - xyz.mean(0), axis=1).max())


def apply_affine(T, pts):
    """T [4, 4] applied to points [..., 3], numpy or torch as pts."""
    if isinstance(pts, torch.Tensor):
        T = torch.as_tensor(T, dtype=pts.dtype, device=pts.device)
        return pts @ T[:3, :3].T + T[:3, 3]
    T = _np(T)
    return np.asarray(pts) @ T[:3, :3].T + T[:3, 3]


def resample_iso(vol: io.Volume, level_mm, binary=False):
    """Isotropic grid of spacing level_mm aligned with the volume axes, streamed plane by plane: box average, then trilinear.
    binary: average the indicator voxel > 0 of a mask file. -> (float32 [d, h, w], affine)"""
    planes = ((k, p > 0) for k, p in io.iter_planes(vol)) if binary else io.iter_planes(vol)
    return _iso(planes, vol.shape, vol.affine, level_mm)


def pool_iso(arr, affine, level_mm):
    """resample_iso for an in-memory array [D, H, W] on the grid `affine`. -> (float32 array, affine)"""
    arr = np.asarray(arr)
    return _iso(((k, arr[:, :, k]) for k in range(arr.shape[2])), arr.shape, affine, level_mm)


def _iso(planes, shape, affine, level_mm):
    A = _np(affine)
    sp = np.linalg.norm(A[:3, :3], axis=0)
    spg = np.array([float(f"{s:.6g}") for s in sp])                            # grid arithmetic ignores float32 header noise
    k = np.maximum(1, np.rint(np.round(level_mm / spg, 9))).astype(int)       # round half to even
    n = np.array(shape[:3]) // k
    step = level_mm / (spg * k)                                                # output step in pooled voxels
    N = np.floor(np.round(n / step, 9)).astype(int)
    if N.min() < 2:
        raise ValueError(f"volume {tuple(shape)} with spacing {sp} mm has fewer than 2 voxels per axis at {level_mm} mm")
    A_out = A.copy()
    A_out[:3, 3] += A[:3, :3] @ ((k - 1) / 2.0)
    A_out[:3, :3] = A[:3, :3] / sp * level_mm
    taps = []
    for a in range(3):
        u = np.round(np.arange(N[a]) * step[a], 9)                             # no float residue weights off a centre
        i0 = np.minimum(np.floor(u).astype(int), n[a] - 1)
        taps.append((i0, u - i0))
    out, stream = np.empty(N, np.float32), _pooled(planes, k, n)
    lo = hi = None
    have = -1                                                                  # pooled planes have - 1 (lo) and have (hi) held
    for m, (q, f) in enumerate(zip(*taps[2])):
        while have < min(q + 1, n[2] - 1):
            lo, hi, have = hi, next(stream), have + 1
        plane = lo * (1 - f) + hi * f if have > q else hi * (1 - f)            # past the last pooled plane, blend with 0
        out[:, :, m] = _lin(_lin(plane, 0, *taps[0]), 1, *taps[1])
    return out, A_out


def _pooled(planes, k, n):
    """float64 [n0, n1] box means of consecutive k[2] planes of the (index, plane [D, H]) stream."""
    acc = 0.0
    for q, plane in planes:
        acc = acc + np.asarray(plane[:n[0] * k[0], :n[1] * k[1]], np.float64).reshape(n[0], k[0], n[1], k[1]).sum((1, 3))
        if q % k[2] == k[2] - 1:
            yield acc / k.prod()
            acc = 0.0


def _lin(x, axis, i0, f):
    """Linear interpolation of the 2-D x along axis at i0 + f, zero beyond the last voxel."""
    if len(i0) == x.shape[axis] and not f.any():
        return x
    x = np.concatenate([x, np.zeros_like(x.take([0], axis))], axis)
    f = f.reshape((-1, 1) if axis == 0 else (1, -1))
    return x.take(i0, axis) * (1 - f) + x.take(i0 + 1, axis) * f


def sample_world(field, affine, pts):
    """Trilinear sample of field [C, D, H, W] (voxel -> world affine) at world points pts [..., 3], 0 outside. -> [C, ...]"""
    C, D, H, W = field.shape
    Ainv = torch.as_tensor(np.linalg.inv(_np(affine)), dtype=pts.dtype, device=pts.device)
    ijk = pts @ Ainv[:3, :3].T + Ainv[:3, 3]
    size = torch.tensor([D - 1, H - 1, W - 1], dtype=pts.dtype, device=pts.device)
    grid = (ijk / size * 2 - 1)[..., [2, 1, 0]].reshape(1, 1, 1, -1, 3).to(field.dtype)
    out = F.grid_sample(field[None], grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return out.reshape(C, *pts.shape[:-1])


def resample_to(arr, affine_src, shape_dst, affine_dst, T_dst_world_to_src_world, order=1, field=None, affine_field=None):
    """Resample arr (affine_src) onto the grid (shape_dst, affine_dst) through T, order 1 (float32) or 0 (arr dtype), 0 outside.
    field [3, d, h, w]: a pull-back displacement u in mm along the destination world axes on the grid affine_field, constant
    beyond it. The destination point x is sampled at T (x + u(x))."""
    src = np.asarray(arr)
    A_d, B = _np(affine_dst), np.linalg.inv(_np(affine_src)) @ _np(T_dst_world_to_src_world)   # B: dst world -> src voxel
    M = B @ A_d                                                                                # dst voxel -> src voxel
    shape = tuple(int(s) for s in shape_dst)
    out = np.zeros(shape, np.float32 if order else src.dtype)
    jk = np.stack(np.meshgrid(np.arange(shape[1]), np.arange(shape[2]), indexing="ij"), 0).reshape(2, -1).astype(float)
    step = max(1, (CHUNK // (1 if field is None else 4)) // jk.shape[1])
    if field is not None:
        field, Fm = np.asarray(field, np.float32), np.linalg.inv(_np(affine_field)) @ A_d
    for i0 in range(0, shape[0], step):
        i = np.repeat(np.arange(i0, min(shape[0], i0 + step), dtype=float), jk.shape[1])
        c = M[:3, :1] * i + np.tile(M[:3, 1:3] @ jk + M[:3, 3:], len(i) // jk.shape[1])
        if field is not None:
            f = Fm[:3, :1] * i + np.tile(Fm[:3, 1:3] @ jk + Fm[:3, 3:], len(i) // jk.shape[1])
            c += B[:3, :3] @ np.stack([ndimage.map_coordinates(u, f, output=np.float64, order=1, mode="nearest") for u in field])
        out[i0:i0 + len(i) // jk.shape[1]] = ndimage.map_coordinates(src, c, output=out.dtype, order=order, mode="grid-constant",
                                                                     cval=0).reshape(-1, shape[1], shape[2])
    return out


def gauss_1d(sigma_vox, order, device):
    """1-D Gaussian of sigma_vox voxels, or its first derivative scaled to be exact on a ramp."""
    r = int(math.ceil(4 * sigma_vox))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    g = torch.exp(-0.5 * (x / sigma_vox) ** 2)
    g = g / g.sum()
    if order == 0:
        return g
    d = x * g                                          # -G'(x) up to a factor, since conv3d correlates
    return d / (x * d).sum()


def filter_sep(vol, sigma_vox, orders):
    """Separable Gaussian (derivative orders per axis, 0 or 1) of vol [D, H, W] (torch), zero padding."""
    out = vol[None, None]
    for ax in range(3):
        k = gauss_1d(sigma_vox, orders[ax], vol.device)
        shape = [1, 1, 1, 1, 1]
        shape[2 + ax] = len(k)
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - ax)] = pad[2 * (2 - ax) + 1] = len(k) // 2
        out = F.conv3d(F.pad(out, pad), k.reshape(shape))
    return out[0, 0]


def nc_gradient(xm, m, sigma_vox):
    """Gradient in array axes of the normalised convolution G(x m) / G(m) at sigma_vox voxels, zero outside the mask.
    xm is x already multiplied by m."""
    num, den = filter_sep(xm, sigma_vox, (0, 0, 0)), filter_sep(m, sigma_vox, (0, 0, 0)).clamp(min=1e-3)
    return torch.stack([(filter_sep(xm, sigma_vox, o) * den - num * filter_sep(m, sigma_vox, o)) / (den * den)
                        for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]) * m

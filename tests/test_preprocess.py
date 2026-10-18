"""preprocess: the MRI foreground valley, a shallow tissue valley passed over, a deep one inside the tissue rejected; the OCT
specimen mask by signal and texture (and chunk invariance); holes filled in the planes across a cut face."""
import dataclasses

import numpy as np
import pytest
from scipy import ndimage

from octreg import preprocess as pp
from octreg.params import Params

P = Params()


def _dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum())


def _ball(n, c, r):
    z, y, x = np.mgrid[:n, :n, :n]
    return (z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2 <= r * r


def test_foreground_valley():
    """A ball in a darker medium: the valley between them; one class only: every positive voxel."""
    rng = np.random.default_rng(0)
    ball = _ball(64, (32, 32, 32), 26)
    bimodal = np.where(ball, rng.normal(500, 30, ball.shape), rng.normal(100, 20, ball.shape)).astype(np.float32)
    m, info = pp.foreground(bimodal, 0.15, P)
    assert 150 < info["threshold"] < 400 and _dice(m, ball) > 0.98
    assert info["volume_cm3"] == pytest.approx(m.sum() * 0.15 ** 3 / 1e3)
    m, info = pp.foreground(rng.normal(500, 40, (48,) * 3).astype(np.float32), 0.15, P)
    assert info["threshold"] == 0 and m.all()


def test_foreground_passes_a_shallow_tissue_valley():
    """Two overlapping tissue classes (white 400, grey 500, a valley above half their peaks) in a medium (60): walking down
    from the brightest peak, the tissue valley is passed over and the medium's valley taken, so the mask keeps both classes."""
    rng = np.random.default_rng(1)
    ball, core = _ball(64, (32, 32, 32), 26), _ball(64, (32, 32, 32), 15)
    tissue = np.where(core, rng.normal(420, 40, ball.shape), rng.normal(500, 40, ball.shape))
    m, info = pp.foreground(np.where(ball, tissue, rng.normal(60, 10, ball.shape)).astype(np.float32), 0.15, P)
    assert 60 < info["threshold"] < 300 and _dice(m, ball) > 0.98, info


def test_foreground_keeps_a_dark_tissue_class():
    """All tissue: a bright class with a darker one running through it (white matter in cortex), a deep valley between them.
    The darker class lies inside the convex hull of the brighter one, so the valley is not the tissue edge."""
    rng = np.random.default_rng(2)
    z = np.indices((64,) * 3)[0]
    slab = (z > 24) & (z < 40)
    m, info = pp.foreground(np.where(slab, rng.normal(250, 25, slab.shape), rng.normal(500, 25, slab.shape)).astype(np.float32),
                            0.15, P)
    assert info["threshold"] == 0 and m.all(), info


def test_foreground_keeps_a_dark_tissue_class_in_medium():
    """The same in a medium (60): the deep valley between the two tissue classes is passed over, since the darker class lies
    inside the hull of the brighter one even though the medium outside outnumbers it, and the medium's valley is taken."""
    rng = np.random.default_rng(3)
    ball = _ball(64, (32, 32, 32), 26)
    z = np.indices((64,) * 3)[0]
    slab = ball & (z > 24) & (z < 40)
    img = np.where(slab, rng.normal(250, 25, ball.shape), np.where(ball, rng.normal(500, 25, ball.shape),
                                                                   rng.normal(60, 10, ball.shape)))
    m, info = pp.foreground(img.astype(np.float32), 0.15, P)
    assert 60 < info["threshold"] < 250 and _dice(m, ball) > 0.98, info


def _embedded_block(seed, medium=1.0, n=96):
    """A tissue slab with isotropic texture, cut by the two faces along axis 2, in a frame of medium of mean intensity
    `medium` x tissue, plus a black tile column. Medium as bright as tissue has anisotropic structure only (section offsets
    along axis 2, tile seams along axis 0, quiet along axis 1); dark medium is speckle (exponential intensities). The specimen is
    the larger part of the measured volume, as in a block imaged with a margin."""
    rng = np.random.default_rng(seed)
    z, y = np.mgrid[:n, :n]
    slab = np.broadcast_to((((z - 48) / 36) ** 4 + ((y - 46) / 38) ** 4 <= 1)[:, :, None], (n,) * 3)
    tex = ndimage.gaussian_filter(rng.standard_normal((n,) * 3), 1.5)
    stripes = rng.normal(0, 0.05, n)[None, None, :] + 0.1 * (np.arange(n) % 24 < 2)[:, None, None]
    noise = 1 + 0.1 * rng.standard_normal((n,) * 3)
    outside = noise if medium == 1 else rng.exponential(1.0, (n,) * 3)
    vol = 1000 * (1 + stripes) * np.where(slab, (1 + 0.25 * tex / tex.std()) * noise, medium * outside)
    vol = np.maximum(vol, 1).astype(np.float32)
    vol[:, :20, :20] = 0
    return vol, slab & (vol > 0)


@pytest.mark.parametrize("medium", [1.0, 0.05])            # medium as bright as tissue (texture decides), dark (signal decides)
@pytest.mark.parametrize("voxel_mm, min_dice", [(0.08, 0.95), (P.fine_mm, 0.95)])   # 0.04 mm: the default fine grid
def test_specimen_mask_signal_and_texture(monkeypatch, medium, voxel_mm, min_dice):
    vol, truth = _embedded_block(0, medium)
    Q = dataclasses.replace(P, texture_smooth_mm=0.32)  # a block of a few mm (the default 1.2 mm suits real blocks)
    m, info = pp.specimen_mask(vol, voxel_mm, Q)
    assert _dice(m, truth) > min_dice and not m[vol == 0].any(), _dice(m, truth)
    assert info["volume_cm3"] == pytest.approx(m.sum() * voxel_mm ** 3 / 1e3)
    monkeypatch.setattr(pp, "_CHUNK", 16)
    assert np.array_equal(pp.specimen_mask(vol, voxel_mm, Q)[0], m)      # chunked == unchunked


def test_fill_planes_fills_holes_open_to_a_face():
    """A cavity that reaches a cut face is not enclosed in 3-D but is in the planes across it; outside stays outside."""
    m = np.zeros((12, 12, 12), bool)
    m[2:10, 2:10, :] = True
    m[5:7, 5:7, :] = False                                   # a tunnel through the block along axis 2, open at both faces
    m[2:6, 2:6, 4:8] &= ~_ball(4, (1.5, 1.5, 1.5), 1.2)      # and an enclosed cavity, m[3:5, 3:5, 5:7], off the tunnel
    filled = ndimage.binary_fill_holes(m)
    assert not filled[5, 5, 0] and filled[3, 3, 5] and not m[3, 3, 5]
    f = pp._fill_planes(m)
    assert f[2:10, 2:10, :].all() and not f[:2].any() and not f[10:].any()

"""preprocess: the MRI foreground valley, also on an integer-valued image, with a hot voxel, next to a medium peak in the
first histogram bin and for tissue a small share of the crop, holes filled and specks dropped, a shallow tissue valley passed
over, a deep one inside the tissue rejected. The OCT specimen mask by signal and texture (and chunk invariance), an OCT without
positive voxels refused. Holes filled in the planes across a cut face."""
import dataclasses

import numpy as np
import pytest
from scipy import ndimage
from scipy.signal import find_peaks

from octreg import preprocess as pp
from octreg.params import Params

P = Params()


def _dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum())


def _ball(n, c, r):
    z, y, x = np.mgrid[:n, :n, :n]
    return (z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2 <= r * r


def test_foreground_valley():
    """A ball in a darker medium: the valley between them. One class only: every positive voxel."""
    rng = np.random.default_rng(0)
    ball = _ball(64, (32, 32, 32), 26)
    bimodal = np.where(ball, rng.normal(500, 30, ball.shape), rng.normal(100, 20, ball.shape)).astype(np.float32)
    m, info = pp.foreground(bimodal, 0.15, P)
    assert 150 < info["threshold"] < 400 and _dice(m, ball) > 0.98
    assert info["volume_cm3"] == pytest.approx(m.sum() * 0.15 ** 3 / 1e3)
    m, info = pp.foreground(rng.normal(500, 40, (48,) * 3).astype(np.float32), 0.15, P)
    assert info["threshold"] == 0 and m.all()


def test_foreground_on_an_integer_image_and_with_a_hot_voxel():
    """An integer-valued crop at the base grid (grey matter at 120 levels, white matter, medium, 10 % noise), whose raw
    histogram is a comb of empty bins, and a crop with one voxel 200 times brighter than the tissue: the medium's valley in
    both."""
    for seed in range(10):
        rng = np.random.default_rng(seed)
        ball, core = _ball(80, (40, 40, 40), 34), _ball(80, (40, 40, 40), 20)
        mean = np.where(core, 75.0, np.where(ball, 120.0, 30.0))
        img = np.maximum(np.round(mean * (1 + 0.1 * rng.standard_normal(ball.shape))), 0).astype(np.float32)
        m, info = pp.foreground(img, 0.15, P)
        assert 30 < info["threshold"] < 75 and _dice(m, ball) > 0.98, (seed, info)
    rng = np.random.default_rng(0)
    ball = _ball(64, (32, 32, 32), 26)
    img = np.where(ball, rng.normal(500, 30, ball.shape), rng.normal(100, 20, ball.shape)).astype(np.float32)
    img[2, 2, 2] = 1e5
    m, info = pp.foreground(img, 0.15, P)
    assert 150 < info["threshold"] < 400 and _dice(m, ball) > 0.98, info


def test_foreground_next_to_an_end_bin_peak_and_for_a_small_tissue_share():
    """A medium of low noise (Rayleigh, sigma 3) whose peak falls in the first histogram bin, and a ball holding 2.4 % of the
    crop, whose peak the 99.5th percentile cuts: the medium's valley in both, since the peaks are found with an empty bin
    added at each end of the histogram."""
    rng = np.random.default_rng(0)
    ball = _ball(64, (32, 32, 32), 26)
    m, info = pp.foreground(np.where(ball, rng.normal(500, 30, ball.shape), rng.rayleigh(3, ball.shape)).astype(np.float32),
                            0.15, P)
    assert 10 < info["threshold"] < 400 and _dice(m, ball) > 0.98, info
    ball = _ball(100, (50, 50, 50), 18)
    m, info = pp.foreground(np.where(ball, rng.normal(500, 30, ball.shape), rng.normal(100, 20, ball.shape)).astype(np.float32),
                            0.15, P)
    assert 150 < info["threshold"] < 400 and _dice(m, ball) > 0.98, info


def test_foreground_fills_holes_and_drops_specks():
    """A cavity of medium inside the tissue is foreground, and single bright voxels in the medium are not."""
    rng = np.random.default_rng(5)
    ball, cavity = _ball(64, (32, 32, 32), 26), _ball(64, (32, 32, 32), 6)
    img = np.where(ball & ~cavity, rng.normal(500, 30, ball.shape), rng.normal(100, 20, ball.shape)).astype(np.float32)
    specks = [(2, 2, 2), (2, 61, 30), (61, 3, 60), (30, 2, 61)]
    for s in specks:
        img[s] = 600
    m, info = pp.foreground(img, 0.15, P)
    assert m[cavity].all() and not any(m[s] for s in specks) and _dice(m, ball) > 0.98, info


def test_foreground_passes_a_shallow_tissue_valley():
    """A bright core (500) in a darker tissue shell (387) in a medium (60), the valley between core and shell above half their
    peaks: walking down from the brightest peak, that valley is passed over and the medium's valley taken, so the mask keeps
    the shell. With every valley deep enough (valley_ratio 1) the shell, which lies outside the core's hull as a medium would,
    is cut off, so the depth of the valley is what keeps it."""
    rng = np.random.default_rng(0)
    ball, core = _ball(64, (32, 32, 32), 26), _ball(64, (32, 32, 32), 20)
    img = np.where(core, rng.normal(500, 35, ball.shape), np.where(ball, rng.normal(387, 35, ball.shape),
                                                                   rng.normal(60, 10, ball.shape))).astype(np.float32)
    s = img[img <= np.percentile(img, 99.5)]
    h = ndimage.uniform_filter1d(np.histogram(s, bins=256)[0].astype(float), 5)
    pk = find_peaks(np.sqrt(np.pad(h, 1)), prominence=0.05 * np.sqrt(h.max()), distance=5)[0] - 1
    assert len(pk) == 3 and 0.5 < h[pk[1]:pk[2]].min() / min(h[pk[1]], h[pk[2]]) < 1, pk     # the premise: a shallow valley
    m, info = pp.foreground(img, 0.15, P)
    assert 60 < info["threshold"] < 300 and _dice(m, ball) > 0.98, info
    m, info = pp.foreground(img, 0.15, dataclasses.replace(P, valley_ratio=1.0))
    assert info["threshold"] > 387 and _dice(m, ball) < 0.7, info


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
    along axis 2, tile seams along axis 0, quiet along axis 1). Dark medium is speckle (exponential intensities). The
    specimen is the larger part of the measured volume, as in a block imaged with a margin."""
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


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("medium, voxel_mm", [(1.0, 0.08), (1.0, P.fine_mm), (0.05, 0.08)])
def test_specimen_mask_signal_and_texture(monkeypatch, seed, medium, voxel_mm):
    """Medium as bright as tissue (texture decides) on a 0.08 mm grid and on the default 0.04 mm fine grid, and dark medium
    (signal decides). At 0.04 mm the 4 mm block next to dark medium is too small for the texture statistics, so dark medium is
    tested at 0.08 mm."""
    vol, truth = _embedded_block(seed, medium)
    Q = dataclasses.replace(P, texture_smooth_mm=0.32)  # a block of a few mm (the default 1.2 mm suits real blocks)
    m, info = pp.specimen_mask(vol, voxel_mm, Q)
    assert _dice(m, truth) > 0.95 and not m[vol == 0].any(), _dice(m, truth)
    assert info["volume_cm3"] == pytest.approx(m.sum() * voxel_mm ** 3 / 1e3)
    if seed == 0:
        monkeypatch.setattr(pp, "_CHUNK", 16)
        assert np.array_equal(pp.specimen_mask(vol, voxel_mm, Q)[0], m)      # chunked == unchunked


def test_specimen_mask_refuses_an_oct_without_positive_voxels():
    """An OCT of zeros or of negative values (a dB image) has no measured voxel: refused with a message."""
    for vol in (np.zeros((40,) * 3, np.float32), -np.ones((40,) * 3, np.float32)):
        with pytest.raises(ValueError, match="no positive voxels"):
            pp.specimen_mask(vol, 0.08, P)


def test_fill_planes_fills_holes_open_to_a_face():
    """A cavity that reaches a cut face is not enclosed in 3-D but is in the planes across it. Outside stays outside."""
    m = np.zeros((12, 12, 12), bool)
    m[2:10, 2:10, :] = True
    m[5:7, 5:7, :] = False                                   # a tunnel through the block along axis 2, open at both faces
    m[2:6, 2:6, 4:8] &= ~_ball(4, (1.5, 1.5, 1.5), 1.2)      # and an enclosed cavity, m[3:5, 3:5, 5:7], off the tunnel
    filled = ndimage.binary_fill_holes(m)
    assert not filled[5, 5, 0] and filled[3, 3, 5] and not m[3, 3, 5]
    f = pp._fill_planes(m)
    assert f[2:10, 2:10, :].all() and not f[:2].any() and not f[10:].any()

"""tilestack.mri: the tissue mask and the levelled copy that octreg reads, and octreg's parameters for an MRI."""
import nibabel as nib
import numpy as np
import pytest

from octreg.params import Params
from tilestack.mri import depth_under_cut, levelled, octreg_params, tissue

SPACING = (0.08, 0.08, 0.08)


def _block(seed=0):
    """A tissue ball with a cavity, in gel that fills most of the grid (mu_s per um, speckle), cut into slabs whose cuts
    tilt across x, every second one also across y, so a slab is 9 to 15 planes deep depending on the column. mu_s falls
    with the depth under the cut and every second slab is 10 % brighter. From 7 to 9 planes under the cut, depending on
    the column as the window's end does, the planes are below the reliable ones, where gel and tissue read alike.
    -> volume, ball, source map, slab and depth under its cut of every voxel, unreliable voxels, cavity"""
    rng = np.random.default_rng(seed)
    x, y, z = np.mgrid[:64, :64, :80]
    r2 = (x - 32) ** 2 + (y - 32) ** 2 + (z - 42) ** 2
    ball, cavity = r2 <= 22 ** 2, r2 <= 7 ** 2
    s = np.arange(9)[:, None, None]
    cuts = 12 * s - np.round(0.1 * x[None, :, :, 0]) + (s % 2) * np.round(0.05 * y[None, :, :, 0])
    slab = (cuts[:, :, :, None] <= z[None]).sum(0) - 1
    depth = (z - cuts[slab, x, y]).astype(int)
    deep = depth >= 9 - x % 3
    mu = np.where(deep, 3.5e-3, np.where(ball & ~cavity, 4e-3, 1.4e-3)) * rng.lognormal(0, 0.25, ball.shape)
    mu = mu * 10 ** (-0.15 * depth / 11) * (1 + 0.1 * (slab % 2))
    return mu.astype(np.float32), ball, (slab + 1000 * deep).astype(np.int16), slab, depth, deep, cavity


def test_tissue_mask_is_the_tissue_although_the_gel_is_most_of_the_volume():
    vol, ball, src, _, _, deep, cavity = _block()
    assert ball.mean() < 0.3
    m = tissue(vol, src, SPACING)
    assert 2 * (m & ball).sum() / (m.sum() + ball.sum()) > 0.95 and m[cavity].all()
    assert 2 * (m & ball)[deep].sum() / (m[deep].sum() + ball[deep].sum()) > 0.9
    assert (m & ~ball)[deep].sum() < 0.1 * m[deep].sum()


def test_depth_is_counted_from_the_first_plane_of_the_slab_in_the_column():
    src = np.array([[-1, 5, 5, 1005, 1005, -1, -1, 7, 7, 1007, 8, -1],        # slab 6 missing, no data between slabs
                    [5, 6, 6, 6, 1006, 1006, 1006, 8, 8, 8, 9, 1009]], np.int16).reshape(2, 1, 12)
    depth = np.array([[0, 0, 1, 2, 3, 0, 0, 0, 1, 2, 0, 0], [0, 0, 1, 2, 3, 4, 5, 0, 1, 2, 0, 1]]).reshape(2, 1, 12)
    assert np.array_equal(depth_under_cut(src)[src >= 0], depth[src >= 0])


def test_levelled_divides_by_the_tissue_median_of_each_slab_at_each_depth_under_its_cut():
    vol, ball, src, slab, depth, _, _ = _block(1)
    vol[:, :5], src[:, :5] = 0, -1
    m = tissue(vol, src, SPACING)
    out = levelled(vol, m, src)
    assert len(np.unique(depth.max(axis=2))) > 3                 # the slabs' depths vary with the column
    groups = slab * 100 + depth
    for g in np.unique(groups[m]):                               # one divisor for the group, reliable planes or not
        sel = (groups == g) & (vol > 0)
        assert np.allclose(vol[sel] / out[sel], np.exp(np.median(np.log(vol[sel & m]))), rtol=1e-5)
    assert ((out > 0) == (vol > 0)).all()
    cut = np.zeros(m.shape, bool)                                # first plane of a newer slab, tissue on both sides
    cut[:, :, 1:] = (slab[:, :, 1:] > slab[:, :, :-1]) & m[:, :, 1:] & m[:, :, :-1]
    step = lambda v: np.median(np.log10(v[cut] / v[np.roll(cut, -1, axis=2)]))
    assert abs(step(out)) < 0.05 < abs(step(vol))


@pytest.mark.parametrize("voxel, base, extra", [(0.03, 0.04, (0.2, 0.133333, 0.0888889, 0.08)), (0.08, 0.08, (0.2, 0.16)),
                                                (0.1, 0.1, (0.2,)), (0.12, 0.12, (0.24,)), (0.15, 0.15, ()), (0.3, 0.15, ())])
def test_refinement_continues_to_two_base_voxels(tmp_path, voxel, base, extra):
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), np.diag([0.9 * voxel, voxel, 0.5 * voxel, 1.0])), tmp_path / "mri.nii")
    P = octreg_params(tmp_path / "mri.nii")
    assert P.base_mm == pytest.approx(base) and P.sigmas_mm == pytest.approx(Params().sigmas_mm + extra, rel=1e-4)
    if not extra:
        assert P == Params()

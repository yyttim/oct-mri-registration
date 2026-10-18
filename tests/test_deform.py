"""§5 smooth deformation (octreg.deform) on a synthetic ellipsoid, CPU only."""
import dataclasses

import numpy as np
import pytest
import torch
from scipy import ndimage
from scipy.spatial.transform import Rotation

from octreg import deform, geometry as G, io
from octreg.params import Params

FAST = dataclasses.replace(Params(), df_evals=75)
AMP, WIDTH = 0.8, 5.0                                              # bump amplitude and Gaussian width, mm
DIRECTION, CENTRE = np.array([0.6, -0.64, 0.48]), np.array([1.5, 1.0, -1.0])
BUMP_STRAIN = AMP * np.abs(DIRECTION).max() / (WIDTH * np.sqrt(np.e))     # largest |dd_i / dz_k| of the bump, 0.062


def structure(shape, rng):
    """Band-limited noise with oriented ridges, unit variance."""
    s = ndimage.gaussian_filter(rng.standard_normal(shape), 2.0) * 8 + ndimage.gaussian_filter(rng.standard_normal(shape), (1.2, 4.0, 1.2)) * 5
    return (s / s.std()).astype(np.float32)


def grid(shape, h, flipped):
    A = np.eye(4)
    A[:3, :3] = np.diag([-h, -h, h]) if flipped else np.array([[h, 0, 0], [0, 0, h], [0, -h, 0]])
    A[:3, 3] = -A[:3, :3] @ ((np.array(shape) - 1) / 2)
    return A


def world(shape, A):
    return G.apply_affine(A, np.indices(shape).reshape(3, -1).T.astype(float))


def bump(z, amplitude=AMP):
    """The bump displacement d at MRI world points z [N, 3], in mm."""
    return amplitude * DIRECTION * np.exp(-((z - CENTRE) ** 2).sum(-1, keepdims=True) / (2 * WIDTH ** 2))


def pair(d, h=0.3, semi=(6.0, 5.4, 5.0), seed=0):
    """A synthetic MRI and OCT related by T and the displacement d, on grids of spacing h mm, the specimen an ellipsoid of
    semi-axes semi mm in the MRI world. -> ((mri, mask, A), (oct, mask, A), T). The OCT voxel x shows the MRI at
    y = T x + d(T x), contrast inverted."""
    rng, semi = np.random.default_rng(seed), np.asarray(semi, float)
    shape_m = tuple(int(np.ceil((2 * a + 3.0) / h)) for a in semi[[0, 2, 1]])  # MRI array axes are R, I, A
    shape_o = tuple(int(np.ceil((2 * a + 4.0) / h)) for a in semi)
    A_M, A_O = grid(shape_m, h, False), grid(shape_o, h, True)
    s = structure(shape_m, rng)
    mask_m = (((world(shape_m, A_M) / semi) ** 2).sum(-1) <= 1).reshape(shape_m)
    mri = np.where(mask_m, 1.0 + 0.3 * s, 0.0).astype(np.float32)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec([0.05, -0.08, 0.06]).as_matrix()
    T[:3, 3] = [0.3, -0.2, 0.25]
    z = G.apply_affine(T, world(shape_o, A_O))
    y = z + d(z)
    so = G.sample_world(torch.as_tensor(s[None]), A_M, torch.as_tensor(y, dtype=torch.float32))[0].numpy()
    tissue = ((y / semi) ** 2).sum(-1) <= 1
    oct_ = np.where(tissue, 1.6 - 0.3 * so, 0.3 + 0.05 * rng.standard_normal(len(y))).reshape(shape_o).astype(np.float32)
    return (mri, mask_m, A_M), (oct_, tissue.reshape(shape_o), A_O), T


def fit(mri, oct_, T, params=FAST):
    """Run smooth_deformation on the CPU. -> (u, info, the LatticeField it fitted)."""
    made = []

    class Recorded(deform.LatticeField):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            made.append(self)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(deform, "LatticeField", Recorded)
        u, info = deform.smooth_deformation(mri, oct_, T, params, "cpu")
    assert len(made) == 1                                                                   # one lattice for all passes
    return u, info, made[0]


def pull_back(d, x):
    """Solve x + u + d(x + u) = x for u by fixed-point iteration."""
    u = np.zeros_like(x)
    for _ in range(30):
        u = -d(x + u)
    return u


def lattice_strain(field):
    return max(float(s.abs().max()) for s in field.strains())


@pytest.fixture(scope="module")
def bumped():
    mri, oct_, T = pair(bump)
    return (mri, oct_, T) + fit(mri, oct_, T)


@pytest.fixture(scope="module")
def exact_copy():
    """The MRI as its own OCT with T = I, so F = 1 at the zero field."""
    mri, _, _ = pair(lambda z: 0 * z)
    oct_ = mri
    return (mri, oct_) + fit(mri, oct_, np.eye(4))


def test_lattice_field_interpolates_and_strains():
    """LatticeField is trilinear on its nodes, exact on linear fields and takes its border value beyond the lattice.
    Its strains are first differences over the spacing."""
    lo, hi, h = np.array([-3.0, 0.0, 1.0]), np.array([7.5, 9.9, 11.0]), 5.0
    f = deform.LatticeField(lo, hi, h, "cpu")
    assert f.n == [4, 3, 3] and tuple(f.c.shape) == (3, 36) and not f.c.detach().any() and f.c.requires_grad
    assert np.all(lo + h * (np.array(f.n) - 1) >= hi)
    rng = np.random.default_rng(0)
    c = rng.standard_normal((3, 4, 3, 3))
    with torch.no_grad():
        f.c.copy_(torch.as_tensor(c.reshape(3, -1)))
        nodes = lo + h * np.indices(f.n).reshape(3, -1).T
        assert np.abs(f(torch.as_tensor(nodes, dtype=torch.float32)).numpy() - c.reshape(3, -1).T).max() < 1e-5
        y = rng.uniform(lo, lo + h * (np.array(f.n) - 1), (200, 3))
        want = np.stack([ndimage.map_coordinates(ci, ((y - lo) / h).T, order=1) for ci in c], 1)
        assert np.abs(f(torch.as_tensor(y, dtype=torch.float32)).numpy() - want).max() < 1e-5
        out = torch.as_tensor(np.stack([lo - [2.0, 0, 0], lo + h * (np.array(f.n) - 1) + [0, 1.5, 3.0]]), dtype=torch.float32)
        assert np.abs(f(out).numpy() - c[:, [0, -1], [0, -1], [0, -1]].T).max() < 1e-5  # the border value outside the lattice
        strains = f.strains()
        assert [tuple(s.shape) for s in strains] == [(3, 3, 3, 3), (3, 4, 2, 3), (3, 4, 3, 2)]
        assert all(np.allclose(s.numpy(), np.diff(c, axis=a + 1) / h, atol=1e-6) for a, s in enumerate(strains))
        J, b = np.array([[0.01, 0.02, -0.01], [0.0, -0.03, 0.02], [0.04, 0.0, 0.0]]), np.array([0.3, -0.1, 0.0])
        f.c.copy_(torch.as_tensor((nodes @ J.T + b).T))
        assert np.abs(f(torch.as_tensor(y, dtype=torch.float32)).numpy() - (y @ J.T + b)).max() < 1e-5
        assert all(np.allclose(s.numpy(), J[:, a, None, None, None], atol=1e-6) for a, s in enumerate(f.strains()))
        assert abs(lattice_strain(f) - 0.04) < 1e-6


def test_lattice_gradient_is_the_sum_of_the_weights():
    """The backward pass of LatticeField matches autograd through a plain trilinear gather."""
    f = deform.LatticeField(np.zeros(3), np.array([10.0, 12.0, 8.0]), 2.0, "cpu")
    rng = np.random.default_rng(1)
    y = torch.as_tensor(rng.uniform(-1.0, 13.0, (500, 3)), dtype=torch.float32)
    r = torch.as_tensor(rng.standard_normal((500, 3)), dtype=torch.float32)
    with torch.no_grad():
        f.c.copy_(torch.as_tensor(rng.standard_normal(tuple(f.c.shape)), dtype=torch.float32))
    (f(y) * r).sum().backward()
    c = f.c.detach().clone().requires_grad_(True)
    n = torch.tensor(f.n)
    q = torch.minimum((y / f.h).clamp(min=0), n - 1)
    i0 = torch.minimum(q.floor().long(), n - 2)
    fr = q - i0
    d = 0
    for a in (0, 1):
        for b in (0, 1):
            for e in (0, 1):
                idx = ((i0[:, 0] + a) * n[1] + i0[:, 1] + b) * n[2] + i0[:, 2] + e
                w = (fr[:, 0] if a else 1 - fr[:, 0]) * (fr[:, 1] if b else 1 - fr[:, 1]) * (fr[:, 2] if e else 1 - fr[:, 2])
                d = d + c[:, idx].T * w[:, None]
    (d * r).sum().backward()
    assert torch.allclose(f.c.grad, c.grad, atol=1e-5)


def test_identical_pair_leaves_the_field_near_zero(exact_copy):
    mri, oct_, u, info, field = exact_copy
    assert set(info) == {"passes"} and u.shape == (3, *mri[0].shape) and u.dtype == np.float32
    assert all(p["F_start"] > 0.9999 and p["F_end"] > 0.9999 for p in info["passes"]), info
    assert np.linalg.norm(u, axis=0)[mri[1]].max() < 0.03, np.linalg.norm(u, axis=0)[mri[1]].max()   # a tenth of a voxel
    mri, oct_, T = pair(lambda z: 0 * z)                       # the aligned pair, d = 0
    u, info, _ = fit(mri, oct_, T)
    n = np.linalg.norm(u, axis=0)[mri[1]]
    assert np.median(n) < 0.1 and np.percentile(n, 95) < 0.2, (np.median(n), np.percentile(n, 95))


def test_exact_copy_leaves_the_whole_field_at_zero(exact_copy):
    """Lattice nodes that no OCT point pulls stay at zero: L-BFGS does not rescale a round-off gradient into a step."""
    mri, oct_, u, info, field = exact_copy
    assert np.abs(u).max() < 0.05 and lattice_strain(field) < 0.01, (np.abs(u).max(), lattice_strain(field))


def test_reduces_a_smooth_displacement(bumped):
    mri, oct_, T, u, info, field = bumped
    assert [p["sigma_mm"] for p in info["passes"]] == list(FAST.df_sigmas_mm)
    assert info["passes"][0]["F_end"] > info["passes"][0]["F_start"] + 0.02, info
    assert all(p["F_end"] >= p["F_start"] - 1e-3 for p in info["passes"]), info
    x = world(mri[0].shape, mri[2])[mri[1].ravel()]
    truth = pull_back(bump, x)
    imposed, err = np.linalg.norm(truth, axis=1), np.linalg.norm(u[:, mri[1]].T - truth, axis=1)
    assert np.median(imposed) > 0.5 and np.median(err) < 0.25 * np.median(imposed), (np.median(imposed), np.median(err))
    assert np.percentile(err, 95) < 0.25 and err.max() < 0.4, (np.percentile(err, 95), err.max())
    assert abs(info["passes"][-1]["displacement_max_mm"] - imposed.max()) < 0.15, (info["passes"][-1], imposed.max())
    assert lattice_strain(field) <= FAST.df_max_strain + 2e-3, lattice_strain(field)
    corners = G.apply_affine(mri[2], np.array([[0, 0, 0], np.array(mri[0].shape) - 1], float))
    assert np.allclose(field.lo.numpy(), corners.min(0) - 2 * FAST.df_grid_mm) and field.h == FAST.df_grid_mm
    # the registered OCT at x is OCT(T^-1 (x + u(x)))
    dice = lambda a: 2 * (a & mri[1]).sum() / (a.sum() + mri[1].sum())
    spec = oct_[1].astype(np.float32)
    affine = G.resample_to(spec, oct_[2], mri[0].shape, mri[2], np.linalg.inv(T)) > 0.5
    warped = G.resample_to(spec, oct_[2], mri[0].shape, mri[2], np.linalg.inv(T), field=u, affine_field=mri[2]) > 0.5
    assert dice(warped) > 0.975 and 1 - dice(warped) < 0.5 * (1 - dice(affine)), (dice(affine), dice(warped))


def test_keeps_the_strain_bound(bumped):
    """With a bound below the bump's strain the field reaches the bound, not beyond, and still moves toward the bump."""
    mri, oct_, T, u_free, *_ = bumped
    P = dataclasses.replace(FAST, df_max_strain=0.04)
    assert BUMP_STRAIN > 1.5 * P.df_max_strain
    u, info, field = fit(mri, oct_, T, P)
    assert 0.9 * P.df_max_strain < lattice_strain(field) <= P.df_max_strain + 2e-3, lattice_strain(field)
    x = world(mri[0].shape, mri[2])[mri[1].ravel()]
    truth = pull_back(bump, x)
    imposed, err = np.linalg.norm(truth, axis=1), np.linalg.norm(u[:, mri[1]].T - truth, axis=1)
    err_free = np.linalg.norm(u_free[:, mri[1]].T - truth, axis=1)
    assert np.median(err_free) < np.median(err) < 0.8 * np.median(imposed), (np.median(err_free), np.median(err), np.median(imposed))


def test_pull_back_inverts_the_fitted_field(bumped):
    mri, oct_, T, u, info, field = bumped
    x = world(mri[0].shape, mri[2])
    ux = u.reshape(3, -1).T
    with torch.no_grad():
        z = torch.as_tensor(x + ux, dtype=torch.float32)
        residual = np.linalg.norm((z + field(z)).numpy() - x, axis=1)
        first_order = np.linalg.norm(ux + field(torch.as_tensor(x, dtype=torch.float32)).numpy(), axis=1)   # u = -d(x)
    assert residual.max() < 1e-4 and first_order.max() > 0.01, (residual.max(), first_order.max())
    with pytest.raises(ValueError):
        dataclasses.replace(FAST, df_evals=0)                                         # §5 is turned off by --no-deform


def test_field_file_and_resample(tmp_path):
    """Field files round trip, and resample_to with a field gives src(T^-1 (x + u(x))) with u constant beyond its grid."""
    rng = np.random.default_rng(0)
    shape_f, shape_s = (24, 22, 20), (40, 36, 44)
    A_F = np.array([[0.5, 0, 0, -6.0], [0, 0, 0.5, -5.0], [0, -0.5, 0, 5.5], [0, 0, 0, 1.0]])
    A_S = np.diag([-0.3, -0.3, 0.3, 1.0])
    A_S[:3, 3] = -A_S[:3, :3] @ ((np.array(shape_s) - 1) / 2)
    T = np.eye(4)                                                                           # source world -> field world
    T[:3, :3] = Rotation.from_rotvec([0.1, -0.2, 0.15]).as_matrix() @ np.diag([1.03, 0.98, 1.0])
    T[:3, 3] = [0.4, -0.3, 0.2]
    field = np.stack([ndimage.gaussian_filter(rng.standard_normal(shape_f), 3.0) for _ in range(3)]).astype(np.float32)
    field *= 0.8 / np.abs(field).max()
    src = ndimage.gaussian_filter(rng.standard_normal(shape_s), 1.5).astype(np.float32)
    io.save_field(field, A_F, tmp_path / "warp.nii.gz")
    import nibabel as nib
    img = nib.load(str(tmp_path / "warp.nii.gz"))
    assert img.shape == (*shape_f, 1, 3) and img.get_data_dtype() == np.float32 and img.header.get_intent()[0] == "vector"
    back, A_back = io.load_field(tmp_path / "warp.nii.gz")
    assert np.array_equal(back, field) and np.allclose(A_back, A_F)
    io.save_nifti(src, A_S, tmp_path / "src.nii.gz")
    with pytest.raises(ValueError, match="displacement field"):
        io.load_field(tmp_path / "src.nii.gz")

    def want(shape_d, A_D):
        x = world(shape_d, A_D)
        ijk = np.clip(G.apply_affine(np.linalg.inv(A_F), x), 0, np.array(shape_f) - 1)
        ux = G.sample_world(torch.as_tensor(field, dtype=torch.float64), A_F, torch.as_tensor(G.apply_affine(A_F, ijk))).T
        y = torch.as_tensor(G.apply_affine(np.linalg.inv(T), x + ux.numpy()))
        return G.sample_world(torch.as_tensor(src[None], dtype=torch.float64), A_S, y)[0].numpy().reshape(shape_d)

    # a finer destination grid inside the field grid
    shape_d = (30, 28, 26)
    A_D = A_F.copy()
    A_D[:3, :3] *= 0.5
    A_D[:3, 3] = G.apply_affine(A_F, np.array([4.0, 4.0, 3.0]))
    got = G.resample_to(src, A_S, shape_d, A_D, np.linalg.inv(T), field=back, affine_field=A_back)
    assert got.dtype == np.float32 and np.abs(got - want(shape_d, A_D)).max() < 1e-5 and np.abs(got).max() > 0.05
    plain = G.resample_to(src, A_S, shape_d, A_D, np.linalg.inv(T))
    assert np.abs(got - plain).max() > 0.01
    assert np.array_equal(G.resample_to(src, A_S, shape_d, A_D, np.linalg.inv(T), field=0 * back, affine_field=A_back), plain)
    # a destination grid 3 voxels wider than the field grid on every side
    shape_w = tuple(n + 6 for n in shape_f)
    A_W = A_F.copy()
    A_W[:3, 3] = G.apply_affine(A_F, np.array([-3.0, -3.0, -3.0]))
    wide = G.resample_to(src, A_S, shape_w, A_W, np.linalg.inv(T), field=back, affine_field=A_back)
    assert np.abs(wide - want(shape_w, A_W)).max() < 1e-5


def test_lattice_covers_an_oblique_mri_grid():
    """The lattice spans all eight corners of an oblique MRI grid with a margin of two spacings."""
    mri, oct_, T = pair(lambda z: 0 * z)
    R = Rotation.from_rotvec([0.3, -0.2, 0.25]).as_matrix()
    A = mri[2].copy()
    A[:3, :3] = R @ A[:3, :3]
    _, _, field = fit((mri[0], mri[1], A), oct_, T, dataclasses.replace(FAST, df_evals=1))
    n = np.array(mri[0].shape) - 1
    corners = G.apply_affine(A, np.array([[i, j, k] for i in (0, n[0]) for j in (0, n[1]) for k in (0, n[2])], float))
    lo = field.lo.numpy()
    hi = lo + field.h * (np.array(field.n) - 1)
    assert (corners >= lo + 2 * FAST.df_grid_mm - 1e-4).all() and (corners <= hi - 2 * FAST.df_grid_mm + 1e-4).all()

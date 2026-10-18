"""The §2 structure similarity (octreg.features) and the §4 refinement on synthetic volumes."""
import dataclasses

import numpy as np
import torch
from scipy import ndimage
from scipy.spatial.transform import Rotation

from octreg import geometry as G
from octreg.features import MRITensors, OCTTemplate, correlation, eroded, gradient_field, score, orientation_tensor, tensors
from octreg.params import Params
from octreg.refine import compose, decompose, refine

H = 0.15


def refine_one(T0, mri, oct_, params, device):
    """Refine from one start. -> (T, its record)."""
    r = refine([T0], mri, oct_, params, device)[0]
    return r["T"], r


def grid(shape, perm=True):
    """Base-grid affine centred on the world origin, with permuted axes and a flip if perm, else two flipped axes."""
    A = np.zeros((4, 4))
    A[3, 3] = 1
    if perm:
        A[0, 0], A[1, 2], A[2, 1] = H, H, -H
    else:
        A[:3, :3] = np.diag([-H, -H, H])
    A[:3, 3] = -A[:3, :3] @ ((np.array(shape) - 1) / 2)
    return A


def world(shape, A):
    return G.apply_affine(A, np.indices(shape).reshape(3, -1).T.astype(float)).reshape(*shape, 3)


def test_gradient_field_ramp():
    shape = (40, 36, 32)
    A = grid(shape)
    x = world(shape, A)
    arr = (2 * x[..., 0] - 3 * x[..., 2]).astype(np.float32)
    mask = np.zeros(shape, bool)
    mask[4:-4, 4:-4, 4:-4] = True
    g = gradient_field(arr, mask, A, 0.3, "cpu").numpy()
    inner = ndimage.binary_erosion(mask, iterations=9)                                # beyond the kernel radius of 8 voxels
    assert np.abs(g[:, inner] - np.array([2, 0, -3])[:, None]).max() < 2e-3
    assert np.abs(g[:, ~mask]).max() == 0
    flat = gradient_field(np.where(mask, 5.0, 0.0).astype(np.float32), mask, A, 0.3, "cpu").numpy()
    assert np.abs(flat).max() < 1e-3                                                  # the mask edge adds no gradient


def test_orientation_tensor_ignores_the_sign():
    """P(n) = P(-n) and <P(n), P(m)> = (n . m)^2 over the six channels."""
    n, m = torch.as_tensor(np.random.default_rng(0).normal(size=(2, 3, 200)), dtype=torch.float32)
    P, Q = orientation_tensor(n), orientation_tensor(m)
    assert P.shape == (6, 200) and torch.equal(P, orientation_tensor(-n))
    assert torch.allclose((P * Q).sum(0), (n * m).sum(0) ** 2, atol=1e-4)


def pair(seed=0):
    """A synthetic MRI and an OCT showing it at T x with inverted non-linear contrast on a differently flipped grid.
    -> ((mri, mask, A_M), (oct, mask, A_O), T)."""
    rng = np.random.default_rng(seed)
    shape_m = (64, 64, 64)
    A_M = grid(shape_m)
    xm = world(shape_m, A_M)
    s = ndimage.gaussian_filter(rng.standard_normal(shape_m), 2.0)
    s /= s.std()
    mask_m = ((xm / np.array([4.2, 4.0, 3.8])) ** 2).sum(-1) <= 1
    mri = np.where(mask_m, 1.0 + 0.3 * s, 0.0).astype(np.float32)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec([0.2, -0.4, 0.3]).as_matrix()
    T[:3, 3] = [0.3, -0.2, 0.25]
    shape_o = (60, 62, 58)
    A_O = grid(shape_o, perm=False)
    xm_of_o = torch.as_tensor(G.apply_affine(T, world(shape_o, A_O).reshape(-1, 3)), dtype=torch.float32)
    samp = lambda vol: G.sample_world(torch.as_tensor(vol[None], dtype=torch.float32), A_M, xm_of_o)[0].numpy().reshape(shape_o)
    so, mo = samp(s.astype(np.float32)), samp(mask_m.astype(np.float32)) > 0.5
    oct_ = np.where(mo, 2.0 - np.tanh(1.5 * so), 0.0).astype(np.float32)
    return (mri, mask_m, A_M), (oct_, mo, A_O), T


def test_score_identical_and_contrast_free():
    """The same volume as OCT and MRI scores 1 at the identity, also with the OCT contrast inverted and rescaled."""
    (mri, mask, A), _, _ = pair()
    inner = eroded(mask, 0.8, H)
    mt = MRITensors(mri, mask, A, 0.3, "cpu")
    F = lambda o: float(score(torch.eye(4), OCTTemplate(o, mask, A, 0.3, inner, device="cpu"), mt))
    same, inverted = F(mri), F(np.where(mask, 7.0 - 3.0 * mri, 0.0).astype(np.float32))
    assert same > 0.9999 and abs(inverted - same) < 1e-5, (same, inverted)


def test_score_of_the_pair():
    """F is high at the true pose of the pair and much lower 0.5 mm off it."""
    mri, (o, mo, A_O), T = pair()
    tpl, mt = OCTTemplate(o, mo, A_O, 0.3, eroded(mo, 0.8, H), device="cpu"), MRITensors(*mri, 0.3, "cpu")
    off = np.eye(4)
    off[:3, 3] = [0.5, 0.0, 0.0]
    F_true, F_off = (float(score(torch.tensor(x, dtype=torch.float32), tpl, mt)) for x in (T, off @ T))
    assert F_true > 0.9 and F_off < 0.5 * F_true, (F_true, F_off)


def test_background_lowers_the_score():
    """With the MRI foreground cut in half, the OCT points over MRI background (tensors 0) stay in the sum and lower F, while
    F over the points still on tissue stays high."""
    (mri, mask, A), _, _ = pair()
    tpl = OCTTemplate(mri, mask, A, 0.3, eroded(mask, 0.8, H), device="cpu")
    half = mask & (world(mask.shape, A)[..., 0] < 0.5)
    mt = MRITensors(mri, half, A, 0.3, "cpu")
    over = G.sample_world(torch.as_tensor(half[None], dtype=torch.float32), A, tpl.pts)[0] > 0.5
    assert 0.3 < float(over.float().mean()) < 0.8
    F_half = float(score(torch.eye(4), tpl, mt))
    F_over = float(correlation(tensors(tpl.g, tpl.eps)[:, over], mt.at(tpl.pts)[:, over]))
    assert F_over > 0.95 and F_half < F_over - 0.15, (F_half, F_over)


def test_no_overlap_gives_a_finite_gradient():
    """A pose with every OCT point beyond the MRI grid scores 0 with a zero, finite gradient, so its candidate stays at L = 1
    and loses the selection instead of breaking it."""
    mri, (o, mo, A_O), T = pair(seed=3)
    far = np.eye(4)
    far[:3, 3] = [200.0, 0.0, 0.0]
    Tt = torch.tensor(far @ T, dtype=torch.float32, requires_grad=True)
    F = score(Tt, OCTTemplate(o, mo, A_O, 0.3, eroded(mo, 0.8, H), device="cpu"), MRITensors(*mri, 0.3, "cpu"))
    F.backward()
    assert float(F) == 0.0 and bool(torch.isfinite(Tt.grad).all()) and float(Tt.grad.abs().max()) == 0.0


def test_refine_recovers_offset_pose():
    mri, oct_, T = pair()
    off = np.eye(4)
    off[:3, :3] = Rotation.from_rotvec(np.radians(3.0) * np.array([1.0, 1.0, 0.0]) / np.sqrt(2)).as_matrix()
    off[:3, 3] = [0.6, -0.5, 0.6]
    T0 = off @ T
    pts = world(oct_[0].shape, oct_[2])[oct_[1]]
    err = lambda X: np.linalg.norm(G.apply_affine(X, pts) - G.apply_affine(T, pts), axis=1).mean()
    T1, info = refine_one(T0, mri, oct_, Params(), "cpu")
    assert err(T0) > 0.9 and err(T1) < 0.05, (err(T0), err(T1))
    assert [f["sigma_mm"] for f in info["F"]] == list(Params().sigmas_mm)
    assert info["F"][0]["start"] < 0.5 < 0.9 < info["F"][-1]["end"]


def test_refine_selects_the_lowest_final_L():
    """refine sorts the poses by final L, and the start near the truth wins even when it is not first. The search rank of a
    start is its index in the list."""
    mri, oct_, T = pair()
    wrong = np.eye(4)
    wrong[:3, :3] = Rotation.from_rotvec(np.radians(60.0) * np.array([0.0, 0.0, 1.0])).as_matrix()
    wrong[:3, 3] = [2.0, 0.0, 0.0]
    near = np.eye(4)
    near[:3, 3] = [0.3, -0.2, 0.3]
    poses = refine([wrong @ T, near @ T], mri, oct_, dataclasses.replace(Params(), iters=60), "cpu")
    pts = world(oct_[0].shape, oct_[2])[oct_[1]]
    err = lambda X: np.linalg.norm(G.apply_affine(X, pts) - G.apply_affine(T, pts), axis=1).mean()
    assert [p["search_rank"] for p in poses] == [1, 0] and poses[0]["L"] < poses[1]["L"], [(p["search_rank"], p["L"]) for p in poses]
    assert err(poses[0]["T"]) < 0.1 and all(len(p["F"]) == len(Params().sigmas_mm) for p in poses), err(poses[0]["T"])


def test_refine_tie_rule(monkeypatch):
    """A final L within TIE of the lowest is a tie, which the better search rank wins."""
    shape = (20, 20, 20)
    A = grid(shape)
    vol = ndimage.gaussian_filter(np.random.default_rng(0).standard_normal(shape), 1.5).astype(np.float32)
    box = np.ones(shape, bool)
    starts = [np.eye(4) for _ in range(3)]
    for i, T in enumerate(starts):
        T[0, 3] = i                                                                  # marks the start of search rank i
    for L, ranks in (((0.2 + 5e-6, 0.2, 0.3), [0, 1, 2]), ((0.2 + 5e-4, 0.2, 0.3), [1, 0, 2])):
        stub = lambda objective, c, rho, T0, params, device, L=L: (T0, 0.5, L[int(T0[0, 3])], 0.4)
        monkeypatch.setattr("octreg.refine.fit_adam", stub)
        poses = refine(starts, (vol, box, A), (vol, box, A), Params(), "cpu")
        assert [p["search_rank"] for p in poses] == ranks, [(p["search_rank"], p["L"]) for p in poses]


def test_prior_holds_the_scale_the_data_do_not_decide():
    """The prior restores a scale that F does not decide, and with lam 0 the size of the block is kept."""
    mri, oct_, T = pair(seed=3)
    c = G.box(oct_[2], oct_[0].shape)[0]
    T1 = refine_one(T, mri, oct_, dataclasses.replace(Params(), lam=0.0, iters=80), "cpu")[0]
    assert np.abs(decompose(T1, c)[2]).max() < 0.01, decompose(T1, c)[2]

    rng = np.random.default_rng(4)
    shape, shape_o = (64, 64, 64), (48, 48, 48)
    A, A_o = grid(shape), grid(shape_o)
    s = ndimage.gaussian_filter(rng.standard_normal((64, 64)), 2.0)
    s /= s.std()
    mri = np.broadcast_to((1.0 + 0.3 * s)[:, None, :], shape).astype(np.float32)   # array axis 1 is world z
    xo = world(shape_o, A_o)
    mo = ((xo[..., 0] / 3.0) ** 2 + (xo[..., 1] / 2.8) ** 2 <= 1) & (np.abs(xo[..., 2]) < 2.7)
    o = np.where(mo, 2.0 - np.broadcast_to(s[8:56, None, 8:56], shape_o), 0.0).astype(np.float32)   # T_true = identity
    fg, c = np.ones(shape, bool), G.box(A_o, shape_o)[0]
    tpl, mt = OCTTemplate(o, mo, A_o, 0.6, eroded(mo, 0.8, H), device="cpu"), MRITensors(mri, fg, A, 0.6, "cpu")
    f = lambda *v: torch.tensor(v, dtype=torch.float32)
    cc = torch.as_tensor(c, dtype=torch.float32)
    F = [float(score(compose(f(0, 0, 0), cc, f(0, 0, ls), f(0, 0, 0), cc), tpl, mt)) for ls in (0.0, np.log(1.1), np.log(0.9))]
    assert F[0] > 0.9 and max(F) - min(F) < 1e-5, F
    T1 = refine_one(np.diag([1.0, 1.0, 1.1, 1.0]), (mri, fg, A), (o, mo, A_o), Params(), "cpu")[0]
    ls = decompose(T1, c)[2]
    assert abs(ls[2]) < 0.005 and np.abs(ls).max() < 0.02, ls


def test_a_linear_field_carries_the_gradients(monkeypatch):
    """A linear field d(y) = M y turns the OCT gradients and scores as the affine (I + M) T to first order. The gradients
    are turned by I - M^T, much closer to those of the affine than unturned or turned by I - M."""
    from octreg import features
    from octreg.deform import LatticeField
    mri, (o, mo, A_O), T = pair(seed=2)
    tpl = OCTTemplate(o, mo, A_O, 0.3, eroded(mo, 0.8, H), device="cpu")
    mt = MRITensors(*mri, 0.3, "cpu")
    t = lambda X: torch.tensor(X, dtype=torch.float32)
    y = G.apply_affine(t(T), tpl.pts)
    lo, hi = y.min(0).values.numpy() - 8, y.max(0).values.numpy() + 8
    field = LatticeField(lo, hi, 4.0, "cpu")
    M = np.array([[0.04, 0.02, 0.0], [0.0, -0.03, 0.02], [0.02, 0.0, 0.03]])
    nodes = np.stack(np.meshgrid(*[lo[a] + field.h * np.arange(field.n[a]) for a in range(3)], indexing="ij"), -1).reshape(-1, 3)
    seen = []
    monkeypatch.setattr(features, "tensors", lambda g, eps: seen.append(g) or tensors(g, eps))
    with torch.no_grad():
        field.c.copy_(t((nodes @ M.T).T))
        TM = np.eye(4)
        TM[:3, :3] += M
        F_field, F_affine = float(score(t(T), tpl, mt, field)), float(score(t(TM @ T), tpl, mt))
        g = torch.linalg.inv(t(T)[:3, :3]).T @ tpl.g
        F_unturned = float(correlation(tensors(g, tpl.eps), mt.at(y + field(y))))
    want = torch.linalg.inv(t(TM @ T)[:3, :3]).T @ tpl.g
    err = lambda x: float(((x - want).norm(dim=0) / want.norm(dim=0)).median())
    assert err(seen[0]) < 0.25 * min(err(g), err(g - t(M) @ g)), (err(seen[0]), err(g), err(g - t(M) @ g))
    assert abs(F_field - F_affine) < 0.25 * abs(F_unturned - F_affine), (F_field, F_affine, F_unturned)


def test_a_field_moves_the_points():
    """A constant field v scores as the translation by v after T."""
    from octreg.deform import LatticeField
    mri, (o, mo, A_O), T = pair(seed=1)
    tpl = OCTTemplate(o, mo, A_O, 0.3, eroded(mo, 0.8, H), device="cpu")
    mt = MRITensors(*mri, 0.3, "cpu")
    v = np.array([0.2, -0.15, 0.1])
    field = LatticeField(np.full(3, -20.0), np.full(3, 20.0), 4.0, "cpu")
    with torch.no_grad():
        field.c.copy_(torch.as_tensor(np.repeat(v[:, None], field.c.shape[1], 1), dtype=torch.float32))
    Tv = np.eye(4)
    Tv[:3, 3] = v
    t = lambda X: torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        assert abs(float(score(t(T), tpl, mt, field)) - float(score(t(Tv @ T), tpl, mt))) < 1e-5

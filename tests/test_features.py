"""§2 structure score (octreg/features.py) and the §4 refinement on it: world gradients of a ramp; the trace-free tensor ignores
the gradient sign; F is 1 for identical volumes and does not change when the OCT contrast is inverted; OCT points over MRI
background lower F (template form); the refinement recovers a pose 1 mm and 3 degrees off; F separates a mirror image; and the
prior holds the scale where the data do not decide it."""
import dataclasses

import numpy as np
import torch
from scipy import ndimage
from scipy.spatial.transform import Rotation

from octreg import geometry as G
from octreg.features import MRITensors, OCTTemplate, correlation, eroded, gradient_field, score, tensors, tracefree
from octreg.params import Params
from octreg.refine import compose, decompose, refine

H = 0.15
MIRROR = np.diag([1.0, 1.0, -1.0, 1.0])


def refine_one(T0, mri, oct_, params, device):
    """refine.refine from one start -> (T, its record)."""
    r = refine([T0], mri, oct_, params, device)[0]
    return r["T"], r


def grid(shape, perm=True):
    """Base-grid affine centred on the world origin: permuted axes with a flip (perm) or two flipped axes."""
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
    inner = ndimage.binary_erosion(mask, iterations=9)                                # beyond the kernel radius (8 voxels)
    assert np.abs(g[:, inner] - np.array([2, 0, -3])[:, None]).max() < 2e-3
    assert np.abs(g[:, ~mask]).max() == 0                                            # zero outside the mask
    flat = gradient_field(np.where(mask, 5.0, 0.0).astype(np.float32), mask, A, 0.3, "cpu").numpy()
    assert np.abs(flat).max() < 1e-3                                                  # the mask edge adds no gradient


def test_tracefree_ignores_the_sign():
    """P(n) = P(-n), trace 0, and <P(n), P(m)> = (n . m)^2 - |n|^2 |m|^2 / 3 over the six channels."""
    n, m = torch.as_tensor(np.random.default_rng(0).normal(size=(2, 3, 200)), dtype=torch.float32)
    P, Q = tracefree(n), tracefree(m)
    assert P.shape == (6, 200) and torch.equal(P, tracefree(-n))
    assert float((P[0] + P[1] + P[2]).abs().max()) < 1e-5
    dot = (n * m).sum(0) ** 2 - (n * n).sum(0) * (m * m).sum(0) / 3
    assert torch.allclose((P * Q).sum(0), dot, atol=1e-4)


def pair(seed=0):
    """MRI: 1 + 0.3 s inside an ellipsoid on 64^3 voxels (s a smoothed Gaussian field); OCT: 60 x 62 x 58 voxels on a
    differently flipped grid showing 2 - tanh(1.5 s) at T x (inverted, non-linear contrast), specimen mask the ellipsoid at T x,
    all measured. -> ((mri, mask, A_M), (oct, mask, measured, A_O), T)."""
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
    return (mri, mask_m, A_M), (oct_, mo, np.ones(shape_o, np.float32), A_O), T


def test_score_identical_and_contrast_free():
    """The same volume as OCT and MRI scores 1 at the identity, also with the OCT contrast inverted and rescaled."""
    (mri, mask, A), _, _ = pair()
    inner = eroded(mask, 0.8, H)
    mt = MRITensors(mri, mask, A, 0.3, "cpu")
    F = lambda o: float(score(torch.eye(4), OCTTemplate(o, mask, A, 0.3, inner, device="cpu"), mt))
    same, inverted = F(mri), F(np.where(mask, 7.0 - 3.0 * mri, 0.0).astype(np.float32))
    assert same > 0.9999 and abs(inverted - same) < 1e-5, (same, inverted)


def test_score_of_the_pair():
    """At the true pose of the pair (inverted, non-linear contrast) F is high, 0.5 mm off it is much lower."""
    mri, (o, mo, _, A_O), T = pair()
    tpl, mt = OCTTemplate(o, mo, A_O, 0.3, eroded(mo, 0.8, H), device="cpu"), MRITensors(*mri, 0.3, "cpu")
    off = np.eye(4)
    off[:3, 3] = [0.5, 0.0, 0.0]
    F_true, F_off = (float(score(torch.tensor(x, dtype=torch.float32), tpl, mt)) for x in (T, off @ T))
    assert F_true > 0.9 and F_off < 0.5 * F_true, (F_true, F_off)


def test_background_lowers_the_score():
    """Template form: with the MRI foreground cut in half, the OCT points over the MRI background (tensors 0) stay in the sum
    and lower F, while the correlation over the points that remain over MRI tissue stays high."""
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
    """A pose that puts every OCT point beyond the MRI grid scores 0 with a zero, finite gradient (no NaN): its fit stays at
    L = 1 and the candidate loses the selection by the lowest final L instead of breaking it."""
    mri, (o, mo, _, A_O), T = pair(seed=3)
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
    pts = world(oct_[0].shape, oct_[3])[oct_[1]]
    err = lambda X: np.linalg.norm(G.apply_affine(X, pts) - G.apply_affine(T, pts), axis=1).mean()
    T1, info = refine_one(T0, mri, oct_, Params(), "cpu")
    assert err(T0) > 0.9 and err(T1) < 0.05, (err(T0), err(T1))
    assert [f["sigma_mm"] for f in info["F"]] == list(Params().sigmas_mm)
    assert info["F"][0]["start"] < 0.5 < 0.9 < info["F"][-1]["end"]


def test_refine_selects_the_lowest_final_L():
    """Of several starts, refine returns every pose sorted by its final L, and the start near the truth wins even when it is not
    the first (the search rank of a start is its position in the list)."""
    mri, oct_, T = pair()
    wrong = np.eye(4)
    wrong[:3, :3] = Rotation.from_rotvec(np.radians(60.0) * np.array([0.0, 0.0, 1.0])).as_matrix()
    wrong[:3, 3] = [2.0, 0.0, 0.0]
    near = np.eye(4)
    near[:3, 3] = [0.3, -0.2, 0.3]
    poses = refine([wrong @ T, near @ T], mri, oct_, dataclasses.replace(Params(), iters=60), "cpu")
    pts = world(oct_[0].shape, oct_[3])[oct_[1]]
    err = lambda X: np.linalg.norm(G.apply_affine(X, pts) - G.apply_affine(T, pts), axis=1).mean()
    assert [p["search_rank"] for p in poses] == [1, 0] and poses[0]["L"] < poses[1]["L"], [(p["search_rank"], p["L"]) for p in poses]
    assert err(poses[0]["T"]) < 0.1 and all(len(p["F"]) == len(Params().sigmas_mm) for p in poses), err(poses[0]["T"])


def test_score_separates_mirror_images():
    """Refined from the true pose and from the same pose in the mirrored OCT frame, the true frame aligns the fine structure
    (F > 0.9) and the mirrored one cannot (F < 0.3)."""
    mri, (o, mo, q, A_O), T = pair(seed=2)
    true = refine_one(T, mri, (o, mo, q, A_O), Params(), "cpu")[1]["F"][-1]["end"]
    mirrored = refine_one(T, mri, (o, mo, q, MIRROR @ A_O), Params(), "cpu")[1]["F"][-1]["end"]
    assert true > 0.9 and mirrored < 0.3, (true, mirrored)


def test_prior_holds_the_scale_the_data_do_not_decide():
    """F does not reward a smaller block: refined from the true pose with lam 0 the size is kept (|log-scale| < 0.01). A
    structure that does not vary along world z gives the same F at every OCT scale along z, so the data do not decide that
    scale; refined from a 10 % stretch along z, the prior (lam) brings it back to 1."""
    mri, oct_, T = pair(seed=3)
    c = G.box(oct_[3], oct_[0].shape)[0]
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
    T1 = refine_one(np.diag([1.0, 1.0, 1.1, 1.0]), (mri, fg, A), (o, mo, np.ones(shape_o, np.float32), A_o), Params(), "cpu")[0]
    ls = decompose(T1, c)[2]
    assert abs(ls[2]) < 0.005 and np.abs(ls).max() < 0.02, ls


def test_gradient_transport_under_a_displacement():
    """score(T, ..., field) carries the OCT gradient through the map y = T x + d(T x) as (I - grad d^T) A^-T g: for a linear d
    the score equals that of the affine (I + B) T to first order in B, and leaving the transport out moves it much more."""
    from octreg.deform import LatticeField
    mri, (o, mo, _, A_O), T = pair(seed=1)
    tpl = OCTTemplate(o, mo, A_O, 0.3, eroded(mo, 0.8, H), device="cpu")
    mt = MRITensors(*mri, 0.3, "cpu")
    B = 0.05 * np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.5], [0.0, -0.5, 0.0]])       # a small turn: it reorients the edges
    y0 = G.apply_affine(T, G.box(A_O, o.shape)[0])
    field = LatticeField(y0 - 12.0, y0 + 12.0, 2.0, "cpu")
    with torch.no_grad():
        n = field.n
        nodes = (field.lo + 2.0 * torch.stack(torch.meshgrid(*[torch.arange(k, dtype=torch.float32) for k in n], indexing="ij"), -1))
        d = (nodes - torch.as_tensor(y0, dtype=torch.float32)) @ torch.as_tensor(B.T, dtype=torch.float32)
        field.c.copy_(d.permute(3, 0, 1, 2)[None])
    Ta = np.eye(4)
    Ta[:3, :3] = np.eye(3) + B
    Ta[:3, 3] = -B @ y0
    t = lambda X: torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        F_field = float(score(t(T), tpl, mt, field))
        F_affine = float(score(t(Ta @ T), tpl, mt))
        y = G.apply_affine(t(T), tpl.pts)
        g = torch.linalg.inv(t(T)[:3, :3]).T @ tpl.g
        F_plain = float(correlation(tensors(g, tpl.eps), mt.at(y + field(y))))           # positions moved, gradients not
    assert abs(F_plain - F_affine) > 1e-3 and abs(F_field - F_affine) < 0.1 * abs(F_plain - F_affine), (F_field, F_affine, F_plain)

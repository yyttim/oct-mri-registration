"""§3 orientation search (octreg/search.py) on a 0.6 mm search grid: the FFT maps F and O are the direct weighted correlations
at every translation, also over the edge of the MRI crop and with partial-volume masks; an OCT block cut through a known
rotation of the search set, with inverted non-linear contrast on a permuted, flipped grid, is found at that rotation and the
nearest translations, embedded or cut from a larger specimen, with and without the outline; a mirror image scores lower."""
import dataclasses

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from octreg import geometry as G
from octreg.features import tracefree
from octreg.params import Params
from octreg.search import Outline, Searcher, angle_deg, search

H = 0.6                                                  # Params().search_mm
FAST = dataclasses.replace(Params(), n_rot=24, topk=4)
SEMI = np.array([6.5, 5.5, 4.5])                         # specimen ellipsoid (mm) in the MRI world
MIRROR = np.diag([1.0, 1.0, -1.0, 1.0])


def structure(x):
    """A smooth random field at world points x [..., 3] (mm): 12 plane waves of wavelength 5 - 9 mm."""
    rng = np.random.default_rng(0)
    k = rng.normal(size=(12, 3))
    k *= (2 * np.pi / rng.uniform(5.0, 9.0, 12) / np.linalg.norm(k, axis=1))[:, None]
    return np.cos(x @ k.T + rng.uniform(0, 2 * np.pi, 12)).sum(-1) / np.sqrt(6)


def phantom(rot_index, cut=None):
    """MRI: 36^3 voxels, 1 + 0.3 structure inside the specimen ellipsoid M, 0.1 outside. OCT: 27 x 25 x 23 voxels on a permuted,
    flipped grid, all measured; voxel x shows 2 - tanh(1.5 structure) at T_true x inside the specimen (mask M(T_true x)), 0.3
    embedding outside. T_true = R (x - c_o) + (0.4, -0.3, 0.2) mm, R = rotation rot_index of the search set, c_o the OCT box
    centre.
    cut: OCT axis-0 index from which the block was cut off the specimen (embedding there, over MRI tissue and medium alike).
    -> ((mri, M, A_M), (oct, mask, measured, A_O), T_true)."""
    shape_m = (36, 36, 36)
    A_M = np.diag([H, H, H, 1.0])
    A_M[:3, 3] = -H * 35 / 2 + np.array([1.0, -0.5, 0.3])
    xm = G.apply_affine(A_M, np.indices(shape_m).reshape(3, -1).T.astype(float))
    M = (((xm / SEMI) ** 2).sum(-1) <= 1).reshape(shape_m)
    mri = np.where(M, 1.0 + 0.3 * structure(xm).reshape(shape_m), 0.1).astype(np.float32)
    shape = (27, 25, 23)
    A_O = np.zeros((4, 4))
    A_O[0, 1], A_O[1, 0], A_O[2, 2], A_O[3, 3] = H, -H, H, 1.0
    A_O[:3, 3] = (12.0, -3.0, 7.0)
    T = np.eye(4)
    T[:3, :3] = G.rotations(FAST.n_rot)[rot_index]
    T[:3, 3] = np.array([0.4, -0.3, 0.2]) - T[:3, :3] @ G.box(A_O, shape)[0]
    xo = G.apply_affine(T, G.apply_affine(A_O, np.indices(shape).reshape(3, -1).T.astype(float)))
    w = (((xo / SEMI) ** 2).sum(-1) <= 1).reshape(shape)
    if cut is not None:
        w &= (np.arange(shape[0]) < cut)[:, None, None]
    oct_ = np.where(w, 2.0 - np.tanh(1.5 * structure(xo).reshape(shape)), 0.3).astype(np.float32)
    return (mri, M.astype(np.float32), A_M), (oct_, w.astype(np.float32), np.ones(shape, np.float32), A_O), T


def index_of(s, T):
    """Flat translation index of the Searcher s nearest to the pose T (template centre at T c_o)."""
    u = G.apply_affine(np.linalg.inv(s.A_M), G.apply_affine(T, s.c_o)) - (s.n - 1) / 2.0
    return int(np.ravel_multi_index(tuple(np.rint(u).astype(int)), s.shape))


def direct_F(s, mri, R, index):
    """F at one translation as a direct sum: the centred correlation, over the rotated template's points (the eroded specimen
    mask), of its tensors with the MRI tensors under it (zero outside the MRI foreground and beyond the crop)."""
    n, pad = s.n, (s.shape[0] - mri[0].shape[0]) // 2
    I, M = (torch.as_tensor(x, dtype=torch.float32) for x in mri[:2])
    Mb = (M > 0.5).float()
    g = torch.einsum("ij,jklm->iklm", s.to_world, G.nc_gradient(I * Mb, Mb, s.sig))
    TM = F.pad(tracefree(g / torch.sqrt((g * g).sum(0) + g.norm(dim=0)[Mb > 0].median() ** 2))[None], [pad] * 6)[0]
    v, w, _, inner = G.sample_world(s.src, s.A_src, s.c_t + s.tgrid @ torch.as_tensor(R, dtype=torch.float32)).reshape(4, n, n, n)
    wb = (w > 0.5).float()
    g = torch.einsum("ij,jklm->iklm", s.to_world, G.nc_gradient(v * wb, wb, s.sig))
    a, wt = tracefree(g / torch.sqrt((g * g).sum(0) + s.eps_O ** 2)), (inner > 0.5).float()
    i, j, k = np.unravel_index(index, s.shape)
    b = TM[:, i:i + n, j:j + n, k:k + n]
    N = wt.sum()
    da, db = a - (wt * a).sum(dim=(1, 2, 3), keepdim=True) / N, b - (wt * b).sum(dim=(1, 2, 3), keepdim=True) / N
    return float((wt * da * db).sum() / torch.sqrt((wt * da * da).sum() * (wt * db * db).sum()))


def test_maps_are_the_direct_sums():
    """F at the true rotation and O at the identity rotation, at the true translation, shifted ones and one with OCT tissue
    beyond the crop, equal the direct sums (O: search.Outline, the outline in direct form), also with partial-volume masks."""
    mri, oct_, T = phantom(5)
    for smooth in (False, True):
        m, o = mri, oct_
        if smooth:                                                          # M^2 != M, w^2 != w
            Ms, ws = gaussian_filter(mri[1], 0.7), gaussian_filter(oct_[1], 0.7)
            m, o = (mri[0], Ms, mri[2]), (oct_[0], ws, oct_[2], oct_[3])
        s = Searcher(*m, *o, FAST.search_sigma_mm, FAST.erode_mm, "cpu")
        assert s.n % 2 == 1 and all(d % 2 == 1 for d in oct_[0].shape)      # template points on OCT voxels at the identity
        R = T[:3, :3]
        Fm, _ = s.score(R)
        _, Om = s.score(np.eye(3))
        out = Outline(m[1], m[2], o[1], o[2], o[3], "cpu")
        tissue = G.apply_affine(o[3], np.argwhere(o[1] > 0.5).astype(float))
        beyond = lambda P: np.mean(np.any(np.abs(G.apply_affine(np.linalg.inv(m[2]) @ P, tissue) - 17.5) > 17.5, axis=1))
        t = np.array(np.unravel_index(index_of(s, T), s.shape))
        for u in (t, t + (2, -1, 0), t + (-1, 1, 3), t + (-11, 0, 0)):        # (-11, 0, 0): OCT tissue beyond the crop
            idx = int(np.ravel_multi_index(tuple(u), s.shape))
            assert bool(s.valid.reshape(-1)[idx])
            assert abs(float(Fm.reshape(-1)[idx]) - direct_F(s, m, R, idx)) < 1e-3
            P = s.pose(np.eye(3), idx)
            assert abs(float(Om.reshape(-1)[idx]) - float(out(torch.as_tensor(P, dtype=torch.float32)))) < 1e-4
            assert (beyond(s.pose(R, idx)) > 0.1 and beyond(P) > 0.1) == (u[0] < t[0] - 5)
        assert float(Fm.reshape(-1)[index_of(s, T)]) > 0.8


@pytest.mark.parametrize("cut", [None, 14])
@pytest.mark.parametrize("outline", [True, False])
@pytest.mark.parametrize("rot_index", [5, 17])
def test_search_recovers_the_pose(rot_index, outline, cut):
    """The best candidate has the true rotation and a centre within a search voxel of the truth (embedded block); a block cut
    from a larger specimen can slide about one voxel more towards its cut face, where the MRI tissue continues and no OCT point
    is scored. O is one-sided: embedding over MRI tissue beyond the cut costs nothing."""
    mri, oct_, T = phantom(rot_index, cut)
    cands, info = search(*mri, *oct_, FAST, "cpu", outline)
    assert info["n_orientations"] == FAST.n_rot and len(cands) == FAST.topk and info["top1"] == cands[0]["S"]
    assert [c["S"] for c in cands] == sorted((c["S"] for c in cands), reverse=True)
    for c in cands:
        assert np.linalg.det(c["T"][:3, :3]) > 0
        if outline:
            assert abs(c["S"] - (c["F"] + c["O"]) / 2) < 1e-6
        else:
            assert c["O"] is None and c["S"] == c["F"]
    c_o = G.box(oct_[3], oct_[0].shape)[0]
    centres = [G.apply_affine(c["T"], c_o) for c in cands]
    for i in range(len(cands)):
        for j in range(i):
            assert (np.linalg.norm(centres[i] - centres[j]) >= FAST.nms_mm
                    or angle_deg(cands[i]["T"][:3, :3], cands[j]["T"][:3, :3]) >= FAST.nms_deg)
    best = cands[0]
    assert angle_deg(best["T"][:3, :3], T[:3, :3]) < 1e-3
    err = np.linalg.norm(centres[0] - G.apply_affine(T, c_o))
    assert err < (H if cut is None else 1.5 * H), err
    assert best["F"] > 0.85 and info["top1"] > info["top2"] + 0.1, (best, info)
    if outline:
        assert best["O"] > 0.9, best["O"]                        # one-sided: embedding beyond a cut face costs nothing


@pytest.mark.parametrize("outline", [True, False])
def test_mirror_image_scores_lower(outline):
    """Proper rotations only: the same block in the mirrored OCT frame finds no pose as good as the true frame's."""
    mri, (o, w, q, A_O), _ = phantom(5)
    true = search(*mri, o, w, q, A_O, FAST, "cpu", outline)[1]["top1"]
    mirrored = search(*mri, o, w, q, MIRROR @ A_O, FAST, "cpu", outline)[1]["top1"]
    assert true > mirrored + 0.2, (true, mirrored)


def test_thin_specimen():
    """A specimen thinner than the search smoothing is still scored (its points lie one voxel inside its mask); a mask with no
    voxel erode_mm inside it is refused with a ValueError."""
    mri, oct_, _ = phantom(17, cut=10)
    cands, info = search(*mri, *oct_, FAST, "cpu", False)
    assert all(np.isfinite(c["S"]) for c in cands) and np.isfinite(info["top1"])
    w = np.zeros_like(oct_[1])
    w[w.shape[0] // 2] = 1                                                  # one plane: nothing 0.8 mm inside
    with pytest.raises(ValueError, match="fewer than 10 OCT points"):
        search(*mri, oct_[0], w, oct_[2], oct_[3], FAST, "cpu", False)


def test_oblique_mri_axes_and_per_mm_eps():
    """The search works in world coordinates: on an MRI grid whose axes are turned 25 degrees off the world axes it recovers the
    true rotation and centre as on an axis-aligned grid. Both eps are medians of per-mm gradients, as in octreg.features."""
    from octreg.features import gradient_field, median_norm
    (_, _, A_M), oct_, T = phantom(5)
    Rq = G.rotations(1000)[37]                                              # an oblique turn of the MRI axes
    c = G.apply_affine(A_M, np.full(3, 17.5))
    A2 = np.eye(4)
    A2[:3, :3] = Rq @ np.diag([H, H, H])
    A2[:3, 3] = c - A2[:3, :3] @ np.full(3, 17.5)
    xm = G.apply_affine(A2, np.indices((36, 36, 36)).reshape(3, -1).T.astype(float))
    M = (((xm / SEMI) ** 2).sum(-1) <= 1).reshape(36, 36, 36)
    mri = (np.where(M, 1.0 + 0.3 * structure(xm).reshape(M.shape), 0.1).astype(np.float32), M.astype(np.float32), A2)
    cands, info = search(*mri, *oct_, FAST, "cpu", False)
    assert angle_deg(cands[0]["T"][:3, :3], T[:3, :3]) < 1e-3, angle_deg(cands[0]["T"][:3, :3], T[:3, :3])
    c_o = G.box(oct_[3], oct_[0].shape)[0]
    assert np.linalg.norm(G.apply_affine(cands[0]["T"], c_o) - G.apply_affine(T, c_o)) < 1.5 * H
    s = Searcher(*mri, *oct_, FAST.search_sigma_mm, FAST.erode_mm, "cpu")
    wb = (oct_[1] > 0.5).astype(np.float32)
    ref = float(median_norm(gradient_field(oct_[0], wb, oct_[3], FAST.search_sigma_mm, "cpu"), wb))
    assert abs(float(s.eps_O) - ref) < 1e-4 * ref, (float(s.eps_O), ref)

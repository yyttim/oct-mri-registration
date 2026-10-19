"""End-to-end tests of register and the octreg command line on a small synthetic pair, on the CPU."""
import dataclasses
import json

import nibabel as nib
import numpy as np
import pytest
import torch
from PIL import Image
from scipy import ndimage

from octreg import geometry as G, io, preprocess as pp
from octreg.cli import main
from octreg.params import Params
from octreg.refine import TIE
from octreg.register import OUTPUTS, QC_CELL_IN, QC_COLOURS, QC_DPI, QC_HEAD_IN, register

# the specimen is 6 mm across, so the texture and search smoothings are smaller than the defaults for real blocks
FAST = dataclasses.replace(Params(), fine_mm=0.08, n_rot=200, topk=4, iters=40, texture_smooth_mm=0.32, search_sigma_mm=0.6)
# for the runs whose pose does not matter
TINY = {"fine_mm": 0.08, "texture_smooth_mm": 0.32, "search_sigma_mm": 0.6, "n_rot": 8, "topk": 1, "iters": 5}
POSE_KEYS = {"search_rank", "search_F", "search_O", "L", "outline_used", "T_search", "F", "scale_per_oct_axis"}


def distance(T1, T2, pts):
    return np.linalg.norm(G.apply_affine(T1, pts) - G.apply_affine(T2, pts), axis=1).mean()


def sample(world, grid):
    """Trilinear samples of grid = (volume [D, H, W], affine) at world points [N, 3]."""
    vol, A = grid
    return G.sample_world(torch.as_tensor(vol[None]), A, torch.as_tensor(world, dtype=torch.float32))[0].numpy()


def pair(tmp, seed=0):
    """Write a synthetic OCT block and MRI crop related by a known affine. The OCT has agarose at the mean tissue intensity,
    section offsets, tile seams, a black tile and inverted contrast.
    -> (OCT path, MRI path, T_true from OCT to MRI world, OCT specimen points in OCT world mm)."""
    rng = np.random.default_rng(seed)
    g = ndimage.gaussian_filter(rng.standard_normal((96,) * 3), 5.0)
    A_g = np.diag([0.15, 0.15, 0.15, 1.0])
    A_g[:3, 3] = -0.15 * 47.5
    classes = ((1 / (1 + np.exp(-g / (0.25 * g.std())))).astype(np.float32), A_g)
    centre, semi = np.array([0.3, -0.2, 0.1]), np.array([3.0, 2.6, 2.3])  # specimen ellipsoid in the MRI world, mm

    def tissue(x):
        return (((x - centre) / semi) ** 2).sum(-1) <= 1

    A_M = np.array([[0.15, 0, 0, 0], [0, 0, 0.15, 0], [0, -0.15, 0, 0], [0, 0, 0, 1.0]])   # i -> R, j -> I, k -> A
    shape_m = (48, 52, 44)
    A_M[:3, 3] = -A_M[:3, :3] @ ((np.array(shape_m) - 1) / 2)
    x = G.apply_affine(A_M, np.indices(shape_m).reshape(3, -1).T.astype(float))
    m = tissue(x)
    mri = np.where(m, 600 + 200 * sample(x, classes), 80) + rng.normal(0, 70, m.shape)    # overlapping classes, one valley
    io.save_nifti(np.maximum(mri, 0).reshape(shape_m).astype(np.float32), A_M, tmp / "mri.nii.gz")

    shape_o = (72, 68, 62)
    A_O = np.diag([-0.08, -0.08, -0.08, 1.0])                                            # i -> L, j -> P, k -> I
    A_O[:3, 3] = np.array([12.0, -3.0, 7.0]) - A_O[:3, :3] @ ((np.array(shape_o) - 1) / 2)
    c_o = G.apply_affine(A_O, (np.array(shape_o) - 1) / 2)
    T = np.eye(4)
    T[:3, :3] = G.rotations(FAST.n_rot)[7] @ _rot([0.6, -0.5, 0.62], 4.0) @ np.diag([1.03, 1.0, 1.0])
    T[:3, 3] = centre + np.array([0.25, 0.1, -0.15]) - T[:3, :3] @ c_o
    xo = G.apply_affine(A_O, np.indices(shape_o).reshape(3, -1).T.astype(float))
    xm = G.apply_affine(T, xo)
    inside = tissue(xm).reshape(shape_o)
    tex = ndimage.gaussian_filter(rng.standard_normal(shape_o), 1.5)
    stripes = rng.normal(0, 0.05, shape_o[2])[None, None, :] + 0.1 * (np.arange(shape_o[0]) % 24 < 2)[:, None, None]
    bright = (0.8 + 0.4 * (1 - sample(xm, classes))).reshape(shape_o)                    # inverted contrast
    oct_ = 1000 * (1 + stripes) * np.where(inside, bright * (1 + 0.25 * tex / tex.std()), 1.0)
    oct_ = np.maximum(oct_ * (1 + 0.1 * rng.standard_normal(shape_o)), 1).astype(np.float32)
    oct_[:, :10, :10] = 0
    io.save_nifti(oct_, A_O, tmp / "oct.nii.gz")
    return tmp / "oct.nii.gz", tmp / "mri.nii.gz", T, xo[inside.ravel() & (oct_.ravel() > 0)]


def png(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(int)


def _rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis) * np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return torch.linalg.matrix_exp(torch.as_tensor(K)).numpy()


def nii(path):
    return nib.load(str(path)).get_fdata()


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    return pair(tmp_path_factory.mktemp("pair"))


def test_register_and_apply(data, tmp_path):
    """The outputs and result.json, octreg apply with and without --inverse, and exit code 2 for bad inputs."""
    oct_path, mri_path, T_true, pts = data
    out = tmp_path / "run"
    register(oct_path, mri_path, out, params=FAST, device="cpu")
    assert all((out / f).is_file() for f in OUTPUTS)
    saved = json.loads((out / "result.json").read_text())
    T = np.loadtxt(out / "T_oct2mri.txt")
    assert distance(T, T_true, pts) < 0.4 and np.allclose(saved["T_oct2mri"], T) and np.linalg.det(T[:3, :3]) > 0

    pose = saved["pose"]
    assert set(pose) == POSE_KEYS and pose["outline_used"]
    assert set(saved) == {"octreg_version", "inputs", "params", "T_oct2mri", "pose", "candidates", "foreground", "seconds"}
    assert set(saved["seconds"]) == {"mri", "oct_fine_grid", "oct_mask", "search", "refine", "outputs", "total"}
    cands = saved["candidates"]
    assert len(cands) == FAST.topk and all(set(c) == {"search_rank", "search_F", "search_O", "L"} for c in cands)
    lo = min(c["L"] for c in cands)                            # within TIE of the lowest L the better search rank comes first
    key = [(0, c["search_rank"]) if c["L"] <= lo + TIE else (1, c["L"]) for c in cands]
    assert key == sorted(key)
    assert sorted(c["search_rank"] for c in cands) == list(range(FAST.topk))
    assert all(pose[k] == cands[0][k] for k in ("search_rank", "search_F", "search_O", "L"))
    assert [f["sigma_mm"] for f in pose["F"]] == list(FAST.sigmas_mm)
    assert all(f["end"] >= f["start"] - 0.01 for f in pose["F"])                  # the prior may trade a little F
    scale = pose["scale_per_oct_axis"]                          # true scale 1.03, 1, 1
    assert 0.9 < min(scale) and max(scale) < 1.1
    assert saved["params"] == json.loads(json.dumps(FAST.to_dict()))

    assert saved["inputs"]["oct_mask"] is None and saved["inputs"]["mri_mask"] is None
    assert saved["foreground"]["mri"]["threshold"] > 0
    # the OCT histogram has no valley, so the threshold is 0 and the texture finds the specimen
    oct_img = nib.load(str(oct_path))
    raw_h = G.pool_iso(np.asarray(oct_img.dataobj, np.float32), oct_img.affine, 0.15)[0]
    assert pp.foreground(raw_h, 0.15, Params())[1]["threshold"] == 0
    assert 0.9 < saved["foreground"]["oct"]["volume_cm3"] / (len(pts) * 0.08 ** 3 / 1e3) < 1.3

    ref = nib.load(str(out / "oct_in_mri.nii.gz"))
    assert ref.shape == nib.load(str(mri_path)).shape and np.allclose(ref.affine, nib.load(str(mri_path)).affine, atol=1e-6)
    assert main(["apply", "--run", str(out), "--moving", str(oct_path), "--reference", str(mri_path), "-o",
                 str(tmp_path / "again.nii.gz")]) == 0
    np.testing.assert_allclose(nii(tmp_path / "again.nii.gz"), ref.get_fdata(), rtol=1e-5, atol=1e-3)
    assert main(["apply", "--run", str(out), "--moving", str(mri_path), "--reference", str(oct_path), "-o",
                 str(tmp_path / "back.nii.gz"), "--inverse"]) == 0
    back = nib.load(str(tmp_path / "back.nii.gz"))
    assert back.shape == nib.load(str(oct_path)).shape and np.asarray(back.dataobj).max() > 500
    mio = nib.load(str(out / "mri_in_oct.nii.gz"))
    assert np.allclose(np.linalg.norm(mio.affine[:3, :3], axis=0), FAST.base_mm)
    assert main(["apply", "--run", str(out), "--moving", str(mri_path), "--reference", str(out / "mri_in_oct.nii.gz"), "-o",
                 str(tmp_path / "back_base.nii.gz"), "--inverse"]) == 0
    np.testing.assert_allclose(nii(tmp_path / "back_base.nii.gz"), mio.get_fdata(), rtol=1e-5, atol=1e-3)

    qc, montage = png(out / "qc.png"), png(out / "qc_montage.png")        # 3 and 12 rows of planes, both outlines drawn
    assert [round((x.shape[0] / QC_DPI - QC_HEAD_IN) / QC_CELL_IN, 2) for x in (qc, montage)] == [3, 12]
    assert all((x == list(bytes.fromhex(c[1:]))).all(-1).sum() > 100 for x in (qc, montage) for c in QC_COLOURS)
    assert main(["register", str(tmp_path / "missing.nii.gz"), str(mri_path), "-o", str(out), "--device", "cpu"]) == 2
    assert main(["register", str(oct_path), str(mri_path), "-o", str(out), "--oct-mask", "missing.nii.gz", "--device", "cpu"]) == 2
    assert main(["register", str(oct_path), str(mri_path), "-o", str(out), "--device", "cpu", "--params",
                 str(tmp_path / "missing.json")]) == 2
    assert main(["apply", "--run", str(tmp_path / "missing_run"), "--moving", str(oct_path), "--reference", str(mri_path),
                 "-o", str(tmp_path / "never.nii.gz")]) == 2


def test_user_masks(tmp_path, monkeypatch):
    """--oct-mask and --mri-mask replace the §1 masks. The OCT mask, on a 0.16 mm grid of another shape than the OCT, keeps
    its own spacing under --oct-spacing-um."""
    import octreg.register as reg
    oct_path, mri_path, T_true, pts = pair(tmp_path, seed=1)
    A = io.load_volume(oct_path).affine.copy()
    A[:3, :3] *= 2
    mask = np.zeros((37, 35, 32), np.uint8)
    mask[tuple(np.rint(G.apply_affine(np.linalg.inv(A), pts)).astype(int).T)] = 1
    io.save_nifti(mask, A, tmp_path / "oct_mask.nii.gz")
    mri = nib.load(str(mri_path))
    io.save_nifti((np.asarray(mri.dataobj) > 350).astype(np.uint8), mri.affine, tmp_path / "mri_mask.nii.gz")
    out, given, fine_mask = tmp_path / "run", [], reg.fine_mask
    monkeypatch.setattr(reg, "fine_mask", lambda fine, A, P, mask: given.append(mask) or fine_mask(fine, A, P, mask))
    res = register(oct_path, mri_path, out, (80, 80, 80), oct_mask=tmp_path / "oct_mask.nii.gz",
                   mri_mask=tmp_path / "mri_mask.nii.gz", params=FAST, device="cpu")
    assert np.allclose(given[0].spacing_mm, 0.16)
    assert res["inputs"]["oct_mask"].endswith("oct_mask.nii.gz") and res["inputs"]["mri_mask"].endswith("mri_mask.nii.gz")
    assert set(res["foreground"]["oct"]) == set(res["foreground"]["mri"]) == {"outside_fraction"}    # neither mask of §1 made
    assert res["pose"]["outline_used"]
    assert distance(np.loadtxt(out / "T_oct2mri.txt"), T_true, pts) < 0.4


def test_oct_spacing_applies_to_a_mask_of_the_oct_shape(data, tmp_path, monkeypatch):
    """--oct-spacing-um applies to a mask of the OCT's shape, and masks with no spacing or no measured voxel are refused. A run
    that stops early leaves no result.json of an earlier run in its directory."""
    import octreg.register as reg
    oct_path, mri_path, _, _ = data
    img = nib.load(str(oct_path))
    A = img.affine.copy()
    A[:3, :3] /= 2
    io.save_nifti(np.asarray(img.dataobj, np.float32), A, tmp_path / "oct.nii.gz")

    class Stop(Exception):
        pass

    def stop(fine, A, P, mask):
        given.append(mask)
        raise Stop

    given = []
    monkeypatch.setattr(reg, "fine_mask", stop)
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "result.json").write_text("{}")
    with pytest.raises(Stop):
        register(tmp_path / "oct.nii.gz", mri_path, tmp_path / "run", (80, 80, 80), oct_mask=tmp_path / "oct.nii.gz",
                 params=FAST, device="cpu")
    assert not (tmp_path / "run" / "result.json").exists()
    assert np.allclose(given[0].spacing_mm, 0.08) and np.allclose(given[0].affine, io.load_volume(oct_path).affine)
    np.save(tmp_path / "oct.npy", np.ones((20, 20, 20), np.float32))
    np.save(tmp_path / "small.npy", np.ones((10, 10, 10), np.uint8))
    with pytest.raises(ValueError, match="needs a spacing of its own"):
        register(tmp_path / "oct.npy", mri_path, tmp_path / "run2", (80, 80, 80), oct_mask=tmp_path / "small.npy", params=FAST,
                 device="cpu")
    assert not (tmp_path / "run2").exists()
    monkeypatch.undo()
    io.save_nifti(-np.ones(img.shape, np.float32), img.affine, tmp_path / "db.nii.gz")
    io.save_nifti(np.ones(img.shape, np.uint8), img.affine, tmp_path / "db_mask.nii.gz")
    with pytest.raises(ValueError, match="no voxel of the mask has a positive OCT intensity"):
        register(tmp_path / "db.nii.gz", mri_path, tmp_path / "run3", oct_mask=tmp_path / "db_mask.nii.gz", params=FAST,
                 device="cpu")


def test_mask_in_another_frame_is_refused(data, tmp_path):
    """A mask in another frame than its image is refused before any output is written."""
    oct_path, mri_path, _, _ = data
    np.save(tmp_path / "oct_mask.npy", np.ones((62, 68, 72), np.uint8))                    # numpy (z, y, x) of the OCT
    nib.save(nib.Nifti1Image(np.ones(nib.load(str(mri_path)).shape, np.uint8), None), str(tmp_path / "mri_mask.nii.gz"))
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="must be in the world of its image"):
        register(oct_path, mri_path, out, (80, 80, 80), oct_mask=tmp_path / "oct_mask.npy", params=FAST, device="cpu")
    with pytest.raises(ValueError, match="must be in the world of its image"):
        register(oct_path, mri_path, out, mri_mask=tmp_path / "mri_mask.nii.gz", params=FAST, device="cpu")
    assert main(["register", str(oct_path), str(mri_path), "-o", str(out), "--oct-spacing-um", "80,80,80", "--oct-mask",
                 str(tmp_path / "oct_mask.npy"), "--device", "cpu"]) == 2
    assert not out.exists()


def test_array_frame(data, tmp_path):
    """An NPY stack in its array frame gives the same pose as the NIfTI it came from, and apply reproduces the overlay. The
    stack is flipped on the three axes the header negates, so its array frame is the header frame up to a translation."""
    oct_path, mri_path, T_true, pts = data
    img = nib.load(str(oct_path))
    np.save(tmp_path / "oct.npy", np.asarray(img.dataobj, np.float32).transpose(2, 1, 0)[::-1, ::-1, ::-1])   # numpy (z, y, x)
    A_arr = np.diag([-0.08, -0.08, -0.08, 1.0])                                                  # NIfTI voxel -> array world
    A_arr[:3, 3] = 0.08 * (np.array(img.shape) - 1)
    T_arr = T_true @ img.affine @ np.linalg.inv(A_arr)                                          # array world -> MRI world
    pts_arr = G.apply_affine(A_arr @ np.linalg.inv(img.affine), pts)
    out = tmp_path / "run"
    register(tmp_path / "oct.npy", mri_path, out, (80, 80, 80), params=FAST, device="cpu")
    T = np.loadtxt(out / "T_oct2mri.txt")
    assert np.linalg.det(T_arr[:3, :3]) > 0 and np.linalg.det(T[:3, :3]) > 0
    assert distance(T, T_arr, pts_arr) < 0.4
    assert main(["apply", "--run", str(out), "--moving", str(tmp_path / "oct.npy"), "--reference", str(mri_path), "-o",
                 str(tmp_path / "again.nii.gz")]) == 0
    np.testing.assert_allclose(nii(tmp_path / "again.nii.gz"), nii(out / "oct_in_mri.nii.gz"), rtol=1e-5, atol=1e-3)


def test_apply_moved_npy_with_spacing(data, tmp_path):
    """octreg apply refuses an NPY run from another path until --oct-spacing-um is given, then gives the same result."""
    stack = np.asarray(nib.load(str(data[0])).dataobj, np.float32).transpose(2, 1, 0)       # numpy (z, y, x)
    np.save(tmp_path / "oct.npy", stack)
    (tmp_path / "moved").mkdir()
    np.save(tmp_path / "moved" / "oct.npy", stack)
    out = tmp_path / "run"
    register(tmp_path / "oct.npy", data[1], out, (80, 80, 80), params=Params.from_dict(TINY), device="cpu")
    assert main(["apply", "--run", str(out), "--moving", str(tmp_path / "oct.npy"), "--reference", str(data[1]), "-o",
                 str(tmp_path / "here.nii.gz")]) == 0
    np.testing.assert_allclose(nii(tmp_path / "here.nii.gz"), nii(out / "oct_in_mri.nii.gz"), rtol=1e-5, atol=1e-3)
    run = ["apply", "--run", str(out), "--moving", str(tmp_path / "moved" / "oct.npy"), "--reference", str(data[1]), "-o",
           str(tmp_path / "moved.nii.gz")]
    assert main(run) == 2
    assert main(run + ["--oct-spacing-um", "80,80,80"]) == 0
    np.testing.assert_array_equal(nii(tmp_path / "moved.nii.gz"), nii(tmp_path / "here.nii.gz"))


def test_register_command(data, tmp_path, capsys):
    """octreg register with --params writes every output."""
    (tmp_path / "params.json").write_text(json.dumps(TINY))
    out = tmp_path / "run"
    assert main(["register", str(data[0]), str(data[1]), "-o", str(out), "--device", "cpu", "--params",
                 str(tmp_path / "params.json")]) == 0
    assert all((out / f).is_file() for f in OUTPUTS)
    res = json.loads((out / "result.json").read_text())
    assert Params.from_dict(res["params"]) == Params.from_dict(TINY)
    assert f"-> {out}" in capsys.readouterr().out

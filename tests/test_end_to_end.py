"""End to end on a small synthetic pair, CPU: an agarose-embedded OCT block (equal mean intensity, section offsets, tile seams,
a black tile, inverted contrast, LPI-like header) and an MRI crop (RIA-like header) related by a known affine. register() runs
with a reduced search, refinement and deformation (FAST). The pose is recovered to < 0.4 mm, and the specimen mask finds the
specimen where an intensity histogram of the OCT has no valley. Without §5 every affine output is written, oct_in_mri is
oct_in_mri_affine and the §5 files of an earlier run are removed; with §5 the field is written on the MRI base grid, the affine
is the same and `octreg apply` reproduces both overlays. `octreg apply` uses a field file put into the run directory; user
masks replace both masks; the same OCT as an NPY stack, taken in its array frame, gives the same pose, and `octreg apply
--oct-spacing-um` replays an NPY run from another path; `octreg register` with --params and --no-deform."""
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
from octreg.register import OUTPUTS, OUTPUTS_DEFORM, QC_CELL_IN, QC_COLOURS, QC_DPI, QC_HEAD_IN, WARP, deform_label, register

FAST = dataclasses.replace(Params(), fine_mm=0.08, n_rot=200, topk=4, iters=40, df_evals=30,
                           texture_smooth_mm=0.32, search_sigma_mm=0.6)
#     the synthetic specimen is 6 mm across (10 search voxels): the default 1.2 mm smoothings of the texture and of the search
#     gradients suit real blocks, so the test pair uses smaller ones
TINY = {"fine_mm": 0.08, "texture_smooth_mm": 0.32, "search_sigma_mm": 0.6, "n_rot": 8, "topk": 1, "iters": 5,
        "df_evals": 2}
#     for the runs whose pose does not matter
POSE_KEYS = {"search_rank", "search_F", "search_O", "L", "outline_used", "T_search", "F", "scale_per_oct_axis"}


def distance(T1, T2, pts):
    return np.linalg.norm(G.apply_affine(T1, pts) - G.apply_affine(T2, pts), axis=1).mean()


def sample(world, grid):
    """Trilinear samples of a volume (numpy [D, H, W], affine) at world points [N, 3]."""
    vol, A = grid
    return G.sample_world(torch.as_tensor(vol[None]), A, torch.as_tensor(world, dtype=torch.float32))[0].numpy()


def pair(tmp, seed=0):
    """-> (OCT path, MRI path, T_true OCT world -> MRI world, OCT specimen points in OCT world mm). The rotation of T_true is
    4 degrees from rotation 7 of the search set of FAST."""
    rng = np.random.default_rng(seed)
    g = ndimage.gaussian_filter(rng.standard_normal((96,) * 3), 5.0)                    # class structure, world grid 0.15 mm
    A_g = np.diag([0.15, 0.15, 0.15, 1.0])
    A_g[:3, 3] = -0.15 * 47.5
    classes = ((1 / (1 + np.exp(-g / (0.25 * g.std())))).astype(np.float32), A_g)
    centre, semi = np.array([0.3, -0.2, 0.1]), np.array([3.0, 2.6, 2.3])                # specimen ellipsoid in the MRI world

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


@pytest.fixture(scope="module")
def deformed(data, tmp_path_factory):
    """The run with §5 on the pair of `data`: -> (run directory, result)."""
    out = tmp_path_factory.mktemp("deformed")
    return out, register(data[0], data[1], out, params=FAST, device="cpu", deform=True)


def test_register_affine_and_apply(data, deformed, tmp_path, capsys):
    """register(..., deform=False): the affine outputs and result.json; apply and apply --inverse; refused inputs exit 2."""
    oct_path, mri_path, T_true, pts = data
    out = tmp_path / "run"
    out.mkdir()
    for f in OUTPUTS_DEFORM:                                   # left by an earlier run with §5 in this directory
        (out / f).write_bytes(b"stale")
    register(oct_path, mri_path, out, params=FAST, device="cpu", deform=False)
    assert all((out / f).is_file() for f in OUTPUTS) and not any((out / f).exists() for f in OUTPUTS_DEFORM)
    saved = json.loads((out / "result.json").read_text())
    T = np.loadtxt(out / "T_oct2mri.txt")
    assert distance(T, T_true, pts) < 0.4 and np.allclose(saved["T_oct2mri"], T) and np.linalg.det(T[:3, :3]) > 0
    assert distance(T, np.loadtxt(deformed[0] / "T_oct2mri.txt"), pts) < 1e-3              # §5 runs after the affine
    assert np.array_equal(nii(out / "oct_in_mri.nii.gz"), nii(out / "oct_in_mri_affine.nii.gz"))

    pose = saved["pose"]
    assert set(pose) == POSE_KEYS and pose["outline_used"] and saved["deform"] is None
    assert set(saved) >= {"octreg_version", "inputs", "params", "T_oct2mri", "pose", "candidates", "foreground", "deform", "seconds"}
    assert set(saved["seconds"]) == {"mri", "oct_fine_grid", "oct_mask", "search", "refine", "deform", "outputs", "total"}
    cands = saved["candidates"]
    assert len(cands) == FAST.topk and all(set(c) == {"search_rank", "search_F", "search_O", "L"} for c in cands)
    lo = min(c["L"] for c in cands)                            # within TIE of the lowest L the better search rank comes first
    key = [(0, c["search_rank"]) if c["L"] <= lo + TIE else (1, c["L"]) for c in cands]
    assert key == sorted(key)
    assert sorted(c["search_rank"] for c in cands) == list(range(FAST.topk))
    assert all(pose[k] == cands[0][k] for k in ("search_rank", "search_F", "search_O", "L"))
    assert [f["sigma_mm"] for f in pose["F"]] == list(FAST.sigmas_mm)
    assert all(f["end"] >= f["start"] - 0.01 for f in pose["F"])                  # the prior may trade a little F
    scale = pose["scale_per_oct_axis"]                          # true 1.03 / 1 / 1
    assert 0.9 < min(scale) and max(scale) < 1.1
    assert saved["params"] == json.loads(json.dumps(FAST.to_dict()))

    assert saved["foreground"]["oct"]["source"] == "octreg" and saved["foreground"]["mri"]["threshold"] > 0
    oct_img = nib.load(str(oct_path))                          # equal intensities: no histogram valley; texture: the specimen
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

    qc, montage = png(out / "qc.png"), png(out / "qc_montage.png")        # 3 and 12 rows of planes, both outlines drawn
    assert [round((x.shape[0] / QC_DPI - QC_HEAD_IN) / QC_CELL_IN, 2) for x in (qc, montage)] == [3, 12]
    assert all((x == list(bytes.fromhex(c[1:]))).all(-1).sum() > 100 for x in (qc, montage) for c in QC_COLOURS)
    assert main(["register", str(tmp_path / "missing.nii.gz"), str(mri_path), "-o", str(out), "--device", "cpu"]) == 2
    assert main(["register", str(oct_path), str(mri_path), "-o", str(out), "--oct-mask", "missing.nii.gz", "--device", "cpu"]) == 2
    assert main(["register", str(oct_path), str(mri_path), "-o", str(out), "--device", "cpu", "--params",
                 str(tmp_path / "missing.json")]) == 2
    assert main(["apply", "--run", str(tmp_path / "missing_run"), "--moving", str(oct_path), "--reference", str(mri_path),
                 "-o", str(tmp_path / "never.nii.gz")]) == 2

    # `octreg apply` with a field in the run directory: u = one MRI voxel along MRI array axis 0, so the OCT content moves by
    # one voxel
    A_f = ref.affine.copy()
    A_f[:3, :3] *= 16
    io.save_field(np.broadcast_to((ref.affine[:3, 0])[:, None, None, None], (3, 4, 4, 4)), A_f, out / WARP)
    run = ["apply", "--run", str(out), "--moving", str(oct_path), "--reference", str(mri_path), "-o"]
    assert main(run + [str(tmp_path / "warped.nii.gz")]) == 0 and main(run + [str(tmp_path / "plain.nii.gz"), "--affine-only"]) == 0
    warped, plain = (nii(tmp_path / f) for f in ("warped.nii.gz", "plain.nii.gz"))
    np.testing.assert_allclose(plain, ref.get_fdata(), rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose(warped[:-1], plain[1:], rtol=1e-4, atol=0.05)
    capsys.readouterr()
    assert main(["apply", "--run", str(out), "--moving", str(mri_path), "--reference", str(oct_path), "-o",
                 str(tmp_path / "back2.nii.gz"), "--inverse"]) == 0 and "not inverted" in capsys.readouterr().out
    np.testing.assert_array_equal(nii(tmp_path / "back2.nii.gz"), back.get_fdata())


def test_register_with_the_deformation(data, deformed, tmp_path):
    """register(..., deform=True): the field file on the MRI base grid, qc_deform.png and the §5 read-outs; `octreg apply`
    reproduces oct_in_mri through the field and oct_in_mri_affine with --affine-only, and the two differ."""
    oct_path, mri_path, T_true, pts = data
    out, res = deformed
    assert all((out / f).is_file() for f in OUTPUTS + OUTPUTS_DEFORM)
    assert distance(np.loadtxt(out / "T_oct2mri.txt"), T_true, pts) < 0.4                   # T is the affine
    saved = json.loads((out / "result.json").read_text())
    d = saved["deform"]
    assert set(d) == {"passes"}
    assert [p["sigma_mm"] for p in d["passes"]] == list(FAST.df_sigmas_mm)
    assert all(set(p) == {"sigma_mm", "F_start", "F_end", "displacement_median_mm", "displacement_max_mm"} for p in d["passes"])
    assert all(p["F_end"] >= p["F_start"] - 1e-3 for p in d["passes"])
    u, A_u = io.load_field(out / WARP)
    warp = nib.load(str(out / WARP))
    assert warp.shape[3:] == (1, 3) and warp.header.get_intent()[0] == "vector"
    base, A_base = G.resample_iso(io.load_volume(mri_path), FAST.base_mm)                 # the field lives on the MRI base grid
    assert u.shape == (3,) + base.shape and np.allclose(A_u, A_base, atol=1e-6)
    top = np.linalg.norm(u, axis=0).max()
    assert 0 < top < 1.0 and d["passes"][-1]["displacement_max_mm"] < 1.0                 # a small field on an affine pair
    T, y = np.loadtxt(out / "T_oct2mri.txt"), G.apply_affine(T_true, pts)                  # the specimen in the MRI world
    moved = G.apply_affine(np.linalg.inv(T), y + np.stack([sample(y, (c, A_u)) for c in u], 1))      # OCT(T^-1 (y + u(y)))
    assert np.linalg.norm(moved - pts, axis=1).mean() < 0.4       # the specimen through T and the field

    run = ["apply", "--run", str(out), "--moving", str(oct_path), "--reference", str(mri_path), "-o"]
    assert main(run + [str(tmp_path / "warped.nii.gz")]) == 0 and main(run + [str(tmp_path / "plain.nii.gz"), "--affine-only"]) == 0
    both = [nii(out / f) for f in ("oct_in_mri.nii.gz", "oct_in_mri_affine.nii.gz")]
    np.testing.assert_allclose(nii(tmp_path / "warped.nii.gz"), both[0], rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose(nii(tmp_path / "plain.nii.gz"), both[1], rtol=1e-5, atol=1e-3)
    assert np.abs(both[0] - both[1]).max() > 1
    assert deform_label(saved["deform"]).startswith("§5: F " + ", ".join(f"{p['F_start']:.4f} -> {p['F_end']:.4f} at {p['sigma_mm']:g} mm"
                                                                          for p in d["passes"]) + "; ")
    fig = png(out / "qc_deform.png")                           # 3 rows of planes, the MRI foreground outline drawn
    assert round((fig.shape[0] / QC_DPI - QC_HEAD_IN) / QC_CELL_IN, 2) == 3
    assert (fig == list(bytes.fromhex(QC_COLOURS[0][1:]))).all(-1).sum() > 100


def test_user_masks(tmp_path):
    """--oct-mask (the true specimen on its own 0.16 mm grid) and --mri-mask replace the two masks of §1."""
    oct_path, mri_path, T_true, pts = pair(tmp_path, seed=1)
    A = io.load_volume(oct_path).affine.copy()
    A[:3, :3] *= 2
    mask = np.zeros((37, 35, 32), np.uint8)
    mask[tuple(np.rint(G.apply_affine(np.linalg.inv(A), pts)).astype(int).T)] = 1
    io.save_nifti(mask, A, tmp_path / "oct_mask.nii.gz")
    mri = nib.load(str(mri_path))
    io.save_nifti((np.asarray(mri.dataobj) > 350).astype(np.uint8), mri.affine, tmp_path / "mri_mask.nii.gz")
    out = tmp_path / "run"
    res = register(oct_path, mri_path, out, oct_mask=tmp_path / "oct_mask.nii.gz", mri_mask=tmp_path / "mri_mask.nii.gz",
                   params=FAST, device="cpu", deform=False)
    fg = res["foreground"]
    assert fg["oct"]["source"].endswith("oct_mask.nii.gz") and fg["mri"]["source"].endswith("mri_mask.nii.gz")
    assert res["pose"]["outline_used"]
    assert distance(np.loadtxt(out / "T_oct2mri.txt"), T_true, pts) < 0.4


def test_mask_in_another_frame_is_refused(data, tmp_path):
    """A mask must be in the frame of its image: an NPY OCT mask (array frame) for the NIfTI OCT (header frame) and a NIfTI
    MRI mask without sform or qform (array frame) for the MRI are refused before any output is written, ValueError in Python
    and exit code 2 on the command line."""
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
    """An NPY stack has only an array frame, taken as given with the given spacing. The stack here is written so that its array
    frame is the header frame it came from up to a translation (the header negates all three axes, so all three are flipped),
    and the same pose is found; `octreg apply` reproduces oct_in_mri from the stack."""
    oct_path, mri_path, T_true, pts = data
    img = nib.load(str(oct_path))
    np.save(tmp_path / "oct.npy", np.asarray(img.dataobj, np.float32).transpose(2, 1, 0)[::-1, ::-1, ::-1])   # numpy (z, y, x)
    A_arr = np.diag([-0.08, -0.08, -0.08, 1.0])                                                  # NIfTI voxel -> array world
    A_arr[:3, 3] = 0.08 * (np.array(img.shape) - 1)
    T_arr = T_true @ img.affine @ np.linalg.inv(A_arr)                                          # array world -> MRI world
    pts_arr = G.apply_affine(A_arr @ np.linalg.inv(img.affine), pts)
    out = tmp_path / "run"
    register(tmp_path / "oct.npy", mri_path, out, (80, 80, 80), params=FAST, device="cpu", deform=False)
    T = np.loadtxt(out / "T_oct2mri.txt")
    assert np.linalg.det(T_arr[:3, :3]) > 0 and np.linalg.det(T[:3, :3]) > 0
    assert distance(T, T_arr, pts_arr) < 0.4
    assert main(["apply", "--run", str(out), "--moving", str(tmp_path / "oct.npy"), "--reference", str(mri_path), "-o",
                 str(tmp_path / "again.nii.gz")]) == 0
    np.testing.assert_allclose(nii(tmp_path / "again.nii.gz"), nii(out / "oct_in_mri.nii.gz"), rtol=1e-5, atol=1e-3)


def test_apply_moved_npy_with_spacing(data, tmp_path):
    """`octreg apply` replays a run whose OCT is an NPY stack from another path: the spacing recorded in result.json belongs to
    the original path, so the run is refused until --oct-spacing-um gives it, and then the result is the same (through the
    field of the run)."""
    stack = np.asarray(nib.load(str(data[0])).dataobj, np.float32).transpose(2, 1, 0)       # numpy (z, y, x)
    np.save(tmp_path / "oct.npy", stack)
    (tmp_path / "moved").mkdir()
    np.save(tmp_path / "moved" / "oct.npy", stack)                                           # the same stack elsewhere
    out = tmp_path / "run"
    register(tmp_path / "oct.npy", data[1], out, (80, 80, 80), params=Params.from_dict(TINY), device="cpu")
    assert (out / WARP).is_file()
    assert main(["apply", "--run", str(out), "--moving", str(tmp_path / "oct.npy"), "--reference", str(data[1]), "-o",
                 str(tmp_path / "here.nii.gz")]) == 0
    np.testing.assert_allclose(nii(tmp_path / "here.nii.gz"), nii(out / "oct_in_mri.nii.gz"), rtol=1e-5, atol=1e-3)
    run = ["apply", "--run", str(out), "--moving", str(tmp_path / "moved" / "oct.npy"), "--reference", str(data[1]), "-o",
           str(tmp_path / "moved.nii.gz")]
    assert main(run) == 2                                      # an NPY without spacing: refused
    assert main(run + ["--oct-spacing-um", "80,80,80"]) == 0
    np.testing.assert_array_equal(nii(tmp_path / "moved.nii.gz"), nii(tmp_path / "here.nii.gz"))


def test_register_command(data, tmp_path, capsys):
    """`octreg register` with a params file: exit 0, every output with §5, a summary line, the params of the file; with
    --no-deform in the same directory the §5 files are gone."""
    (tmp_path / "params.json").write_text(json.dumps(TINY))
    out = tmp_path / "run"
    assert main(["register", str(data[0]), str(data[1]), "-o", str(out), "--device", "cpu", "--params",
                 str(tmp_path / "params.json")]) == 0
    assert all((out / f).is_file() for f in OUTPUTS + OUTPUTS_DEFORM)
    res = json.loads((out / "result.json").read_text())
    assert Params.from_dict(res["params"]) == Params.from_dict(TINY)
    assert f"-> {out}" in capsys.readouterr().out
    assert main(["register", str(data[0]), str(data[1]), "-o", str(out), "--device", "cpu", "--params",
                 str(tmp_path / "params.json"), "--no-deform"]) == 0
    assert json.loads((out / "result.json").read_text())["deform"] is None and not any((out / f).exists() for f in OUTPUTS_DEFORM)
    assert np.array_equal(nii(out / "oct_in_mri.nii.gz"), nii(out / "oct_in_mri_affine.nii.gz"))

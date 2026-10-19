"""params, io and geometry on small synthetic volumes."""
import dataclasses
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import tifffile
import torch
from scipy import ndimage
from scipy.spatial.transform import Rotation

from octreg import geometry as g
from octreg import io
from octreg.params import Params


def rot(seed):
    """A random rotation [3, 3], uniform on SO(3) and independent of the search set."""
    return Rotation.random(random_state=seed).as_matrix()


def oblique(spacing, origin, seed=7, mirror=True):
    """voxel -> world affine with rotated axes, det < 0 when mirror."""
    A = np.eye(4)
    A[:3, :3] = rot(seed) @ np.diag(spacing) @ np.diag([1.0, 1.0, -1.0 if mirror else 1.0])
    A[:3, 3] = origin
    return A


def corners(shape):
    return np.array([[i, j, k] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1) for k in (0, shape[2] - 1)], float)


def direct(data, A, k, shape_out, A_out):
    """Reference pooling: k^3 box means, then trilinear at the output voxel centres with 0 beyond the pooled centres."""
    n = np.array(data.shape) // k
    pooled = data[:n[0] * k[0], :n[1] * k[1], :n[2] * k[2]].reshape(n[0], k[0], n[1], k[1], n[2], k[2]).mean((1, 3, 5), dtype=np.float64)
    A_p = np.array(A, float)
    A_p[:3, 3] += A_p[:3, :3] @ ((k - 1) / 2.0)
    A_p[:3, :3] *= k
    c = g.apply_affine(np.linalg.inv(A_p) @ A_out, np.indices(shape_out).reshape(3, -1).T.astype(float))
    return ndimage.map_coordinates(pooled, c.T, order=1, mode="grid-constant", cval=0.0).reshape(shape_out), pooled.shape


def stack(vol):
    planes = list(io.iter_planes(vol))
    assert [k for k, _ in planes] == list(range(vol.shape[2])) and all(p.dtype == np.float32 for _, p in planes)
    return np.stack([p for _, p in planes], -1)


def test_params_defaults_and_dict():
    p = Params()
    assert (p.search_mm, p.base_mm, p.fine_mm, p.n_rot, p.topk, p.iters, p.lam) == (0.6, 0.15, 0.04, 8000, 24, 150, 2.0)
    assert (p.sigmas_mm, p.lr_mm, p.erode_mm) == ((0.6, 0.4, 0.3), 0.06, 0.8)
    v = dataclasses.replace(p, topk=12, lam=0.0, sigmas_mm=(0.5,), lr_mm=0.1, iters=50)
    assert Params.from_dict(json.loads(json.dumps(v.to_dict()))) == v                 # JSON lists back to tuples
    assert Params.from_dict({"lam": 1, "sigmas_mm": [1, 0.5]}) == dataclasses.replace(p, lam=1.0, sigmas_mm=(1.0, 0.5))
    with pytest.raises(ValueError, match="unknown keys"):
        Params.from_dict({"no_such_field": 0.8})
    with pytest.raises(ValueError, match="must be a JSON object"):
        Params.from_dict([0.5])
    for bad in ({"fine_mm": 0.2}, {"sigmas_mm": []}, {"sigmas_mm": [0.3, -0.1]}, {"erode_mm": -0.1}, {"iters": 0},
                {"sigmas_mm": 0.3}, {"lam": "2"}, {"iters": 10.5}, {"topk": True}, {"sigmas_mm": [[0.3]]}, {"lam": -1},
                {"lr_mm": 0}, {"texture_window_mm": -0.36}, {"valley_ratio": 0}, {"min_component": 1.0}, {"nms_mm": -3},
                {"outline_min_outside": 0}, json.loads('{"lam": NaN}'), json.loads('{"lam": Infinity}')):
        with pytest.raises(ValueError, match="^Params: "):
            Params.from_dict(bad)
    for bad in (lambda: dataclasses.replace(p, iters=10.5), lambda: Params(lam="2"), lambda: Params(lam=float("nan"))):
        with pytest.raises(ValueError, match="^Params: "):
            bad()


@pytest.mark.parametrize("suffix", [".nii", ".nii.gz"])
@pytest.mark.parametrize("form", ["sform", "qform"])
def test_oblique_det_negative_nifti_roundtrip(tmp_path, suffix, form):
    shape = (10, 12, 14)
    A = oblique((20.0, 25.0, 30.0), (-7300.0, 4100.0, 6900.0))                       # um
    assert np.linalg.det(A[:3, :3]) < 0
    data = np.random.default_rng(0).random(shape).astype(np.float32)
    img = nib.Nifti1Image(data, None)
    img.header.set_xyzt_units("micron")
    img.header.set_qform(A, code=1)
    img.header.set_sform(A if form == "sform" else None, code=1 if form == "sform" else 0)
    nib.save(img, str(tmp_path / f"o{suffix}"))
    v = io.load_volume(tmp_path / f"o{suffix}")
    assert v.frame == "header" and v.shape == shape and np.allclose(v.spacing_mm, (0.02, 0.025, 0.03), atol=1e-7)
    assert np.abs(corners(shape) @ (v.affine[:3, :3] - A[:3, :3] / 1000).T + v.affine[:3, 3] - A[:3, 3] / 1000).max() < 1e-6
    io.save_nifti(data, v.affine, tmp_path / f"s{suffix}")
    w = io.load_volume(tmp_path / f"s{suffix}")
    assert np.abs(corners(shape) @ (w.affine - v.affine)[:3, :3].T + (w.affine - v.affine)[:3, 3]).max() < 1e-6
    assert np.array_equal(stack(w), data)
    s = io.load_volume(tmp_path / f"o{suffix}", spacing_um=(45, 30, 25))              # Z,Y,X = along (k, j, i)
    assert np.allclose(s.spacing_mm, (0.025, 0.03, 0.045)) and np.allclose(s.affine[:3, 3], A[:3, 3] / 1000, atol=1e-6)
    assert np.allclose(s.affine[:3, :3] / s.spacing_mm, A[:3, :3] / (20.0, 25.0, 30.0), atol=1e-6)


@pytest.mark.parametrize("suffix", [".nii", ".nii.gz"])
def test_streamed_planes_equal_nibabel(tmp_path, suffix):
    rng = np.random.default_rng(1)
    raw = rng.integers(-3000, 3000, (9, 11, 13, 1)).astype(np.int16)                  # trailing singleton axis, scaled int16
    img = nib.Nifti1Image(raw, np.diag([0.08, 0.08, 0.08, 1.0]))
    img.header.set_slope_inter(0.5, 3.0)
    nib.save(img, str(tmp_path / f"i{suffix}"))
    f = rng.random((9, 11, 13)).astype(np.float32)
    f[1, 2, 3], f[4, 5, 6] = np.nan, np.inf
    nib.save(nib.Nifti1Image(f, None), str(tmp_path / f"f{suffix}"))                     # no sform or qform: array frame
    for name in ("i", "f"):
        v = io.load_volume(tmp_path / f"{name}{suffix}")
        ref = np.nan_to_num(np.asarray(nib.load(str(tmp_path / f"{name}{suffix}")).get_fdata(), np.float32), posinf=0.0)
        assert v.shape == (9, 11, 13) and np.array_equal(stack(v), ref.reshape(v.shape))
    assert io.load_volume(tmp_path / f"f{suffix}").frame == "array"
    with pytest.raises(ValueError, match="3-D"):
        nib.save(nib.Nifti1Image(np.zeros((4, 5, 6, 2), np.float32), np.eye(4)), str(tmp_path / f"t{suffix}"))
        io.load_volume(tmp_path / f"t{suffix}")


def test_tiff_npy_axes_spacing_and_refusals(tmp_path):
    a = (np.random.default_rng(2).random((6, 8, 10)) * 4000).astype(np.uint16)          # numpy (z, y, x)
    tifffile.imwrite(tmp_path / "o.ome.tif", a, ome=True, metadata={"axes": "ZYX", "PhysicalSizeX": 0.012, "PhysicalSizeXUnit": "mm",
                     "PhysicalSizeY": 13.0, "PhysicalSizeZ": 14.0})
    v = io.load_volume(tmp_path / "o.ome.tif")
    assert v.frame == "array" and v.shape == (10, 8, 6) and np.allclose(v.affine, np.diag([0.012, 0.013, 0.014, 1.0]))
    assert np.array_equal(stack(v), a.transpose(2, 1, 0))
    tifffile.imwrite(tmp_path / "c.tif", a, compression="zlib")                         # compressed, so read page by page
    with pytest.raises(ValueError, match="no voxel spacing"):
        io.load_volume(tmp_path / "c.tif")
    v = io.load_volume(tmp_path / "c.tif", spacing_um=(35, 31, 30))
    assert np.allclose(v.spacing_mm, (0.030, 0.031, 0.035)) and np.array_equal(stack(v), a.transpose(2, 1, 0))
    np.save(tmp_path / "n.npy", a)
    v = io.load_volume(tmp_path / "n.npy", spacing_um=(14, 12, 10))
    assert np.allclose(v.spacing_mm, (0.010, 0.012, 0.014)) and np.array_equal(stack(v), a.transpose(2, 1, 0))
    with pytest.raises(ValueError, match="no voxel spacing"):
        io.load_volume(tmp_path / "n.npy")
    np.save(tmp_path / "flat.npy", np.zeros((5, 6), np.float32))
    with pytest.raises(ValueError, match="3-D"):
        io.load_volume(tmp_path / "flat.npy", spacing_um=(1, 1, 1))
    with pytest.raises(ValueError, match="spacing_um"):
        io.load_volume(tmp_path / "n.npy", spacing_um=(12, 0, 12))
    (tmp_path / "x.mgz").write_bytes(b"0")
    for bad, msg in (("x.mgz", "unsupported"), ("missing.nii.gz", "no such file")):
        with pytest.raises(ValueError, match=msg):
            io.load_volume(tmp_path / bad)


def test_resample_iso_equals_direct_pooling(tmp_path):
    rng = np.random.default_rng(3)
    shape, sp = (64, 72, 80), (0.020, 0.025, 0.030)
    A = oblique(sp, (1.0, -2.0, 3.0))
    ijk = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1).astype(float)
    ramp = (ijk @ A[:3, :3].T + A[:3, 3]) @ np.array([0.7, -1.1, 0.4])
    noise = rng.random(shape).astype(np.float32)
    for name, data in (("ramp", ramp.astype(np.float32)), ("noise", noise)):
        io.save_nifti(data, A, tmp_path / f"{name}.nii.gz")
        vol = io.load_volume(tmp_path / f"{name}.nii.gz")
        data = stack(vol)
        for level in (0.15, 0.1, 0.06):
            out, A_out = g.resample_iso(vol, level)
            assert np.allclose(np.linalg.norm(A_out[:3, :3], axis=0), level)
            p2, A2 = g.pool_iso(data, vol.affine, level)
            assert np.array_equal(out, p2) and np.array_equal(A_out, A2)
            k = np.maximum(1, np.rint(np.round(level / np.array(sp), 9))).astype(int)
            ref, pooled_shape = direct(data, vol.affine, k, out.shape, A_out)
            assert out.shape == tuple(int(np.floor(np.round(pooled_shape[i] * sp[i] * k[i] / level, 9))) for i in range(3))
            np.testing.assert_allclose(out, ref, atol=2e-4 * max(1.0, np.abs(data).max()))
            if name == "ramp":                                                        # trilinear of a linear field is exact
                ijk_out = np.stack(np.meshgrid(*[np.arange(s) for s in out.shape], indexing="ij"), -1).astype(float)
                inside = np.all(ijk_out * level <= (np.array(pooled_shape) - 1) * vol.spacing_mm * k + 1e-9, -1)
                truth = (ijk_out @ A_out[:3, :3].T + A_out[:3, 3]) @ np.array([0.7, -1.1, 0.4])
                assert inside.mean() > 0.5 and np.abs(out - truth)[inside].max() < 1e-3
    io.save_nifti(noise, np.diag([0.02, -0.02, 0.02, 1.0]), tmp_path / "iso.nii.gz")    # float32 pixdim 0.0199999996
    out, A_out = g.resample_iso(io.load_volume(tmp_path / "iso.nii.gz"), 0.04)
    assert out.shape == (32, 36, 40) and np.allclose(out, noise.reshape(32, 2, 36, 2, 40, 2).mean((1, 3, 5)), atol=1e-6)
    mask = noise > 0.7
    io.save_nifti((mask * 255).astype(np.uint8), np.diag([0.02, -0.02, 0.02, 1.0]), tmp_path / "mask.nii.gz")
    frac, _ = g.resample_iso(io.load_volume(tmp_path / "mask.nii.gz"), 0.04, binary=True)       # indicator, not the values
    assert np.allclose(frac, mask.reshape(32, 2, 36, 2, 40, 2).mean((1, 3, 5)), atol=1e-6)
    sparse = np.zeros((90, 90, 66), np.float32)                         # 12 x 12 x 15.15 um, the steps are not exact in binary
    sparse[::7, ::5, ::4] = 5
    io.save_nifti(sparse, np.diag([0.012, 0.012, 0.01515, 1]), tmp_path / "sparse.nii.gz")
    out, _ = g.resample_iso(io.load_volume(tmp_path / "sparse.nii.gz"), 0.04)
    assert out[out > 0].min() > 1e-6                                    # partial weights only, no float residue near zeros
    # k = 8 and step 0.9375: 15 pooled planes give 16 output planes, the last one past the last pooled centre
    A = oblique((0.02, 0.02, 0.02), (0.0, 0.0, 0.0))
    ones, _ = g.pool_iso(np.ones((120, 24, 120), np.float32), A, 0.15)
    assert ones.shape == (16, 3, 16) and np.allclose(ones[:15, :, :15], 1) and np.allclose(ones[15, :, :15], 0.9375)
    assert np.allclose(ones[15, :, 15], 0.9375 ** 2)
    edge = rng.random((120, 24, 120)).astype(np.float32)
    out, A_out = g.pool_iso(edge, A, 0.15)
    assert np.allclose(out, direct(edge, A, np.array([8, 8, 8]), out.shape, A_out)[0], atol=1e-5)
    with pytest.raises(ValueError, match="fewer than 2"):
        g.pool_iso(noise[:4, :4, :4], np.diag([0.02, 0.02, 0.02, 1]), 0.6)


def test_sampling_primitives(monkeypatch):
    rng = np.random.default_rng(4)
    shape = (12, 14, 16)
    A = oblique((0.2, 0.3, 0.25), (-1.0, 2.0, 0.5))
    x = rng.random(shape).astype(np.float32)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rot(3) * 1.05, (0.3, -0.2, 0.1)
    A_dst = oblique((0.22, 0.22, 0.22), (0.0, 0.0, 0.0), seed=2, mirror=False)
    A_dst[:3, 3] = g.apply_affine(np.linalg.inv(T), g.apply_affine(A, (np.array(shape) - 1) / 2)) - A_dst[:3, :3] @ [7.0, 6.0, 8.0]
    out = g.resample_to(x, A, (15, 13, 17), A_dst, T)                                 # scipy map_coordinates
    ijk = np.stack(np.meshgrid(*[np.arange(s) for s in (15, 13, 17)], indexing="ij"), -1).reshape(-1, 3).astype(float)
    pts = torch.as_tensor(g.apply_affine(T, g.apply_affine(A_dst, ijk)))
    ref = g.sample_world(torch.as_tensor(x)[None], A, pts)[0].numpy().reshape(out.shape)  # torch grid_sample
    assert (out != 0).mean() > 0.3 and (out == 0).mean() > 0.05 and np.allclose(out, ref, atol=1e-5)
    monkeypatch.setattr(g, "CHUNK", 13 * 17 * 4)                                   # 15 planes in chunks of 4, 4, 4 and 3
    assert np.array_equal(g.resample_to(x, A, (15, 13, 17), A_dst, T), out)


def _nearest_deg(q, probes):
    """Geodesic angle in degrees from each probe quaternion [m, 4] to the nearest of q [n, 4]."""
    near = np.concatenate([np.abs(probes[i:i + 2000] @ q.T).max(1) for i in range(0, len(probes), 2000)])
    return np.degrees(2 * np.arccos(np.minimum(near, 1.0)))


def test_rotation_set():
    """The rotation set is proper, deterministic, near-uniform on SO(3) and has no two rotations within 3 degrees."""
    assert [len(g.rotations(n)) for n in (1, 2, 100)] == [1, 2, 100]
    R = g.rotations(8000)
    assert R.shape == (8000, 3, 3) and R.dtype == np.float64
    assert np.allclose(R @ R.transpose(0, 2, 1), np.eye(3), atol=1e-12) and np.allclose(np.linalg.det(R), 1.0, atol=1e-12)
    assert np.array_equal(g.rotations(8000), R)
    q = Rotation.from_matrix(R).as_quat()
    # covering radius over 20000 uniform probes, about 16 degrees for 8000 iid uniform rotations
    assert _nearest_deg(q, Rotation.random(20000, random_state=0).as_quat()).max() < 12.0
    angle = Rotation.from_matrix(R).magnitude()
    # the share within angle t of the identity is (t - sin t) / pi under the Haar measure
    for t in np.radians([30, 60, 90, 120, 150]):
        assert abs((angle < t).mean() - (t - np.sin(t)) / np.pi) < 0.005
    # second largest |q_i . q_j| of each row, the largest being the rotation itself
    sep = np.concatenate([np.partition(np.abs(q[i:i + 2000] @ q.T), -2, 1)[:, -2] for i in range(0, len(q), 2000)])
    assert np.degrees(2 * np.arccos(np.minimum(sep, 1.0))).min() > 3.0


def test_transform_txt_lta_itk_roundtrip(tmp_path):
    shape_o, shape_m = (40, 30, 20), (34, 48, 50)
    A_o, A_m = oblique((0.02, 0.02, 0.02), (5.0, -3.0, 2.0)), oblique((0.08, 0.08, 0.08), (-10.0, 12.0, -7.0), seed=3, mirror=False)
    io.save_nifti(np.zeros(shape_o, np.float32), A_o, tmp_path / "oct.nii.gz")
    io.save_nifti(np.zeros(shape_m, np.float32), A_m, tmp_path / "mri.nii.gz")
    src, dst = io.load_volume(tmp_path / "oct.nii.gz"), io.load_volume(tmp_path / "mri.nii.gz")
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rot(5) @ np.diag([1.04, 0.97, -1.01]), (3.5, -8.25, 11.0)
    io.write_transform_txt(T, tmp_path / "T.txt")
    assert np.array_equal(np.loadtxt(tmp_path / "T.txt"), T)
    io.write_lta(T, src, dst, tmp_path / "x.lta")
    assert b"\r" not in (tmp_path / "x.lta").read_bytes() + (tmp_path / "T.txt").read_bytes()
    text = (tmp_path / "x.lta").read_text()
    rows = text.split("1 4 4\n")[1].splitlines()[:4]
    assert "LINEAR_RAS_TO_RAS" in text and np.allclose(np.array([r.split() for r in rows], float), T, rtol=1e-14, atol=1e-14)
    for block, vol in ((text.split("src volume info")[1].split("dst volume info")[0], src), (text.split("dst volume info")[1], dst)):
        info = {k.strip(): val.split("#")[0].split() for k, val in (ln.split("=", 1) for ln in block.splitlines() if "=" in ln)}
        num = lambda key: np.array(info[key], float)
        hdr = nib.MGHImage(np.zeros(vol.shape, np.float32), vol.affine).header               # FreeSurfer's own geometry
        assert [int(s) for s in info["volume"]] == list(vol.shape) and info["filename"] == [str(Path(vol.path).resolve())]
        assert np.allclose(num("cras"), hdr["Pxyz_c"], atol=1e-5) and np.allclose(num("voxelsize"), hdr["delta"], atol=1e-6)
        assert np.allclose(np.stack([num("xras"), num("yras"), num("zras")]), hdr["Mdc"], atol=1e-6)
    io.write_json({"T": T, "p": Params().to_dict(), "nan": float("nan"), "f": np.float32(1.5)}, tmp_path / "r.json")
    back = json.loads((tmp_path / "r.json").read_text())
    assert np.array_equal(back["T"], T) and back["p"]["search_mm"] == 0.6 and back["nan"] is None and back["f"] == 1.5
    sitk = pytest.importorskip("SimpleITK")                   # ITK file: dst voxel -> dst physical -> transform -> src voxel
    np.save(tmp_path / "a.npy", np.zeros((14, 12, 10), np.float32))                  # numpy (z, y, x): volume (10, 12, 14)
    arr = io.load_volume(tmp_path / "a.npy", spacing_um=(40, 30, 20))
    bare = nib.Nifti1Image(np.zeros((10, 12, 14), np.float32), None)                   # no sform or qform: array frame
    bare.header.set_zooms((0.04, 0.03, 0.02))
    nib.save(bare, str(tmp_path / "bare.nii.gz"))
    nii = io.load_volume(tmp_path / "bare.nii.gz")
    assert nii.frame == "array"
    for v, f in ((arr, "arr_copy.nii.gz"), (nii, "bare_copy.nii.gz")):              # a NIfTI copy with the world as sform
        io.save_nifti(np.zeros(v.shape, np.float32), v.affine, tmp_path / f)
    images = {id(v): sitk.ReadImage(str(tmp_path / f)) for v, f in ((src, "oct.nii.gz"), (dst, "mri.nii.gz"),
                                                                     (arr, "arr_copy.nii.gz"), (nii, "bare_copy.nii.gz"))}
    q = np.array([3.0, 4.0, 5.0])
    io.write_itk(T, tmp_path / "x_itk.txt")
    assert b"\r" not in (tmp_path / "x_itk.txt").read_bytes()
    tr = sitk.ReadTransform(str(tmp_path / "x_itk.txt"))
    for s, d in ((src, dst), (arr, dst), (src, arr), (nii, dst), (src, nii)):
        p = tr.TransformPoint(images[id(d)].TransformContinuousIndexToPhysicalPoint(tuple(q)))
        want = g.apply_affine(np.linalg.inv(s.affine) @ np.linalg.inv(T) @ d.affine, q)
        assert np.allclose(images[id(s)].TransformPhysicalPointToContinuousIndex(p), want, atol=1e-4)

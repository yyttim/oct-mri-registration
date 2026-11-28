"""tilestack: the working grid from the voxel of the slab files, the volume at that voxel, its world and the warps."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _block(tmp_path, um, shape=(64, 48, 40), empty_cols=0):
    for s in (1, 2):
        M = np.random.default_rng(s).random(shape).astype(np.float32)
        M[:shape[0] // 8] *= 0.01                        # water above the cut
        if s == 2 and empty_cols:
            M[:, :, -empty_cols:] = 0                     # no data at the edge of slab 2
        with h5py.File(tmp_path / f"s{s:03d}.mat", "w") as f:
            f["M"] = M
            f["res_um"] = np.asarray(um, float).reshape(3, 1)
    window = shape[0] * um[2] / 1000
    block = dict(slab_file=str(tmp_path / "s{:03d}.mat"), work=str(tmp_path / "work"), slabs=[1, 2], neighbours=1,
                 margin_mm=0.06, deep_mm=0.5 * window, deep_a_mm=0.6 * window)
    (tmp_path / "block.json").write_text(json.dumps(block))
    return tmp_path / "block.json"


def _python(block, code, check=True):
    r = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, TILESTACK_BLOCK=str(block)), cwd=ROOT,
                       capture_output=True, text=True)
    if not check:
        return r
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout)


COMPOSE = """
import json, numpy as np
from tilestack import compose, data, export, solve, tiles
compose.dev = "cpu"
W = data.WORK
tiles2, field = {tiles2}, {field}
if tiles2:                                               # 2 x 2 tiles in working pixels, moving apart at the seams
    lay = tiles.Layout(0.0, 0.0, data.NX / 2, data.NY / 2, 0.0, 0.0, data.NX / 2, data.NY / 2, 2, 2, exists={{1: [0, 1, 2, 3], 2: [0, 1, 2, 3]}})
    e = np.array([[-0.7, -0.6, 0.0], [0.8, -0.5, 0.1], [-0.6, 0.7, -0.1], [0.7, 0.9, 0.0]])
else:
    lay = tiles.Layout(0.0, 0.0, float(data.NX), float(data.NY), 0.0, 0.0, float(data.NX), float(data.NY), 1, 1, exists={{1: [0], 2: [0]}})
    e = np.zeros((1, 3))
aff = np.zeros((2, 3, 3)); aff[:, :, 0] = [[0.3, -0.6, 0.25], [-0.3, 0.6, -0.25]]      # working pixels and planes
sol = dict(slabs=[1, 2], aff=aff, tiles=np.stack([e, e]), field=None, G=None)
if field:                                                # a smooth field of up to 1 working pixel or plane, 4 px spacing
    sol.update(G=4, field=np.random.default_rng(0).uniform(-1, 1, (2, 3, solve.Spline(4).n)))
vol, z0, src = compose.compose(sol, lay, W + "/v.npy")
vol = np.asarray(vol)
"""


def test_working_grid_from_res_um(tmp_path):
    out = _python(_block(tmp_path, [10, 10, 5]), """
import json
from tilestack import data
M, G = data.load(1), data.load_grid(1)
ref = M.reshape(16, 4, 24, 2, 20, 2).mean((1, 3, 5))
print(json.dumps(dict(vox=data.VOX, factor=data.FACTOR, grid=[data.NP, data.NY, data.NX], err=float(abs(G - ref).max()))))
""")
    assert out["vox"] == pytest.approx(0.02) and out["factor"] == [4, 2, 2] and out["grid"] == [16, 24, 20]
    assert out["err"] < 1e-6


@pytest.mark.parametrize("um", [[5, 5, 3], [10, 20, 10]])
def test_voxel_without_working_grid_is_refused(tmp_path, um):
    r = _python(_block(tmp_path, um), "from tilestack import data", check=False)
    assert r.returncode != 0 and "has no working grid" in r.stderr


@pytest.mark.parametrize("um,name", [([20, 20, 20], "registered_20um"), ([10, 10, 10], "registered_10um"),
                                     ([5, 5, 5], "registered_5um"), ([5, 5, 2.5], "registered_5x5x2.5um")])
def test_volume_voxel_and_slab_voxel_share_the_world(tmp_path, um, name):
    """With no placement, output voxel (j, l, i) holds slab voxel (j - PAD fx, l - PAD fy, i + z0 fz): both at one world point.
    The volume is named by the voxel of the slab files."""
    out = _python(_block(tmp_path, um), """
import json
from tilestack import data, export
print(json.dumps(dict(aff=export._affine(-7).tolist(), slab=export.slab_affine().tolist(), f=data.FACTOR, pad=data.PAD,
                      name=export.name())))
""")
    aff, slab, (fz, fy, fx), pad = np.array(out["aff"]), np.array(out["slab"]), out["f"], out["pad"]
    assert out["name"] == name and np.linalg.det(aff[:3, :3]) < 0 and aff[2, 2] < 0
    for j, l, i in ((0, 0, 0), (70, 33, 9), (5, 140, 31)):
        assert np.allclose(aff @ [j, l, i, 1], slab @ [j - pad * fx, l - pad * fy, i - 7 * fz, 1])


def test_layout_in_file_pixels(tmp_path):
    """--layout in pixels of 10 um files gives the lattice of the same tiles on the 20 um working grid."""
    out = _python(_block(tmp_path, [10, 10, 10]), """
import json, numpy as np
from tilestack import tiles
spec = dict(X0=3.0, Y0=-2.0, PX=21.0, PY=25.0, SX=1.5, SY=-2.0, W=22.0, H=26.0, rows=2, cols=2)
lay = tiles.layout(spec)
xn, yn = np.random.default_rng(0).uniform(0, 47, (2, 500))
a = np.array(tiles.Layout(**spec).lattice(xn, yn)); b = np.array(lay.lattice((xn - 0.5) / 2, (yn - 0.5) / 2))
print(json.dumps(dict(err=float(np.abs(a - b).max()))))
""")
    assert out["err"] < 1e-9


def test_volume_takes_the_slab_point_the_placement_puts_there(tmp_path):
    """With a smooth field and moved tiles, compose takes for every output pixel the slab point and tile that the solve's
    forward map puts there: exactly in depth, in-plane within the sampling of the field on a 16 px lattice."""
    out = _python(_block(tmp_path, [20, 20, 20], shape=(64, 70, 40)), """
import json, numpy as np
from tilestack import compose, data, solve, tiles
compose.dev = "cpu"
lay = tiles.Layout(0.0, 0.0, data.NX / 2, data.NY / 2, 0.0, 0.0, data.NX / 2, data.NY / 2, 2, 2, exists={1: [0, 1, 2, 3]})
e = np.array([[-0.7, -0.6, 0.0], [0.8, -0.5, 0.1], [-0.6, 0.7, -0.1], [0.7, 0.9, 0.0]])
aff = np.zeros((1, 3, 3)); aff[0, :, 0] = [0.3, -0.6, 0.25]
sol = dict(slabs=[1], aff=aff, tiles=e[None], G=64, field=np.random.default_rng(0).uniform(-1, 1, (1, 3, solve.Spline(64).n)))
ny, nx = data.NY + 2 * data.PAD, data.NX + 2 * data.PAD
y, x, cell, zo = (v.numpy() for v in compose.inverse_map(sol, lay, 1, ny, nx, 1.0))
Y, X = np.mgrid[0:ny, 0:nx] - data.PAD
k = cell >= 0
u, v, w = solve.displacement(sol, lay, 1, x[k], y[k])
print(json.dumps(dict(n=int(k.sum()), tile=bool((lay.cell(x[k], y[k], 1) == cell[k]).all()),
                      inplane=float(np.hypot(x[k] + u - X[k], y[k] + v - Y[k]).max()), depth=float(np.abs(w - zo[k]).max()))))
""")
    assert out["n"] > 1000 and out["tile"] and out["inplane"] < 0.1 and out["depth"] < 1e-5


@pytest.mark.parametrize("um,factor", [([5, 5, 5], [4, 4, 4]), ([5, 5, 2.5], [8, 4, 4])])
def test_volume_samples_the_files_at_the_placed_points(tmp_path, um, factor):
    """compose at factor 4, or 8 in depth (files of 5 um or 5 x 5 x 2.5 um, sizes not multiples of the factors), writes
    the trilinear samples of the newest slab, nothing of slab 2 where its file holds no data, and nothing within the
    margin under a cut."""
    out = _python(_block(tmp_path, um, shape=(90, 59, 49), empty_cols=12), COMPOSE.format(tiles2=False, field=False) + """
from scipy.ndimage import map_coordinates
(fz, fy, fx), err, n, under = data.FACTOR, 0.0, 0, 1e9
for k, s in enumerate((1, 2)):
    jz, jy, jx = np.nonzero((src == s) | (src == s + 1000))
    rz = z0 + (jz + 0.5) / fz - 0.5 - aff[k, 2, 0]
    cut = map_coordinates(data.geometry(s)[0].astype(float), [(jy + 0.5) / fy - 0.5 - data.PAD - aff[k, 1, 0],
                          (jx + 0.5) / fx - 0.5 - data.PAD - aff[k, 0, 0]], order=1, mode="nearest")
    under = min(under, float((rz - cut).min()))
    iz, iy, ix = np.nonzero(src == s)
    w = [z0 + (iz + 0.5) / fz - 0.5 - aff[k, 2, 0], (iy + 0.5) / fy - 0.5 - data.PAD - aff[k, 1, 0], (ix + 0.5) / fx - 0.5 - data.PAD - aff[k, 0, 0]]
    c = np.array([(v + 0.5) * f - 0.5 for v, f in zip(w, data.FACTOR)])        # the points in the file's voxels
    inside = np.all((c >= 0) & (c <= np.array(data.SHAPE)[:, None] - 1), 0)
    err = max(err, float(np.abs(vol[iz[inside], iy[inside], ix[inside]] - map_coordinates(data.load(s), c[:, inside], order=1)).max()))
    n += int(inside.sum())
    if s == 2:
        last = float(c[2].max())
print(json.dumps(dict(err=err, n=n, factor=data.FACTOR, last=last, nx=data.SHAPE[2], under=under, margin=data.MARGIN,
                      plane=vol.shape[1:], pad=data.PAD)))
""")
    assert out["factor"] == factor and out["n"] > 500 and out["err"] < 1e-5
    assert out["plane"] == [59 + 2 * out["pad"] * factor[1], 49 + 2 * out["pad"] * factor[2]]
    assert out["last"] <= out["nx"] - 13 + 0.01
    assert out["under"] >= out["margin"] - 1e-4                  # no voxel closer to its slab's cut than the margin


@pytest.mark.parametrize("um,tiles2,field", [([20, 20, 20], False, False), ([5, 5, 5], False, False), ([20, 20, 20], True, False),
                                             ([5, 5, 5], True, False), ([20, 20, 20], True, True), ([5, 5, 2.5], True, True)])
def test_warps_resample_the_slabs_as_composed(tmp_path, um, tiles2, field):
    """SimpleITK, given a slab in the world of export.slab_affine and its warp, resamples it as compose placed it, also
    with a smooth field and at 5 x 5 x 2.5 um; between moved tiles the warp holds the nearest placement, never no
    displacement."""
    pytest.importorskip("SimpleITK")
    out = _python(_block(tmp_path, um, shape=(90, 59, 49)), COMPOSE.format(tiles2=tiles2, field=field) + """
import nibabel as nib, SimpleITK as sitk
export.save_volume(vol, z0, W + "/reg.nii", src)
export.save_warps(sol, lay, W, z0, vol.shape[0])
ref, err, n, low = sitk.ReadImage(W + "/reg.nii"), 0.0, 0, 1e9
for s in (1, 2):
    nib.save(nib.Nifti1Image(np.transpose(data.load(s), (2, 1, 0)), export.slab_affine()), W + f"/slab{s}.nii")
    warp = nib.load(W + f"/warps/slab_{s:03d}.nii.gz").get_fdata()
    low = min(low, float(np.hypot(warp[..., 0], warp[..., 1]).min()))
    t = sitk.DisplacementFieldTransform(sitk.ReadImage(W + f"/warps/slab_{s:03d}.nii.gz", sitk.sitkVectorFloat64))
    res = sitk.GetArrayFromImage(sitk.Resample(sitk.ReadImage(W + f"/slab{s}.nii"), ref, t, sitk.sitkLinear, 0.0))
    m = (src == s) & (res != 0)
    err, n = max(err, float(np.abs(res[m] - vol[m]).max(initial=0.0))), n + int(m.sum())
print(json.dumps(dict(err=err, n=n, low=low, vox=data.VOX)))
""")
    assert out["n"] > 500 and out["err"] < 1e-5
    if tiles2 and not field:
        assert out["low"] > 0.2 * out["vox"]                 # every in-plane displacement at least 0.2 working voxels

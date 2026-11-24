"""tilestack.match: the depth offset plane of a pair of slabs from its coarse blocks."""
import json

import numpy as np
import pytest

XC, YC = 816, 1020.5        # data.XC, data.YC of the block below, the origin of the plane


@pytest.fixture(scope="module")
def plane(tmp_path_factory):
    """match._plane for slabs of 2040 x 1632 pixels with a 0.02 mm voxel (the package reads the block when imported)."""
    d = tmp_path_factory.mktemp("block")
    (d / "shape.json").write_text(json.dumps([64, 2040, 1632]))
    (d / "block.json").write_text(json.dumps(dict(slab_file="", work=str(d), slabs=[1, 2], voxel_mm=0.02, neighbours=1,
                                                  margin_mm=0.06, deep_mm=0.86, deep_a_mm=1.12)))
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TILESTACK_BLOCK", str(d / "block.json"))
        from tilestack import match
        yield match._plane


def at(c, x, y):
    return c[0] + c[1] * (x - XC) / 1000 + c[2] * (y - YC) / 1000


def test_one_false_block_beside_the_others_does_not_tilt_the_plane(plane):
    x = np.array([160, 256, 352, 160, 256, 352, 160, 256, 640.0])
    y = np.array([256, 256, 256, 352, 352, 352, 448, 448, 448.0])
    v = np.array([20, 20.1, 19.9, 20.05, 20, 19.95, 20.1, 20, 45.0])
    c, n = plane(x, y, v)
    assert n == 8 and np.abs(at(c, x[:8], y[:8]) - v[:8]).max() < 0.5


def test_one_false_block_far_away_does_not_tilt_the_plane_of_a_small_patch(plane):
    x = np.array([256, 352, 256, 352, 256, 352, 256, 352, 304.0])
    y = np.array([352, 352, 448, 448, 352, 352, 448, 448, 1408.0])
    v = np.r_[np.full(8, 14.0), 4]
    assert np.abs(v - (14 - 10 * (y - 400) / 1000)).max() < 1         # a tilted plane lies within the tolerance of all nine
    c, n = plane(x, y, v)
    assert n == 8 and np.abs(at(c, x[:8], y[:8]) - 14).max() < 0.5


def test_blocks_at_the_two_ends_of_a_tilted_plane_all_agree(plane):
    rng = np.random.default_rng(1)
    x = np.r_[rng.uniform(64, 256, 8), rng.uniform(1024, 1216, 8)]
    y = np.r_[rng.uniform(160, 256, 8), rng.uniform(1792, 1888, 8)]
    true = (12.0, -2.0, -2.0)
    v = at(true, x, y) + rng.normal(0, 0.1, 16)
    assert np.abs(v[:8, None] - v[None, 8:]).min() > 4          # the two ends lie more than twice the tolerance apart
    c, n = plane(x, y, v)
    assert n == 16 and np.abs(at(c, x, y) - at(true, x, y)).max() < 0.5


def test_blocks_on_one_line_give_no_tilt_across_it_also_with_a_false_block_beside_it(plane):
    x, y, v = 200 + 96.0 * np.arange(8), np.full(8, 160.0), np.full(8, 33.0)
    for xf, yf, vf in ((x, y, v), (np.r_[x, 584], np.r_[y, 1408], np.r_[v, 12.6])):
        c, n = plane(xf, yf, vf)
        assert n == 8 and abs(at(c, 600, 660) - 33) < 1e-4

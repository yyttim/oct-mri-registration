"""tilestack.solve: the placement of the slabs and their mosaic tiles from matches."""
import json

import numpy as np
import pytest


@pytest.fixture(scope="module")
def ts(tmp_path_factory):
    """tilestack.solve and tilestack.tiles for slabs of 2040 x 1632 pixels with a 0.02 mm voxel (the package reads the
    block when imported)."""
    d = tmp_path_factory.mktemp("block")
    (d / "shape.json").write_text(json.dumps([64, 2040, 1632]))
    (d / "block.json").write_text(json.dumps(dict(slab_file="", work=str(d), slabs=[1, 2], voxel_mm=0.02, neighbours=1,
                                                  margin_mm=0.06, deep_mm=0.86, deep_a_mm=1.12)))
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TILESTACK_BLOCK", str(d / "block.json"))
        from tilestack import solve, tiles
        yield solve, tiles


def test_a_slab_step_a_tile_step_and_a_tile_without_matches(ts):
    """Slab 2 lies (3, -2, 200) from slab 1, and its tile 4 lies 4 pixels further along x. Tile 8 of slab 2 has no
    match and no present neighbour."""
    solve, tiles = ts
    lay = tiles.Layout(0, 0, 544, 680, 0, 0, 544, 680, 3, 3, exists={1: list(range(9)), 2: [0, 1, 3, 4, 8]})
    y, x = (v.ravel().astype(float) for v in np.mgrid[40:1360:80, 40:1088:80])     # in tiles 0, 1, 3 and 4 of slab 2
    step = np.array([3.0, -2.0, 200.0])
    tile4 = 4.0 * (lay.cell(x, y, 2) == 4)
    m = [dict(s=np.array(1), t=np.array(2), xB=x, yB=y, zB=np.full(len(x), 10.0), xA=x + step[0] + tile4,
              yA=y + step[1], zA=np.full(len(x), 10.0 + step[2]))]
    sol = solve.solve(m, lay, [1, 2])
    moved = (np.array(solve.displacement(sol, lay, 2, x, y))
             - np.array(solve.displacement(sol, lay, 1, x + step[0] + tile4, y + step[1])))
    off = np.abs(moved - np.array([step[0] + tile4, step[1] + 0 * x, step[2] + 0 * x]))
    assert off.max() < 0.5 and np.median(off) < 0.02       # the field takes part of tile 4's step at its border
    assert np.abs(sol["shift"].sum(0)).max() < 1e-6
    assert np.abs(sol["tiles"][1][[0, 1, 3]]).max() < 0.05  # the median of slab 2's tile translations is zero
    assert np.abs(sol["tiles"][1][8]).max() < 0.01           # tile 8 moves with its slab


def test_a_tile_without_matches_moves_with_its_neighbours(ts):
    """Slab 2 lies (3, -2, 200) from slab 1, and its tiles 2, 5 and 8, a column of its mosaic, lie 4 pixels further
    along x. Tile 5 has no match."""
    solve, tiles = ts
    lay = tiles.Layout(0, 0, 544, 680, 0, 0, 544, 680, 3, 3, exists={1: list(range(9)), 2: list(range(9))})
    y, x = (v.ravel().astype(float) for v in np.mgrid[40:2040:80, 40:1632:80])
    out5 = lay.cell(x, y, 2) != 5
    x, y = x[out5], y[out5]
    col = 4.0 * (lay.cell(x, y, 2) % 3 == 2)
    m = [dict(s=np.array(1), t=np.array(2), xB=x, yB=y, zB=np.full(len(x), 10.0), xA=x + 3 + col, yA=y - 2,
              zA=np.full(len(x), 210.0))]
    e = solve.solve(m, lay, [1, 2])["tiles"]
    assert e[1][2, 0] - e[0][2, 0] > 2                              # the tiles take most of the column's step
    assert np.abs(e[:, 5] - e[:, [2, 8]].mean(1)).max() < 0.2      # in both slabs tile 5 moves with tiles 2 and 8

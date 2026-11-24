"""tilestack.compose: the tile that every pixel of a slab takes in the registered volume."""
import json

import numpy as np
import pytest


@pytest.fixture(scope="module")
def tc(tmp_path_factory):
    """tilestack.compose, solve and tiles on the CPU for slabs of 2040 x 1632 pixels with a 0.02 mm voxel (the package
    reads the block when imported)."""
    d = tmp_path_factory.mktemp("block")
    (d / "shape.json").write_text(json.dumps([64, 2040, 1632]))
    (d / "block.json").write_text(json.dumps(dict(slab_file="", work=str(d), slabs=[1, 2], voxel_mm=0.02, neighbours=1,
                                                  margin_mm=0.06, deep_mm=0.86, deep_a_mm=1.12)))
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TILESTACK_BLOCK", str(d / "block.json"))
        import tilestack
        mp.setattr(tilestack, "dev", "cpu")
        from tilestack import compose, data, solve, tiles
        mp.setattr(compose, "dev", "cpu")
        yield compose, data, solve, tiles


def unmoved(solve, n):
    """Two slabs of n tiles in place."""
    spl = solve.Spline(solve.FIELD)
    return dict(slabs=[1, 2], shift=np.zeros((2, 3)), tiles=np.zeros((2, n, 3)), field=np.zeros((2, 3, spl.n)),
                G=solve.FIELD)


def test_the_cell_of_a_tile_a_slab_lacks_and_the_field_beyond_the_mosaic_take_the_tile_of_the_solve(tc):
    """Slab 2 lacks rows 0 and 1 of a 4 x 4 mosaic of 100 pixel tiles, so the cells of row 0 go to tiles two rows away,
    and its field reaches two cells beyond the mosaic on every side."""
    compose, data, solve, tiles = tc
    lay = tiles.Layout(200, 200, 100, 100, 0, 0, 100, 100, 4, 4, exists={1: list(range(16)), 2: list(range(8, 16))})
    P = data.PAD
    y, x = (v.astype(float) for v in np.mgrid[0:800, 0:800])
    cell = compose.inverse_map(unmoved(solve, 16), lay, 2, 800 + 2 * P, 800 + 2 * P, 1)[2].numpy()[P:P + 800, P:P + 800]
    assert (cell == lay.cell(x, y, 2)).all()


def test_where_moved_cells_overlap_a_pixel_takes_the_tile_it_lies_farthest_inside(tc):
    """Slab 2 lacks the centre tile of a 3 x 3 mosaic of 100 pixel tiles, whose cell goes to tile 1, and its tile 7 lies
    10 pixels up, into that cell."""
    compose, data, solve, tiles = tc
    lay = tiles.Layout(0, 0, 100, 100, 0, 0, 100, 100, 3, 3, exists={1: list(range(9)), 2: [0, 1, 2, 3, 5, 6, 7, 8]})
    sol = unmoved(solve, 9)
    sol["tiles"][1, 7, 1] = -10.0
    P = data.PAD
    cell = compose.inverse_map(sol, lay, 2, 300 + 2 * P, 300 + 2 * P, 1)[2].numpy()[P:P + 300, P:P + 300]
    assert (cell[180:195, 130:170] == 1).all() and (cell[196:200, 130:170] == 7).all()


def test_beyond_the_mosaic_a_pixel_where_moved_cells_overlap_takes_the_tile_it_lies_farthest_inside(tc):
    """Tiles 12 and 13, the first two of the last row of a 4 x 4 mosaic of 100 pixel tiles, lie 10 pixels towards each
    other, so that their moved cells also overlap beyond the mosaic."""
    compose, data, solve, tiles = tc
    lay = tiles.Layout(200, 200, 100, 100, 0, 0, 100, 100, 4, 4, exists={1: list(range(16)), 2: list(range(16))})
    sol = unmoved(solve, 16)
    sol["tiles"][1, 12, 0], sol["tiles"][1, 13, 0] = 10.0, -10.0
    P = data.PAD
    cell = compose.inverse_map(sol, lay, 2, 800 + 2 * P, 800 + 2 * P, 1)[2].numpy()[P:P + 800, P:P + 800]
    assert (cell[600:800, 291:300] == 12).all() and (cell[600:800, 301:310] == 13).all()


def test_at_an_inner_corner_a_pixel_takes_the_tile_it_lies_farthest_inside_in_straight_lines(tc):
    """Slab 2 lacks tiles 1 and 3 of a 3 x 3 mosaic of 100 pixel tiles, so tile 0 holds the three cells around a corner
    of the cell of tile 4, and tile 4 lies 20 pixels up and left, into them."""
    compose, data, solve, tiles = tc
    lay = tiles.Layout(0, 0, 100, 100, 0, 0, 100, 100, 3, 3, exists={1: list(range(9)), 2: [0, 2, 4, 5, 6, 7, 8]})
    sol = unmoved(solve, 9)
    sol["tiles"][1, 4, :2] = -20.0
    P = data.PAD
    cell = compose.inverse_map(sol, lay, 2, 300 + 2 * P, 300 + 2 * P, 1)[2].numpy()[P:P + 300, P:P + 300]
    y, x = np.mgrid[80:100, 80:100]
    in0, in4 = np.hypot(100 - x, 100 - y), np.minimum(x - 80, y - 80)      # how far inside the cells of tiles 0 and 4
    assert (cell[80:100, 80:100] == np.where(in4 > in0, 4, 0))[np.abs(in4 - in0) > 0.1].all()

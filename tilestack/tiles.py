"""Mosaic tile layout of the slabs (docs/TILESTACK.md, Usage).

Each slab is stitched from W x H pixel tiles whose top-left corners lie on a slightly rotated lattice,
    x = X0 + j * PX + i * SX,   y = Y0 + i * PY + j * SY      (row i along y, column j along x, pixels),
the same in every slab. A pixel belongs to the tile whose centre is nearest in lattice coordinates.
"""
import json
import os

import numpy as np

from . import data


def _runs(v):
    """Values, lengths and starts of the runs of constant value in a 1-D integer array longer than a tenth of the
    longest. The short ones, at the ends, come from the neighbouring tile columns or rows of the rotated lattice."""
    cut = np.r_[0, np.nonzero(np.diff(v))[0] + 1, len(v)]
    val, n, start = v[cut[:-1]], np.diff(cut), cut[:-1]
    keep = n > 0.1 * n.max()
    return val[keep], n[keep], start[keep]


class Layout:
    def __init__(self, X0, Y0, PX, PY, SX, SY, W, H, rows, cols, exists=None):
        self.X0, self.Y0, self.PX, self.PY, self.SX, self.SY = X0, Y0, PX, PY, SX, SY
        self.W, self.H, self.rows, self.cols = W, H, rows, cols
        self.n = rows * cols
        self.exists = exists or {}            # slab -> list of tile ids present in that slab's mosaic
        self._A = np.array([[PX, SX], [SY, PY]])
        self._cellmap = {}

    def corner(self, i, j):
        return self.X0 + j * self.PX + i * self.SX, self.Y0 + i * self.PY + j * self.SY

    def lattice(self, x, y):
        """Continuous lattice coordinates (i, j) of points, tile centres at integers."""
        cx = np.asarray(x, float) - self.X0 - self.W / 2
        cy = np.asarray(y, float) - self.Y0 - self.H / 2
        ji = np.linalg.solve(self._A, np.vstack([np.ravel(cx), np.ravel(cy)]))
        return ji[1].reshape(np.shape(x)), ji[0].reshape(np.shape(x))

    def present(self, s):
        """Boolean (rows, cols) grid of the tiles present in slab s."""
        g = np.zeros((self.rows, self.cols), bool)
        g.flat[np.asarray(self.exists[s])] = True
        return g

    def cellmap(self, s):
        """For every lattice cell, the id of the nearest tile present in slab s (itself if present)."""
        if s not in self._cellmap:
            g = self.present(s)
            ii, jj = np.nonzero(g)
            I, J = np.mgrid[0:self.rows, 0:self.cols]
            d = (I[..., None] - ii) ** 2 + (J[..., None] - jj) ** 2
            k = d.argmin(-1)
            self._cellmap[s] = ii[k] * self.cols + jj[k]
        return self._cellmap[s]

    def cell(self, x, y, s):
        """Tile id of points of slab s: the lattice cell, or the nearest tile present in slab s where that cell has no
        tile."""
        i, j = self.lattice(x, y)
        i = np.clip(np.rint(i), 0, self.rows - 1).astype(int)
        j = np.clip(np.rint(j), 0, self.cols - 1).astype(int)
        return self.cellmap(s)[i, j]

    def to_dict(self):
        return dict(X0=self.X0, Y0=self.Y0, PX=self.PX, PY=self.PY, SX=self.SX, SY=self.SY, W=self.W, H=self.H,
                    rows=self.rows, cols=self.cols, exists={str(k): v for k, v in self.exists.items()})


def present_tiles(lay, masks):
    """The tiles present in each slab: more than 90 % of the tile's own cell covered by the slab's mask."""
    for s, m in masks.items():
        ok = []
        for i in range(lay.rows):
            for j in range(lay.cols):
                x0, y0 = lay.corner(i, j)
                xa, xb = int(round(x0 + lay.W / 2 - lay.PX / 2)), int(round(x0 + lay.W / 2 + lay.PX / 2))
                ya, yb = int(round(y0 + lay.H / 2 - lay.PY / 2)), int(round(y0 + lay.H / 2 + lay.PY / 2))
                xa, ya = max(xa, 0), max(ya, 0)
                sub = m[ya:yb, xa:xb]
                if sub.size and sub.mean() > 0.9:
                    ok.append(i * lay.cols + j)
        lay.exists[s] = ok
    return lay


def fit_layout(masks):
    """Lattice from the coverage masks {slab: bool (y, x)}: the left edge of their union steps by SX once per tile row
    and its top edge by SY once per tile column."""
    U = np.zeros_like(next(iter(masks.values())), bool)
    for m in masks.values():
        U |= m
    ys = np.nonzero(U.any(1))[0]; xs = np.nonzero(U.any(0))[0]
    lv, ln, ls = _runs(U[ys].argmax(1))
    tv, tn, ts = _runs(U[:, xs].argmax(0))
    flat = ValueError("the coverage outline shows no staircase of three or more tile rows and columns stepping by more than a "
                      "pixel per tile: give the tile lattice from the acquisition metadata with --layout")
    if len(lv) < 3 or len(tv) < 3:
        raise flat
    SX, X0 = np.polyfit(np.arange(len(lv)), lv.astype(float), 1)
    SY, Y0 = np.polyfit(np.arange(len(tv)), tv.astype(float), 1)
    if min(abs(SX), abs(SY)) < 1 + 1e-6:               # where tile corners round to one pixel, runs merge and every step is one
        raise flat
    PY = np.polyfit(np.arange(1, len(lv)), ys[ls[1:]].astype(float), 1)[0]     # run starts advance by one pitch
    PX = np.polyfit(np.arange(1, len(tv)), xs[ts[1:]].astype(float), 1)[0]
    H = float(ln[0] if SX > 0 else ln[-1])            # the long run is the tile that owns the overlap band
    W = float(tn[0] if SY > 0 else tn[-1])
    return present_tiles(Layout(float(X0), float(Y0), float(PX), float(PY), float(SX), float(SY), W, H, len(lv), len(tv)), masks)


def layout(spec=None):
    """Cached layout of the slabs: the lattice given as spec (Layout's arguments) or read from the coverage masks
    (fit_layout), and the tiles present in each slab from the masks."""
    path = f"{data.WORK}/layout.json"
    if spec or not os.path.exists(path):
        masks = {s: data.geometry(s)[1] for s in data.SLABS}
        lay = present_tiles(Layout(**spec), masks) if spec else fit_layout(masks)
        json.dump(lay.to_dict(), open(path, "w"), indent=1)
    d = json.load(open(path))
    ex = {int(k): v for k, v in d.pop("exists").items()}
    return Layout(**d, exists=ex)

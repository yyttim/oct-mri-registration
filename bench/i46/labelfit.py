"""Label-only fit of the I46 reference check: an affine from the OCT vessel segmentation to the MRI vessel labels.

Robust symmetric ICP with Geman-McClure weights for S9 (rotation, scale per OCT axis, shift) and A12 (full affine), and an
FFT global search over the rotations of both handednesses.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.fft import irfftn, next_fast_len, rfftn
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

RADIUS = 2.0                                # OCT voxels, smallest local vessel radius kept
CELL = 4                                    # OCT voxels per cell and axis
CLOSING = 14                                # cells
MARGIN = 0.15                               # mm inside the footprint for an MRI label to count
NEAR = 2.0                                  # mm around the label box for the OCT partners of the labels
SCALE_BOUNDS = (np.log(0.75), np.log(1.40))  # S9 scale per OCT axis, relative to the test's OCT spacing
C_START = (1.2, 0.8, 0.5, 0.3, 0.2, 0.15)   # Geman-McClure c (mm) from a search candidate
C_LOCAL = (0.6, 0.4, 0.25, 0.15)            # c from a close start
C_A12 = (0.25, 0.15)                        # c of A12 from an S9 fit
GRID = 0.6                                  # mm, search grid and smoothing
COARSE = 3                                  # footprint cells per search template cell and axis


def apply(T, p):
    return p @ T[:3, :3].T + T[:3, 3]


def oct_points(ves_tiff, spacing, chunk=48, overlap=8):
    """OCT vessel points (OCT world, mm) and the footprint on cells (x, y, plane) from the segmentation ves_tiff (plane, y, x).
    A point is the centroid per cell of the voxels of local radius >= RADIUS."""
    import tifffile
    with tifffile.TiffFile(str(ves_tiff)) as tf:
        pages = tf.pages
        nz = len(pages)
        ny, nx = pages[0].shape
        nc = np.array([nz, ny, nx]) // CELL
        ncell = int(np.prod(nc))
        count, occ = np.zeros(ncell), np.zeros(ncell, bool)
        sums = np.zeros((3, ncell))
        for z0 in range(0, nc[0] * CELL, chunk):
            a, b = max(0, z0 - overlap), min(nz, z0 + chunk + overlap)
            v = np.stack([pages[i].asarray() > 0 for i in range(a, b)])
            r = ndimage.grey_dilation(ndimage.distance_transform_edt(v).astype(np.float32), size=(5, 5, 5))
            core = slice(z0 - a, min(z0 + chunk, nz) - a)
            z, y, x = np.nonzero(v[core])
            rr = r[core][z, y, x]
            z = z + z0
            ok = (z // CELL < nc[0]) & (y // CELL < nc[1]) & (x // CELL < nc[2])
            z, y, x, rr = z[ok], y[ok], x[ok], rr[ok]
            cid = np.ravel_multi_index((z // CELL, y // CELL, x // CELL), nc)
            occ[cid] = True
            s = rr >= RADIUS
            count += np.bincount(cid[s], minlength=ncell)
            for i, c in enumerate((z, y, x)):
                sums[i] += np.bincount(cid[s], weights=c[s], minlength=ncell)
            del v, r
    keep = count > 0
    idx = (sums[:, keep] / count[keep]).T                        # (plane, y, x)
    return idx[:, ::-1] * np.asarray(spacing), footprint(occ.reshape(nc)).transpose(2, 1, 0)


def footprint(occ, r=CLOSING):
    """The region the OCT vessels cover: occupied cells (plane, y, x) closed by r, holes filled, the largest component."""
    b = ndimage.iterate_structure(ndimage.generate_binary_structure(3, 1), 2)
    pad = r + 2
    d = ndimage.binary_dilation(np.pad(occ, pad), structure=b, iterations=r // 2)
    for ax in range(3):
        d = np.moveaxis(np.stack([ndimage.binary_fill_holes(p) for p in np.moveaxis(d, ax, 0)]), 0, ax)
    d = ndimage.binary_fill_holes(d)
    e = ndimage.binary_erosion(d, structure=b, iterations=r // 2, border_value=0)[pad:-pad, pad:-pad, pad:-pad]
    lab, n = ndimage.label(e)
    if n > 1:
        size = np.bincount(lab.ravel())
        size[0] = 0
        e = lab == np.argmax(size)
    return e


def cell_affine(spacing, n=CELL):
    """Cell index -> OCT world for cells of n voxels per axis: the centre of the cell's voxels."""
    A = np.eye(4)
    A[:3, :3] = np.diag(n * np.asarray(spacing))
    A[:3, 3] = (n - 1) / 2 * np.asarray(spacing)
    return A


class Problem:
    """E(T): mean Geman-McClure rho of the MRI labels to the T-mapped OCT points plus that of the OCT points to the labels.
    O: OCT points (mm). foot, box: footprint and MRI label box (bool), with their index -> world affines. cell: mm."""

    def __init__(self, O, foot, foot_aff, box, box_aff, cell):
        self.O = np.asarray(O, float)
        self.M = apply(box_aff, np.argwhere(box).astype(float))
        self.Mtree = cKDTree(self.M)
        vs = np.linalg.norm(box_aff[:3, :3], axis=0)
        self.edt = ndimage.distance_transform_edt(~box, sampling=vs).astype(np.float32)
        self.ibox = np.linalg.inv(box_aff)
        self.near = NEAR / vs.min()
        self.shape = np.array(box.shape)
        self.fin = ndimage.distance_transform_edt(foot, sampling=np.linalg.norm(foot_aff[:3, :3], axis=0)).astype(np.float32)
        self.ifoot = np.linalg.inv(foot_aff)
        # components of the MRI labels and of the OCT points, for the split halves
        self.mlab = ndimage.label(box, structure=np.ones((3, 3, 3)))[0][box]
        c = np.round((self.O - self.O.min(0)) / np.asarray(cell)).astype(int)
        g = np.zeros(c.max(0) + 1, bool)
        g[tuple(c.T)] = True
        self.olab = ndimage.label(g, structure=np.ones((3, 3, 3)))[0][tuple(c.T)]

    def sets(self, T, idxM=None, idxO=None):
        M = self.M if idxM is None else self.M[idxM]
        O = self.O if idxO is None else self.O[idxO]
        TO = apply(T, O)
        qe = apply(self.ibox, TO)
        near = np.all((qe >= -self.near) & (qe <= self.shape - 1 + self.near), 1)
        fin = ndimage.map_coordinates(self.fin, apply(self.ifoot @ np.linalg.inv(T), M).T, order=1, cval=0.0)
        Min = M[fin >= MARGIN]
        inB = np.all((qe >= 0) & (qe <= self.shape - 1), 1)
        return Min, TO[near], O[near], O[inB], TO[inB], qe[inB]

    def energy(self, T, c, idxM=None, idxO=None, detail=False):
        Min, TOA, OA, OB, TOB, qe = self.sets(T, idxM, idxO)
        if len(Min) < 20 or len(OB) < 100 or len(TOA) < 100:
            return (9.0, {}) if detail else 9.0
        dA = cKDTree(TOA).query(Min)[0]
        dB = ndimage.map_coordinates(self.edt, qe.T, order=1)
        E = float((dA ** 2 / (dA ** 2 + c ** 2)).mean() + (dB ** 2 / (dB ** 2 + c ** 2)).mean())
        if not detail:
            return E
        return E, {"n_mri": len(Min), "n_oct": len(OB), "mri_to_oct_mean_mm": float(dA.mean()),
                   "mri_to_oct_median_mm": float(np.median(dA)), "mri_within_0.15mm": float((dA <= 0.15).mean()),
                   "oct_to_mri_mean_mm": float(dB.mean()), "oct_within_0.15mm": float((dB <= 0.15).mean())}

    def pairs(self, T, c, idxM=None, idxO=None):
        Min, TOA, OA, OB, TOB, qe = self.sets(T, idxM, idxO)
        if len(Min) < 20 or len(OB) < 100 or len(TOA) < 100:
            return None
        dA, j = cKDTree(TOA).query(Min)
        tree = self.Mtree if idxM is None else cKDTree(self.M[idxM])
        dB, k = tree.query(TOB)
        w = lambda d: c ** 2 / (d ** 2 + c ** 2) ** 2        # noqa: E731
        return np.r_[OA[j], OB], np.r_[Min, tree.data[k]], np.r_[w(dA) / len(Min), w(dB) / len(OB)]


def solve_affine(src, dst, w):
    sw = np.sqrt(w)[:, None]
    B = np.linalg.lstsq(np.c_[src, np.ones(len(src))] * sw, dst * sw, rcond=None)[0]
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = B[:3].T, B[3]
    return T


def s9_matrix(x, H):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix() @ H @ np.diag(np.exp(x[3:6]))
    T[:3, 3] = x[6:9]
    return T


def s9_params(T, H):
    """The closest S9 parameters: the polar rotation of T H and the diagonal of the rest."""
    L = T[:3, :3] @ H
    U, _, Vt = np.linalg.svd(L)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        raise ValueError("handedness of T and H differ")
    return np.r_[Rotation.from_matrix(R).as_rotvec(), np.log(np.clip(np.diag(R.T @ L), 1e-3, None)), T[:3, 3]]


def solve_s9(src, dst, w, x0, H):
    sw = np.sqrt(w)[:, None]
    lo = np.r_[[-np.inf] * 3, [SCALE_BOUNDS[0]] * 3, [-np.inf] * 3]
    hi = np.r_[[np.inf] * 3, [SCALE_BOUNDS[1]] * 3, [np.inf] * 3]
    res = lambda x: ((apply(s9_matrix(x, H), src) - dst) * sw).ravel()      # noqa: E731
    return least_squares(res, np.clip(x0, lo + 1e-9, hi - 1e-9), method="trf", bounds=(lo, hi), max_nfev=200).x


def fit(P, T0, model, cs=C_LOCAL, iters=25, tol=1e-4, idxM=None, idxO=None):
    """ICP fit of model "S9" (keeping the handedness of T0) or "A12" from T0 over the c stages cs.
    -> T and its energy and residuals at the last c, "plausible" when all singular values of T lie between 0.7 and 1.5."""
    T = T0.copy()
    if model == "S9":
        H = np.diag([1.0, 1.0, np.sign(np.linalg.det(T0[:3, :3]))])
        x = s9_params(T0, H)
        T = s9_matrix(x, H)
    for c in cs:
        for _ in range(iters):
            pr = P.pairs(T, c, idxM, idxO)
            if pr is None:
                return T, {"E": 9.0, "plausible": False}
            if model == "A12":
                Tn = solve_affine(*pr)
            else:
                x = solve_s9(*pr, x, H)
                Tn = s9_matrix(x, H)
            change = np.abs(Tn - T).max()
            T = Tn
            if change < tol:
                break
    E, d = P.energy(T, cs[-1], idxM, idxO, detail=True)
    sv = np.linalg.svd(T[:3, :3], compute_uv=False)
    return T, {"E": E, **d, "plausible": bool(sv.min() >= 0.7 and sv.max() <= 1.5)}


def super_fibonacci(n):
    """n quasi-uniform rotations (super-Fibonacci spirals, Alexa 2022), as 3 x 3 matrices."""
    phi, psi = np.sqrt(2.0), 1.533751168755204288118041
    s = np.arange(n) + 0.5
    r, R = np.sqrt(s / n), np.sqrt(1.0 - s / n)
    a, b = 2 * np.pi * s / phi, 2 * np.pi * s / psi
    return Rotation.from_quat(np.stack([r * np.sin(a), r * np.cos(a), R * np.sin(b), R * np.cos(b)], 1)).as_matrix()


def template_cells(foot, spacing, n=COARSE):
    """Centres (OCT world) of the n x n x n footprint cells at least half covered, and the volume of one (mm^3)."""
    s = np.array(foot.shape) // n * n
    f = foot[:s[0], :s[1], :s[2]].reshape(s[0] // n, n, s[1] // n, n, s[2] // n, n).mean((1, 3, 5)) >= 0.5
    m = n * CELL
    return (np.argwhere(f) * m + (m - 1) / 2) * np.asarray(spacing), float(np.prod(m * np.asarray(spacing)))


class Search:
    """FFT search over all shifts of the masked NCC of the OCT point density with the MRI label density, one rotation at a time.
    box: MRI label box (bool) with voxel size vs. B: template size in grid cells. Poses are in W', the box index times vs."""

    def __init__(self, box, vs, B, g=GRID, sigma=GRID):
        self.g, self.sigma, self.B, self.pad = g, sigma, B, B - 1
        vs = np.asarray(vs, float)
        n = np.ceil((np.array(box.shape) - 1) * vs / g).astype(int) + 1
        cells = np.ravel_multi_index(np.floor(np.argwhere(box) * vs / g + 0.5).astype(int).T, n)
        D = np.bincount(cells, minlength=int(np.prod(n))).reshape(n).astype(np.float32) * np.float32(np.prod(vs) / g ** 3)
        self.n = n
        # B - 1 empty cells on every face, so that every shift keeping the footprint inside the box is scored
        image = np.pad(ndimage.gaussian_filter(D, sigma / g), self.pad)
        self.sI = np.array(image.shape)
        self.fshape = [next_fast_len(int(a + B)) for a in self.sI]
        self.fI = rfftn(image, self.fshape, workers=1)
        self.fI2 = rfftn(image ** 2, self.fshape, workers=1)

    def _corr(self, fa_rev, fb):
        c = irfftn(fa_rev * fb, self.fshape, workers=1)
        o = self.B - 1
        return c[o:self.sI[0], o:self.sI[1], o:self.sI[2]]

    def run(self, O, F, L, cen, fcell, fthr=0.5):
        """NCC over all shifts for the linear map L about cen, and the template origin tlo.
        Index s + pad puts template cell 0 at label cell s."""
        g, B = self.g, self.B
        tlo = -g * (B - 1) / 2.0
        io = np.floor(((O - cen) @ L.T - tlo) / g + 0.5).astype(int)
        iff = np.floor(((F - cen) @ L.T - tlo) / g + 0.5).astype(int)
        io = io[np.all((io >= 0) & (io < B), 1)]
        iff = iff[np.all((iff >= 0) & (iff < B), 1)]
        T = np.bincount(np.ravel_multi_index(io.T, (B, B, B)), minlength=B ** 3).reshape(B, B, B).astype(np.float32)
        M = np.bincount(np.ravel_multi_index(iff.T, (B, B, B)), minlength=B ** 3).reshape(B, B, B).astype(np.float32)
        Ms = (M >= fthr * fcell).astype(np.float32)
        Tb = ndimage.gaussian_filter(T, self.sigma / g)
        nM = Ms.sum()
        Tz = (Tb - (Tb * Ms).sum() / nM) * Ms
        fT = rfftn(Tz[::-1, ::-1, ::-1], self.fshape, workers=1)
        fM = rfftn(Ms[::-1, ::-1, ::-1], self.fshape, workers=1)
        num, s1, s2 = self._corr(fT, self.fI), self._corr(fM, self.fI), self._corr(fM, self.fI2)
        varI = s2 - s1 ** 2 / nM
        ncc = num / np.sqrt(np.maximum((Tz ** 2).sum() * varI, 1e-12))
        nz = np.argwhere(Ms > 0)                    # the box of the mask must lie inside the label box
        lo = np.maximum(self.pad - nz.min(0), 0)
        hi = np.minimum(self.pad + self.n - 1 - nz.max(0), np.array(ncc.shape) - 1)
        valid = np.zeros(ncc.shape, bool)
        if np.all(hi >= lo):
            valid[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] = True
        ncc[~valid | (varI < 1e-6)] = -1
        return ncc, tlo

    def pose(self, L, cen, k, tlo):
        """OCT world -> W' for the linear map L and the correlation index k of run."""
        T = np.eye(4)
        T[:3, :3] = L
        T[:3, 3] = (np.asarray(k) - self.pad) * self.g - tlo - L @ cen
        return T


def top_peaks(ncc, k=3, sep=3):
    """The k highest values of ncc at least sep + 1 cells apart (per axis), as (value, index)."""
    out, n = [], ncc.copy()
    for _ in range(k):
        i = np.unravel_index(np.argmax(n), n.shape)
        if n[i] <= -1:
            break
        out.append((float(n[i]), i))
        n[tuple(slice(max(0, x - sep), x + sep + 1) for x in i)] = -1
    return out

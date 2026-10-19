"""`octreg register` and `octreg apply`. T is the 4x4 affine (mm) from the OCT world to the MRI world."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from . import __version__, geometry as G, io, preprocess as pp
from .params import Params
from .refine import refine
from .search import search

OUTPUTS = ("T_oct2mri.txt", "oct2mri.lta", "oct2mri_itk.txt", "oct_in_mri.nii.gz", "mri_in_oct.nii.gz", "qc.png",
           "qc_montage.png", "result.json")


def register(oct_path, mri_path, out_dir, oct_spacing_um=None, oct_mask=None, mri_mask=None, params: Params = Params(),
             device="cuda") -> dict:
    """Register an OCT block to an MRI cropped around it and write OUTPUTS into out_dir.
    oct_spacing_um: (z, y, x) um. oct_mask, mri_mask: files with voxels > 0 in the world of their image. -> result.json dict.
    Raises ValueError for a refused input."""
    P, out, t0, sec = params, Path(out_dir), time.time(), {}
    cuda = str(device).startswith("cuda")
    if cuda and not torch.cuda.is_available():
        raise ValueError("CUDA is not available: use --device cpu (device='cpu' in Python)")
    vo, vm = io.load_volume(oct_path, oct_spacing_um), io.load_volume(mri_path)
    mo = None if oct_mask is None else io.load_volume(oct_mask, oct_spacing_um)
    mm = None if mri_mask is None else io.load_volume(mri_mask)
    for mask, image, hint in ((mo, vo, "a NIfTI mask for a NIfTI OCT, an array-frame TIFF / NPY mask for a TIFF / NPY OCT read "
                               "in its array frame"),
                              (mm, vm, "a NIfTI mask with an sform or qform exactly when the MRI has one")):
        if mask is not None and mask.frame != image.frame:
            raise ValueError(f"{mask.path} is in the {mask.frame} frame and {image.path} in the {image.frame} frame: a mask "
                             f"must be in the world of its image ({hint})")
    if mo is not None and oct_spacing_um is not None and mo.shape != vo.shape:
        try:
            mo = io.load_volume(oct_mask)                       # a mask on its own grid keeps the spacing of its file
        except ValueError:
            raise ValueError(f"{oct_mask}: an OCT mask of another array shape than the OCT needs a spacing of its own (an "
                             "OME-TIFF with PhysicalSize X / Y / Z, or a NIfTI without sform or qform), since --oct-spacing-um "
                             "applies only to a mask of the OCT's shape") from None
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").unlink(missing_ok=True)               # left by an earlier run in this directory

    def lap(name):
        sec[name] = time.time() - t0 - sum(sec.values())
        print(f"octreg [{time.time() - t0:6.0f} s] {name} done in {sec[name]:.0f} s", flush=True)

    (mri_h, mask_m, A_m), fg_m = prepare_mri(vm, P, mm)
    lap("mri")
    fine, A_f = G.resample_iso(vo, P.fine_mm)
    overlay = G.pool_iso(fine, A_f, max(G.mm6(vm.spacing_mm.min()), P.fine_mm))                 # for oct_in_mri
    lap("oct_fine_grid")
    (oct_h, mask_o, A_o), measured_o, fg_o = prepare_oct(fine, A_f, P, mo)
    del fine
    lap("oct_mask")
    mri_b, oct_b = (mri_h, mask_m, A_m), (oct_h, mask_o, A_o)
    outline, outside = outline_used(mask_o, measured_o, mask_m, P)
    pool = lambda x, A: G.pool_iso(np.asarray(x, np.float32), A, P.search_mm)
    (ms, A_ms), (os_, A_os) = pool(mri_h, A_m), pool(oct_h, A_o)
    candidates = search(ms, pool(mask_m, A_m)[0], A_ms, os_, pool(mask_o, A_o)[0], pool(measured_o, A_o)[0], A_os, P,
                                     device, outline)
    lap("search")
    poses = refine([c["T"] for c in candidates], mri_b, oct_b, P, device)
    best = poses[0]
    T = best["T"]
    lap("refine")
    io.write_transform_txt(T, out / "T_oct2mri.txt")
    io.write_lta(T, vo, vm, out / "oct2mri.lta")
    io.write_itk(T, out / "oct2mri_itk.txt")

    found = candidates[best["search_rank"]]
    pose = {"search_rank": best["search_rank"], "search_F": found["F"], "search_O": found["O"], "L": best["L"],
            "outline_used": outline, "T_search": found["T"], "F": best["F"],
            "scale_per_oct_axis": np.linalg.norm(T[:3, :3] @ (vo.affine[:3, :3] / vo.spacing_mm), axis=0).tolist()}
    inputs = {"oct": {**vars(vo), "path": _abs(vo.path)}, "mri": {**vars(vm), "path": _abs(vm.path)},
              "oct_spacing_um": oct_spacing_um, "oct_mask": oct_mask and _abs(oct_mask), "mri_mask": mri_mask and _abs(mri_mask), "device": str(device)}
    result = {"octreg_version": __version__, "inputs": inputs, "params": P.to_dict(), "T_oct2mri": T, "pose": pose, "candidates": [
                  {"search_rank": c["search_rank"], "search_F": candidates[c["search_rank"]]["F"],
                   "search_O": candidates[c["search_rank"]]["O"], "L": c["L"]} for c in poses],
              "foreground": {"oct": {**fg_o, "outside_fraction": outside["oct"]},
                             "mri": {**fg_m, "outside_fraction": outside["mri"]}}}
    io.write_json(result, out / "result.json")                  # before the overlays, so a failing overlay loses no result
    io.save_nifti(G.resample_to(*overlay, vm.shape, vm.affine, np.linalg.inv(T)), vm.affine, out / "oct_in_mri.nii.gz")
    io.save_nifti(G.resample_to(mri_h, A_m, oct_h.shape, A_o, T), A_o, out / "mri_in_oct.nii.gz")
    qc_figures(out / "qc", oct_b, mri_b, T, vo.affine,
               f"{pose_label(pose)}   scale per OCT axis {np.round(pose['scale_per_oct_axis'], 3).tolist()}")
    lap("outputs")
    result["seconds"] = {**sec, "total": time.time() - t0}
    io.write_json(result, out / "result.json")
    return result


def prepare_mri(vol, params: Params = Params(), mask=None):
    """§1 for the MRI: base grid and foreground, from the mask file if given, else from the histogram valley.
    -> ((mri_h float32 [D, H, W], foreground bool, affine), info)."""
    h = params.base_mm
    arr, A = G.resample_iso(vol, h)
    fg, info = pp.foreground(arr, h, params) if mask is None else (_mask_on(mask, arr.shape, A, h), {})
    return (arr, fg, A), info


def prepare_oct(fine, affine, params: Params = Params(), mask=None):
    """§1 for the OCT: specimen mask on the fine grid, then OCT, mask and measured fraction pooled to base_mm.
    -> ((oct_h float32, mask bool, base affine), measured bool, info)."""
    P = params
    mask_f, info = fine_mask(fine, affine, P, mask)
    arr, A = G.pool_iso(fine, affine, P.base_mm)
    m, measured = (G.pool_iso(x, affine, P.base_mm)[0] > 0.5 for x in (mask_f, fine > 0))
    return (arr, m, A), measured, info


def fine_mask(fine, affine, params: Params = Params(), mask=None):
    """OCT specimen mask on the fine grid: the mask file AND fine > 0 if given, else preprocess.specimen_mask.
    Voxels <= 0 are missing data. -> (bool [D, H, W], info)."""
    if mask is not None:
        m, info = _mask_on(mask, fine.shape, affine, params.fine_mm) & (fine > 0), {}
    else:
        m, info = pp.specimen_mask(fine, params.fine_mm, params)
    if not m.any():
        raise ValueError("OCT specimen mask: no voxel of the mask has a positive OCT intensity, octreg needs positive OCT "
                         "intensities on a linear scale (voxels <= 0 are missing data)")
    return m, info


def outline_used(mask_o, measured_o, mask_m, params: Params = Params()):
    """§2: use the outline only if both images have at least outline_min_outside measured outside the tissue.
    -> (bool, {'oct': fraction, 'mri': fraction})."""
    # the erosion removes the resampling shell of a mask
    out_o = float(ndimage.binary_erosion(np.asarray(measured_o, bool) & ~np.asarray(mask_o, bool)).sum()
                  / max(np.sum(measured_o), 1))
    out_m = float(ndimage.binary_erosion(~np.asarray(mask_m, bool)).sum() / np.asarray(mask_m).size)
    return min(out_o, out_m) >= params.outline_min_outside, {"oct": round(out_o, 5), "mri": round(out_m, 5)}


def apply(run_dir, moving, reference, out, inverse=False, oct_spacing_um=None) -> Path:
    """Resample `moving` onto the grid of `reference` through the run's T, or T^-1 with inverse. oct_spacing_um: (z, y, x)
    um of the OCT-frame file, by default the run's for its own OCT file at its recorded path. -> out.
    Raises ValueError for a refused input."""
    run = Path(run_dir)
    for f in (run / "T_oct2mri.txt", run / "result.json"):
        if not f.is_file():
            raise ValueError(f"{f}: no such file")
    T = np.loadtxt(run / "T_oct2mri.txt")
    T = np.linalg.inv(T) if inverse else T
    inp = json.loads((run / "result.json").read_text())["inputs"]
    oct_i = int(bool(inverse))                                  # the file in the OCT frame: moving, reference with inverse
    mv, rf = (io.load_volume(f, oct_spacing_um if oct_spacing_um is not None and i == oct_i else
                             inp["oct_spacing_um"] if _abs(f) == inp["oct"]["path"] else None)
              for i, f in enumerate((moving, reference)))
    arr, A = G.resample_iso(mv, max(G.mm6(rf.spacing_mm.min()), G.mm6(mv.spacing_mm.min())))
    io.save_nifti(G.resample_to(arr, A, rf.shape, rf.affine, np.linalg.inv(T)), rf.affine, out)
    return Path(out)


def pose_label(pose):
    """One line of the §3 and §4 read-outs from the pose of result.json."""
    F = ", ".join(f"{p['start']:.4f} -> {p['end']:.4f} at {p['sigma_mm']:g} mm" for p in pose["F"])
    O = "" if pose["search_O"] is None else f", O {pose['search_O']:.4f}"
    return f"§3: rank {pose['search_rank']} (F {pose['search_F']:.4f}{O}). §4: F {F}"


def _abs(path):
    return str(Path(path).resolve())


def _mask_on(vol, shape, affine, voxel_mm):
    """A mask file (voxels > 0) box-averaged and resampled onto a grid of spacing voxel_mm in the same world. -> bool array."""
    frac, A = G.resample_iso(vol, max(voxel_mm, G.mm6(vol.spacing_mm.min())), binary=True)
    return G.resample_to(frac, A, shape, affine, np.eye(4)) > 0.5


QC_COLOURS, QC_CELL_IN, QC_HEAD_IN, QC_DPI = ("#ff3030", "#30d0ff"), 2.8, 0.4, 100


def _match(x, y, where, n=256):
    """x mapped by rank onto the grey scale of y over the voxels `where`, for display."""
    if where.sum() < 100:
        return x
    q = np.linspace(0, 1, n)
    return np.interp(x, np.quantile(x[where], q), np.quantile(y[where], q))


def qc_figures(prefix, oct_grid, mri_grid, T, oct_file_affine, label):
    """Visual QC: one plane per OCT axis through the mask centroid in prefix.png, 4 per axis in prefix_montage.png.
    oct_grid, mri_grid: (array, mask, affine) on the base grids."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    from PIL import Image
    arr, mask, A_o = oct_grid
    h, sp = G.mm6(np.linalg.norm(A_o[:3, 0])), np.linalg.norm(oct_file_affine[:3, :3], axis=0)
    mri, fg = (G.resample_to(np.asarray(x, np.float32), mri_grid[2], arr.shape, A_o, T) for x in mri_grid[:2])
    tissue = fg > 0.5
    inside = mask & tissue
    polarity = -1 if inside.sum() >= 10 and np.corrcoef(arr[inside], mri[inside])[0, 1] < 0 else 1
    both = inside if inside.any() else mask
    o, m = (np.clip((x - lo) / max(hi - lo, 1e-12), 0, 1)
            for x, w in ((arr, mask), (mri, both)) for lo, hi in [np.percentile(x[w], [1, 99])])
    m = np.where(tissue, 1 - m, m) if polarity < 0 else m
    m = np.where(tissue, _match(m, o, inside), m)
    origin = (np.linalg.inv(oct_file_affine) @ A_o)[:3, 3] * sp
    centre = np.rint(ndimage.center_of_mass(mask)).astype(int)
    span = [np.flatnonzero(mask.any(axis=tuple({0, 1, 2} - {a})))[[0, -1]] for a in range(3)]
    views = {"": [(a, centre[a]) for a in range(3)],
             "_montage": [(a, int(round(lo + (hi - lo) * t / 5))) for a, (lo, hi) in enumerate(span) for t in range(1, 5)]}
    heads = ("OCT", "MRI through T" + ", inverted" * (polarity < 0), "checkerboard", "MRI foreground (red), OCT mask (cyan)")
    tile, palette = max(2, round(2.0 / h)), Image.new("P", (1, 1))                   # PNG: 254 greys and the outline colours
    palette.putpalette(np.repeat(np.linspace(0, 255, 254).round().astype(np.uint8), 3).tobytes()
                       + bytes.fromhex("".join(c[1:] for c in QC_COLOURS)))
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    for suffix, planes in views.items():
        W, H = 0.3 + 4 * QC_CELL_IN, QC_HEAD_IN + len(planes) * QC_CELL_IN
        fig = Figure(figsize=(W, H), dpi=QC_DPI)
        fig.suptitle(label, y=1 - 0.06 / H, va="top", fontsize=9)
        for r, (a, c) in enumerate(planes):
            po, pm, pf, pw = (np.take(x, c, a).T.astype(np.float32) for x in (o, m, fg, mask))
            board = (np.arange(po.shape[0])[:, None] // tile + np.arange(po.shape[1]) // tile) % 2 == 0
            y = H - QC_HEAD_IN - (r + 1) * QC_CELL_IN
            fig.text(0.2 / W, (y + QC_CELL_IN / 2) / H, f"axis {a} at {origin[a] + c * h:.2f} mm", rotation=90, ha="center",
                     va="center", fontsize=8)
            for j, img in enumerate((po, pm, np.where(board, po, pm) * ((pw > 0.5) | (pf > 0.5)), po)):
                ax = fig.add_axes(((0.34 + j * QC_CELL_IN) / W, (y + 0.04) / H, (QC_CELL_IN - 0.08) / W, (QC_CELL_IN - 0.08) / H))
                ax.imshow(img, cmap="gray", vmin=0, vmax=1, origin="lower")
                ax.set_axis_off()
                if r == 0:
                    ax.set_title(heads[j], fontsize=8)
                if j == 0:
                    bar = max([b for b in (0.5, 1, 2, 5, 10, 20) if b <= po.shape[1] * h / 4], default=0.5)
                    ax.add_artist(AnchoredSizeBar(ax.transData, bar / h, f"{bar:g} mm", "lower left", color="w", frameon=False,
                                                  size_vertical=max(1, po.shape[0] / 100), fontproperties={"size": 7}))
            for x, colour in zip((pf, pw), QC_COLOURS):
                if x.min() < 0.5 < x.max():
                    ax.contour(x, [0.5], colors=colour, linewidths=1.2)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        Image.fromarray(np.asarray(canvas.buffer_rgba())[..., :3]).quantize(palette=palette, dither=0).save(
            f"{prefix}{suffix}.png", optimize=True)

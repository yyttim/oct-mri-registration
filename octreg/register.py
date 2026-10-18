"""`octreg register` and `octreg apply`. T is the 4x4 affine (mm) from the OCT world to the MRI world.
With the field u of §5 the registered OCT at the MRI point x is OCT(T^-1 (x + u(x))).
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from . import __version__, geometry as G, io, preprocess as pp
from .deform import smooth_deformation
from .params import Params
from .refine import refine
from .search import search

OUTPUTS = ("T_oct2mri.txt", "oct2mri.lta", "oct2mri_itk.txt", "oct_in_mri.nii.gz", "oct_in_mri_affine.nii.gz",
           "mri_in_oct.nii.gz", "qc.png", "qc_montage.png", "result.json")
WARP = "oct2mri_warp.nii.gz"
OUTPUTS_DEFORM = (WARP, "qc_deform.png")


def register(oct_path, mri_path, out_dir, oct_spacing_um=None, oct_mask=None, mri_mask=None, params: Params = Params(),
             device="cuda", deform=True) -> dict:
    """Register an OCT block to an MRI cropped around it and write OUTPUTS, and OUTPUTS_DEFORM with deform, into out_dir.
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
    for f in OUTPUTS_DEFORM + ("result.json",):                 # left by an earlier run in this directory
        (out / f).unlink(missing_ok=True)

    def lap(name):
        sec[name] = time.time() - t0 - sum(sec.values())
        print(f"octreg [{time.time() - t0:6.0f} s] {name} done in {sec[name]:.0f} s", flush=True)

    (mri_h, mask_m, A_m), fg_m = prepare_mri(vm, P, mm)
    lap("mri")
    fine, A_f = G.resample_iso(vo, P.fine_mm)
    overlay = G.pool_iso(fine, A_f, max(_mm(vm.spacing_mm.min()), P.fine_mm))                   # for oct_in_mri
    lap("oct_fine_grid")
    (oct_h, mask_o, A_o), valid_o, fg_o = prepare_oct(fine, A_f, P, mo)
    del fine
    lap("oct_mask")
    mri_b, oct_b = (mri_h, mask_m, A_m), (oct_h, mask_o, A_o)
    outline, outside = outline_used(mask_o, valid_o, mask_m, P)
    pool = lambda x, A: G.pool_iso(np.asarray(x, np.float32), A, P.search_mm)
    (ms, A_ms), (os_, A_os) = pool(mri_h, A_m), pool(oct_h, A_o)
    candidates = search(ms, pool(mask_m, A_m)[0], A_ms, os_, pool(mask_o, A_o)[0], pool(valid_o, A_o)[0], A_os, P,
                                     device, outline)
    lap("search")
    poses = refine([c["T"] for c in candidates], mri_b, oct_b, P, device)
    best = poses[0]
    T = best["T"]
    lap("refine")
    io.write_transform_txt(T, out / "T_oct2mri.txt")            # before §5, so a failure there keeps the affine
    io.write_lta(T, vo, vm, out / "oct2mri.lta")
    io.write_itk(T, vo, vm, out / "oct2mri_itk.txt")
    field, info_deform = smooth_deformation(mri_b, oct_b, T, P, device) if deform else (None, None)
    lap("deform")

    found = candidates[best["search_rank"]]
    pose = {"search_rank": best["search_rank"], "search_F": found["F"], "search_O": found["O"], "L": best["L"],
            "outline_used": outline, "T_search": found["T"], "F": best["F"],
            "scale_per_oct_axis": np.linalg.norm(T[:3, :3] @ (vo.affine[:3, :3] / vo.spacing_mm), axis=0).tolist()}
    polarity = display_polarity(oct_b, mri_b, T)
    inputs = {"oct": {**vars(vo), "path": _abs(vo.path)}, "mri": {**vars(vm), "path": _abs(vm.path)},
              "oct_spacing_um": oct_spacing_um, "oct_mask": oct_mask and _abs(oct_mask), "mri_mask": mri_mask and _abs(mri_mask), "device": str(device)}
    result = {"octreg_version": __version__, "inputs": inputs, "params": P.to_dict(), "T_oct2mri": T, "pose": pose, "candidates": [
                  {"search_rank": c["search_rank"], "search_F": candidates[c["search_rank"]]["F"],
                   "search_O": candidates[c["search_rank"]]["O"], "L": c["L"]} for c in poses],
              "foreground": {"oct": {**fg_o, "outside_fraction": outside["oct"]},
                             "mri": {"source": "histogram", **fg_m, "outside_fraction": outside["mri"]}},
              "deform": info_deform}
    io.write_json(result, out / "result.json")                  # before the overlays, so a failing overlay loses no result
    io.save_nifti(G.resample_to(*overlay, vm.shape, vm.affine, np.linalg.inv(T)), vm.affine, out / "oct_in_mri_affine.nii.gz")
    if field is not None:
        io.save_field(field, A_m, out / WARP)
        io.save_nifti(G.resample_to(*overlay, vm.shape, vm.affine, np.linalg.inv(T), field=field, affine_field=A_m), vm.affine,
                      out / "oct_in_mri.nii.gz")
    else:
        shutil.copyfile(out / "oct_in_mri_affine.nii.gz", out / "oct_in_mri.nii.gz")
    io.save_nifti(G.resample_to(mri_h, A_m, oct_h.shape, A_o, T), A_o, out / "mri_in_oct.nii.gz")
    qc_figures(out / "qc", (oct_h, mask_o, A_o), (mri_h, mask_m, A_m), T, polarity, vo.affine, pose_label(pose))
    if field is not None:
        qc_deform_figure(out / "qc_deform.png", (oct_h, mask_o, A_o), (mri_h, mask_m, A_m), T, field, polarity,
                         deform_label(info_deform))
    lap("outputs")
    result["seconds"] = {**sec, "total": time.time() - t0}
    io.write_json(result, out / "result.json")
    return result


def display_polarity(oct_b, mri_b, T):
    """Sign of the OCT and MRI intensity correlation inside the OCT specimen mask at T, for the QC display only."""
    (o, mo, A_o), (m, mm, A_m) = oct_b, mri_b
    vals = G.resample_to(m, A_m, o.shape, A_o, T)
    inside = np.asarray(mo, bool) & (G.resample_to(np.asarray(mm, np.float32), A_m, o.shape, A_o, T) > 0.5)
    if inside.sum() < 10:
        return 1
    return 1 if np.corrcoef(np.asarray(o)[inside], vals[inside])[0, 1] >= 0 else -1


def prepare_mri(vol, params: Params = Params(), mask=None):
    """§1 for the MRI: base grid and foreground, from the mask file if given, else from the histogram valley.
    -> ((mri_h float32 [D, H, W], foreground bool, affine), info)."""
    h = params.base_mm
    arr, A = G.resample_iso(vol, h)
    fg, info = pp.foreground(arr, h, params) if mask is None else (_mask_on(mask, arr.shape, A, h), {"source": mask.path})
    return (arr, fg, A), info


def prepare_oct(fine, affine, params: Params = Params(), mask=None):
    """§1 for the OCT: specimen mask on the fine grid, then OCT, mask and measured fraction box-averaged to base_mm.
    -> ((oct_h float32, mask bool, base affine), measured bool, info)."""
    P = params
    mask_f, info = fine_mask(fine, affine, P, mask)
    arr, A = G.pool_iso(fine, affine, P.base_mm)
    m, valid = (G.pool_iso(x, affine, P.base_mm)[0] > 0.5 for x in (mask_f, fine > 0))
    return (arr, m, A), valid, info


def fine_mask(fine, affine, params: Params = Params(), mask=None):
    """OCT specimen mask on the fine grid: the mask file if given, else preprocess.specimen_mask, AND fine > 0.
    Voxels <= 0 are missing data. -> (bool [D, H, W], info)."""
    if mask is not None:
        m, info = _mask_on(mask, fine.shape, affine, params.fine_mm), {"source": mask.path}
    else:
        m, info = pp.specimen_mask(fine, params.fine_mm, params)
        info["source"] = "octreg"
    np.logical_and(m, fine > 0, out=m)
    if not m.any():
        raise ValueError("OCT specimen mask: no voxel of the mask has a positive OCT intensity, octreg needs positive OCT "
                         "intensities on a linear scale (voxels <= 0 are missing data)")
    return m, info


def outline_used(mask_o, valid_o, mask_m, params: Params = Params()):
    """§2: use the outline only if both images have at least outline_min_outside measured outside the tissue.
    -> (bool, {'oct': fraction, 'mri': fraction})."""
    # the erosion removes the resampling shell of a mask
    out_o = float(ndimage.binary_erosion(np.asarray(valid_o, bool) & ~np.asarray(mask_o, bool)).sum() / max(np.sum(valid_o), 1))
    out_m = float(ndimage.binary_erosion(~np.asarray(mask_m, bool)).sum() / np.asarray(mask_m).size)
    return min(out_o, out_m) >= params.outline_min_outside, {"oct": round(out_o, 5), "mri": round(out_m, 5)}


def apply(run_dir, moving, reference, out, inverse=False, affine_only=False, oct_spacing_um=None) -> Path:
    """Resample `moving` onto the grid of `reference` through the run's T and §5 field, T alone with affine_only, or T^-1
    alone with inverse. oct_spacing_um: (z, y, x) um of the OCT-frame file, by default the run's for its own OCT file at
    its recorded path. -> out. Raises ValueError for a refused input."""
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
    field, A_field = io.load_field(run / WARP) if (run / WARP).is_file() and not (inverse or affine_only) else (None, None)
    if inverse and (run / WARP).is_file():
        print(f"octreg: note: {WARP} is not inverted, --inverse resamples through the affine alone", flush=True)
    arr, A = G.resample_iso(mv, max(_mm(rf.spacing_mm.min()), _mm(mv.spacing_mm.min())))
    io.save_nifti(G.resample_to(arr, A, rf.shape, rf.affine, np.linalg.inv(T), field=field, affine_field=A_field), rf.affine, out)
    return Path(out)


def deform_label(d):
    """One line of the §5 read-outs from the info of deform.smooth_deformation."""
    f = d["passes"][-1]
    F = ", ".join(f"{p['F_start']:.4f} -> {p['F_end']:.4f} at {p['sigma_mm']:g} mm" for p in d["passes"])
    return f"§5: F {F}. Displacement median {f['displacement_median_mm']:.2f}, max {f['displacement_max_mm']:.2f} mm"


def pose_label(pose):
    """One line of the §3 and §4 read-outs from the pose of result.json."""
    F = ", ".join(f"{p['start']:.4f} -> {p['end']:.4f} at {p['sigma_mm']:g} mm" for p in pose["F"])
    O = "" if pose["search_O"] is None else f", O {pose['search_O']:.4f}"
    return f"§3: rank {pose['search_rank']} (F {pose['search_F']:.4f}{O}). §4: F {F}"


def _mm(x):
    """A length to 6 significant digits, dropping float32 header noise."""
    return float(f"{x:.6g}")


def _abs(path):
    return str(Path(path).resolve())


def _mask_on(vol, shape, affine, voxel_mm):
    """A mask file (voxels > 0) box-averaged and resampled onto a grid of spacing voxel_mm in the same world. -> bool array."""
    frac, A = G.resample_iso(vol, max(voxel_mm, _mm(vol.spacing_mm.min())), binary=True)
    return G.resample_to(frac, A, shape, affine, np.eye(4)) > 0.5


QC_COLOURS, QC_CELL_IN, QC_HEAD_IN, QC_DPI = ("#ff3030", "#30d0ff"), 2.8, 0.4, 100


def _match(x, y, where, n=256):
    """x mapped by rank onto the grey scale of y over the voxels `where`, for display."""
    if where.sum() < 100:
        return x
    q = np.linspace(0, 1, n)
    return np.interp(x, np.quantile(x[where], q), np.quantile(y[where], q))


def qc_figures(prefix, oct_grid, mri_grid, T, polarity, oct_file_affine, label=""):
    """Visual QC of the affine: one plane per OCT axis through the mask centroid in prefix.png, 4 per axis in the montage.
    oct_grid, mri_grid: (array, mask, affine) on the base grids. -> (prefix.png, prefix_montage.png) Paths."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    from PIL import Image
    arr, mask, A_o = oct_grid
    h, sp = _mm(np.linalg.norm(A_o[:3, 0])), np.linalg.norm(oct_file_affine[:3, :3], axis=0)
    mri, fg = (G.resample_to(np.asarray(x, np.float32), mri_grid[2], arr.shape, A_o, T) for x in mri_grid[:2])
    tissue = fg > 0.5
    both = mask & tissue if (mask & tissue).any() else mask
    o, m = (np.clip((x - lo) / max(hi - lo, 1e-12), 0, 1)
            for x, w in ((arr, mask), (mri, both)) for lo, hi in [np.percentile(x[w], [1, 99])])
    m = np.where(tissue, 1 - m, m) if polarity < 0 else m
    m = np.where(tissue, _match(m, o, mask & tissue), m)
    origin = (np.linalg.inv(oct_file_affine) @ A_o)[:3, 3] * sp
    centre = np.rint(ndimage.center_of_mass(mask)).astype(int)
    span = [np.flatnonzero(mask.any(axis=tuple({0, 1, 2} - {a})))[[0, -1]] for a in range(3)]
    views = {"": [(a, centre[a]) for a in range(3)],
             "_montage": [(a, int(round(lo + (hi - lo) * t / 5))) for a, (lo, hi) in enumerate(span) for t in range(1, 5)]}
    heads = ("OCT", "MRI through T" + ", inverted" * (polarity < 0), "checkerboard", "MRI foreground (red), OCT mask (cyan)")
    scales = np.round(np.linalg.norm(T[:3, :3] @ (oct_file_affine[:3, :3] / sp), axis=0), 3).tolist()
    tile, paths, palette = max(2, round(2.0 / h)), [], Image.new("P", (1, 1))        # PNG: 254 greys and the outline colours
    palette.putpalette(np.repeat(np.linspace(0, 255, 254).round().astype(np.uint8), 3).tobytes()
                       + bytes.fromhex("".join(c[1:] for c in QC_COLOURS)))
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    for suffix, planes in views.items():
        W, H = 0.3 + 4 * QC_CELL_IN, QC_HEAD_IN + len(planes) * QC_CELL_IN
        fig = Figure(figsize=(W, H), dpi=QC_DPI)
        fig.suptitle(f"{label}   scale per OCT axis {scales}", y=1 - 0.06 / H, va="top", fontsize=9)
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
        paths.append(Path(f"{prefix}{suffix}.png"))
        Image.fromarray(np.asarray(canvas.buffer_rgba())[..., :3]).quantize(palette=palette, dither=0).save(paths[-1], optimize=True)
    return tuple(paths)


def qc_deform_figure(path, oct_grid, mri_grid, T, field, polarity, label=""):
    """Visual QC of §5: one plane per MRI axis through the OCT mask in the MRI foreground, on the MRI base grid.
    oct_grid, mri_grid: (array, mask, affine) on the base grids. field: [3, D, H, W] on the MRI base grid. -> Path."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from PIL import Image
    (o_arr, o_mask, A_o), (mri, fg, A_m) = oct_grid, mri_grid
    h, Ti = _mm(np.linalg.norm(A_m[:3, 0])), np.linalg.inv(T)
    inside = G.resample_to(np.asarray(o_mask, np.float32), A_o, mri.shape, A_m, Ti) > 0.5
    octs = [G.resample_to(o_arr, A_o, mri.shape, A_m, Ti, **kw) for kw in ({}, {"field": field, "affine_field": A_m})]
    both = inside & fg if (inside & fg).any() else fg
    lo, hi = np.percentile(octs[0][inside if inside.any() else fg], [1, 99])
    octs = [np.clip((x - lo) / max(hi - lo, 1e-12), 0, 1) for x in octs]
    lo, hi = np.percentile(mri[both], [1, 99])
    m = np.clip((mri - lo) / max(hi - lo, 1e-12), 0, 1)
    m = np.where(fg, 1 - m, m) if polarity < 0 else m
    m = np.where(fg, _match(m, octs[0], inside & fg), m)
    mag = np.where(fg, np.linalg.norm(field, axis=0), np.nan)
    top = max(float(np.nanmax(mag)) if fg.any() else 0.0, 1e-3)
    centre = np.rint(ndimage.center_of_mass(both)).astype(int)
    heads = ("MRI" + ", inverted" * (polarity < 0), "OCT through T", "OCT through T and the field", "field magnitude (mm)")
    W, H = 0.3 + 4 * QC_CELL_IN + 0.55, QC_HEAD_IN + 3 * QC_CELL_IN
    fig = Figure(figsize=(W, H), dpi=QC_DPI)
    fig.suptitle(label, y=1 - 0.06 / H, va="top", fontsize=8)
    for a in range(3):
        y = H - QC_HEAD_IN - (a + 1) * QC_CELL_IN
        fig.text(0.2 / W, (y + QC_CELL_IN / 2) / H, f"axis {a} at {centre[a] * h:.2f} mm", rotation=90, ha="center", va="center",
                 fontsize=8)
        contour = np.take(fg, centre[a], a).T.astype(np.float32)
        for j, x in enumerate((m, *octs, mag)):
            ax = fig.add_axes(((0.34 + j * QC_CELL_IN) / W, (y + 0.04) / H, (QC_CELL_IN - 0.08) / W, (QC_CELL_IN - 0.08) / H))
            ax.set_facecolor("k")
            im = ax.imshow(np.take(x, centre[a], a).T, origin="lower", **({"cmap": "gray", "vmin": 0, "vmax": 1} if j < 3 else
                                                                         {"cmap": "viridis", "vmin": 0, "vmax": top}))
            ax.set_xticks([])
            ax.set_yticks([])
            if contour.min() < 0.5 < contour.max():
                ax.contour(contour, [0.5], colors=QC_COLOURS[0], linewidths=1.0)
            if a == 0:
                ax.set_title(heads[j], fontsize=8)
        bar = fig.colorbar(im, cax=fig.add_axes(((0.34 + 4 * QC_CELL_IN) / W, (y + 0.04) / H, 0.1 / W, (QC_CELL_IN - 0.08) / H)))
        bar.ax.tick_params(labelsize=7)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(canvas.buffer_rgba())[..., :3]).save(path, optimize=True)
    return Path(path)

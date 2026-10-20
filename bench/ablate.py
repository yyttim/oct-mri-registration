#!/usr/bin/env python3
"""Ablations of octreg: the method with one change per variant, run on one pair.

    python bench/ablate.py OCT MRI -o OUT [--oct-mask MASK] [--oct-spacing-um Z,Y,X] [--only A,B] [--turn RX,RY,RZ]
                           [--start FILE] [--params F] [--device cuda]

Each variant changes one function or Params value for its run, or the OCT it is given:

{variants}

OUT/<variant>/ holds the octreg register outputs, variant.json (the change, the Params and the turn) and code.txt (the
code_id of the octreg code that made it). A variant already finished by the same octreg code is skipped.

--turn starts every run from the OCT and its mask turned by the rotation vector RX,RY,RZ in degrees about the centre of the
OCT image box. T_oct2mri.txt and result.json are then rewritten for the original OCT world (T @ turn), the files tied to the
turned header (oct2mri.lta, oct2mri_itk.txt, mri_in_oct.nii.gz) are removed, and no_search also writes its start to T_start.txt.

--start FILE gives no_search its start, a 4x4 OCT world -> MRI world pose, in place of the header orientation: the key H of a
JSON file (the start.json of bench/i46/baselines.py --near) or a text matrix. It excludes --turn.

The spacing variants give octreg the OCT voxel size of the file (or of --oct-spacing-um) scaled by their factor. Their
T_oct2mri.txt and result.json are rewritten for the OCT world of the file as for --turn, with the 4x4 from that world to
the one the run used as the key turn, and the scale per OCT axis and the OCT voxel size recomputed in it. The foreground
entries of result.json stay in the voxel size the run used. An --oct-mask given with a spacing variant must have the
OCT's array shape, since a mask on its own grid keeps the spacing of its file.

--params F applies the Params overrides of F, a JSON file or JSON text, on top of the variant's change, as octreg register
--params.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import dataclasses
import hashlib
import json
import sys
import tokenize
from io import StringIO
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import octreg  # noqa: E402
from octreg import features as Fe, geometry as G, io, preprocess as pp, refine as Rf, register as Reg  # noqa: E402
from octreg import search as Se                                                                    # noqa: E402
from octreg.params import Params                                                                   # noqa: E402

VARIANTS = {
    "no_outline": "the outline never used (outline_min_outside 2): every pose proposed by F",
    "outline_decides": "every pose proposed by S = (F + O) / 2 where the outline is used",
    "no_search": "no orientation search: the refinement from the header orientation with the image box centres aligned",
    "no_refine": "no refinement: the first search pose",
    "first_choice": "only the first search pose refined",
    "no_prior": "no scale and shear prior (lam 0)",
    "prior_0.5": "the scale and shear prior at a quarter of its weight (lam 0.5)",
    "prior_8": "the scale and shear prior at four times its weight (lam 8)",
    "rigid": "rigid refinement: the scales and shears held at zero",
    "signed": "the signed normalised gradient n instead of its tensor n n^T",
    "signed_neg": "the signed normalised gradient n instead of its tensor n n^T, with the OCT intensities negated",
    "tracefree": "the trace-free tensor n n^T - |n|^2 I / 3 instead of n n^T",
    "uncentred": "F without the subtraction of the channel means (the squared NGF of Cross-Sim-NGF, normalised)",
    "search_whole_mask": "in the search, the OCT gradients taken inside the whole specimen mask, not its erosion",
    "flatfield": "the MRI divided by its Gaussian blur (normalised convolution, σ 5 mm) inside the foreground",
    "signal_only": "the OCT specimen mask from the signal threshold alone (small components dropped, holes filled as in §1)",
    "spacing_p5": "the OCT voxel size given 5 % too large",
    "spacing_m5": "the OCT voxel size given 5 % too small",
    "mirrored": "the OCT array mirrored along its first axis: a mirror image of the block, which no rotation aligns",
}
__doc__ = __doc__.replace("{variants}", "\n".join(f"  {k:<18} {v}" for k, v in VARIANTS.items()))
REFINE, TENSOR, PREPARE_OCT, PREPARE_MRI = Rf.refine, Fe.orientation_tensor, Reg.prepare_oct, Reg.prepare_mri
COMPOSE, PLANES = Rf.compose, io.iter_planes
SPACING = {"spacing_p5": 1.05, "spacing_m5": 0.95}
FLAT_MM = 5.0


def header_start(mri_affine, mri_shape, oct_affine, oct_shape):
    """Pose with the header orientation and the two image box centres aligned, 4x4 OCT world to MRI world."""
    T = np.eye(4)
    T[:3, 3] = G.box(mri_affine, mri_shape)[0] - G.box(oct_affine, oct_shape)[0]
    return T


def fixed_start(T):
    """Stand-in for search.search that proposes only the pose T."""
    return lambda *args, **kwargs: [dict(T=np.asarray(T, float), F=float("nan"), O=None)]


def read_start(path):
    """A 4x4 OCT world -> MRI world pose: the key H of a JSON file, or a text matrix."""
    path = Path(path)
    T = np.array(json.loads(path.read_text(encoding="utf-8"))["H"] if path.suffix == ".json" else np.loadtxt(path), float)
    if T.shape != (4, 4):
        raise ValueError(f"{path}: not a 4x4 pose")
    return T


def no_refine(starts, mri, oct_, params, device):
    return [{"T": np.asarray(starts[0], float), "L": float("nan"), "F": [], "search_rank": 0}]


def first_choice(starts, mri, oct_, params, device):
    return REFINE(starts[:1], mri, oct_, params, device)


def tracefree(n):
    """In place of features.orientation_tensor: the trace-free tensor n n^T - |n|^2 I / 3 in the same six channels."""
    P = TENSOR(n)
    return torch.cat([P[:3] - P[:3].sum(0) / 3, P[3:]])


def negated_oct(fine, affine, params, mask=None):
    """register.prepare_oct with the base grid OCT intensities negated and the masks of the method."""
    (arr, m, A), measured, info = PREPARE_OCT(fine, affine, params, mask)
    return (-arr, m, A), measured, info


def flatfield_mri(vol, params=Params(), mask=None):
    """register.prepare_mri, the MRI divided by its Gaussian blur (normalised convolution, FLAT_MM) inside the foreground."""
    from scipy import ndimage
    (arr, fg, A), info = PREPARE_MRI(vol, params, mask)
    s, m = FLAT_MM / params.base_mm, fg.astype(np.float32)
    blur = ndimage.gaussian_filter(arr * m, s) / np.maximum(ndimage.gaussian_filter(m, s), 1e-3)
    return (np.where(fg, arr / np.maximum(blur, 1e-6), arr).astype(np.float32), fg, A), info


def signal_only_mask(fine, voxel_mm, params=Params()):
    """preprocess.specimen_mask from the signal threshold alone: the components and holes of §1, no texture rule."""
    chunks = lambda: (np.asarray(fine[z:z + pp._CHUNK], np.float32) for z in range(0, fine.shape[0], pp._CHUNK))
    ts = float(np.exp(pp.threshold(*pp._hist(lambda: (np.log(x[x > 0]) for x in chunks())))))
    out = pp._fill_planes(pp._components(np.asarray(fine) > ts, params.min_component))
    out &= np.asarray(fine) > 0
    return out, {"signal_threshold": ts, "texture_threshold": None, "volume_cm3": float(out.sum()) * voxel_mm ** 3 / 1e3}


def patched(method, *pairs):
    """A copy of a function of octreg.search with source replacements (old, new), each of which must occur once."""
    import inspect
    import textwrap
    src = textwrap.dedent(inspect.getsource(method))
    for old, new in pairs:
        if src.count(old) != 1:
            raise RuntimeError(f"{old!r} does not occur once in search.{method.__qualname__}")
        src = src.replace(old, new)
    ns = {}
    exec(compile(src, f"<ablate: {method.__qualname__}>", "exec"), vars(Se), ns)
    return ns[method.__name__]


def whole_mask_searcher():
    """search.Searcher.__init__ and score with the template gradients and their eps taken inside the whole specimen mask."""
    wb = "(w > 0.5).float()"
    return (patched(Se.Searcher.__init__, ("gradient_field(o, inner, A_O, sigma_mm, device), inner)",
                                           f"gradient_field(o, {wb}, A_O, sigma_mm, device), {wb})")),
            patched(Se.Searcher.score, ("G.nc_gradient(v * wt, wt, self.sig)", f"G.nc_gradient(v * {wb}, {wb}, self.sig)")))


def rigid_compose(r, t, ls, sh, c):
    """refine.compose with the scales and shears held at zero."""
    return COMPOSE(r, t, 0 * ls, 0 * sh, c)


def uncentred(a, b):
    """features.correlation without the subtraction of the channel means."""
    return (a * b).sum() / torch.sqrt((a * a).sum() * (b * b).sum() + 1e-24)


def uncentred_searcher():
    """search.Searcher.score with the centring terms of F dropped."""
    return patched(Se.Searcher.score, (" - (ST.reshape(-1, 1, 1, 1) * SI).sum(0) / N", ""),
                   (" - (SI * SI).sum(0) / N", ""), ("(STT - (ST * ST).sum() / N)", "STT"))


def mirrored_planes(paths):
    """io.iter_planes with the planes of the files in paths mirrored along the first array axis."""
    paths = {Path(p).resolve() for p in paths}

    def it(vol):
        flip = Path(vol.path).resolve() in paths
        for k, plane in PLANES(vol):
            yield k, (plane[::-1] if flip else plane)
    return it


def spacing_change(oct_path, spacing_um, factor):
    """The OCT voxel size (um, Z,Y,X) scaled by factor, and the 4x4 from the OCT world to the world of that voxel size."""
    v = io.load_volume(oct_path, spacing_um)
    um = tuple(round(float(x), 6) for x in (v.spacing_mm * 1000.0 * factor)[::-1])
    return um, io.load_volume(oct_path, um).affine @ np.linalg.inv(v.affine)


def turned(oct_paths, rotvec_deg, spacing_um=None):
    """Turn about the OCT image box centre and an io.load_volume that reads oct_paths with the turned header.
    -> (4x4 OCT world to turned OCT world, loader)"""
    load = io.load_volume
    vo = load(oct_paths[0], spacing_um)
    c = G.box(vo.affine, vo.shape)[0]
    turn = np.eye(4)
    turn[:3, :3] = Rotation.from_rotvec(np.radians(rotvec_deg)).as_matrix()
    turn[:3, 3] = c - turn[:3, :3] @ c

    def load_turned(path, spacing_um=None):
        v = load(path, spacing_um)
        return dataclasses.replace(v, affine=turn @ v.affine) if str(path) in oct_paths else v
    return turn, load_turned


@contextlib.contextmanager
def variant(name, oct_path, mri_path, oct_spacing_um=None, start=None, oct_mask=None):
    """Context with the change of variant `name` in force, yielding the Params of the run.
    oct_path, mri_path and oct_spacing_um give no_search its start, unless the pose `start` is given. oct_path and
    oct_mask are the files mirrored. The spacing variants change nothing here (main gives them their voxel size)."""
    P = Params()
    with contextlib.ExitStack() as st:
        patch = lambda obj, attr, new: st.enter_context(mock.patch.object(obj, attr, new))
        if name in SPACING or name == "base":
            pass
        elif name == "no_outline":
            P = dataclasses.replace(P, outline_min_outside=2.0)        # a share above 1: never met
        elif name == "outline_decides":
            patch(Reg, "search", patched(Se.search, ("(P.topk // 2, P.topk) if outline", "(P.topk, P.topk) if outline")))
        elif name == "no_search":
            if start is None:
                vo, vm = io.load_volume(oct_path, oct_spacing_um), io.load_volume(mri_path)
                start = header_start(vm.affine, vm.shape, vo.affine, vo.shape)
            patch(Reg, "search", fixed_start(start))
        elif name == "no_refine":
            patch(Reg, "refine", no_refine)
        elif name == "first_choice":
            patch(Reg, "refine", first_choice)
        elif name == "no_prior":
            P = dataclasses.replace(P, lam=0.0)
        elif name in ("prior_0.5", "prior_8"):
            P = dataclasses.replace(P, lam=float(name.split("_")[1]))
        elif name == "rigid":
            patch(Rf, "compose", rigid_compose)
        elif name == "uncentred":
            patch(Fe, "correlation", uncentred)
            patch(Se.Searcher, "score", uncentred_searcher())
        elif name == "mirrored":
            patch(io, "iter_planes", mirrored_planes([oct_path] + ([oct_mask] if oct_mask else [])))
        elif name in ("signed", "signed_neg"):
            patch(Fe, "orientation_tensor", lambda n: n)
            if name == "signed_neg":
                patch(Reg, "prepare_oct", negated_oct)
        elif name == "tracefree":
            patch(Fe, "orientation_tensor", tracefree)
        elif name == "search_whole_mask":
            init, score = whole_mask_searcher()
            patch(Se.Searcher, "__init__", init)
            patch(Se.Searcher, "score", score)
        elif name == "flatfield":
            patch(Reg, "prepare_mri", flatfield_mri)
        elif name == "signal_only":
            patch(pp, "specimen_mask", signal_only_mask)
        else:
            raise ValueError(f"unknown variant {name}")
        yield P


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("oct")
    ap.add_argument("mri")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--oct-mask")
    ap.add_argument("--oct-spacing-um", type=lambda s: tuple(float(x) for x in s.split(",")))
    ap.add_argument("--only", help="comma-separated variants (default all, 'base' runs the method itself)")
    ap.add_argument("--turn", type=lambda s: [float(x) for x in s.split(",")], help="rotation vector of the start, degrees")
    ap.add_argument("--start", type=Path, help="no_search: its start pose, OCT world -> MRI world, the key H of a JSON file "
                                                "or a text matrix")
    ap.add_argument("--params", metavar="F", help="Params overrides applied after the variant's change: a JSON file or text")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    names = a.only.split(",") if a.only else list(VARIANTS)
    unknown = [n for n in names if n not in VARIANTS and n != "base"]
    if unknown:
        ap.error(f"unknown variants {unknown}")
    if a.start and a.turn:
        ap.error("--start and --turn exclude each other")
    if a.turn and set(names) & set(SPACING):
        ap.error("--turn excludes the spacing variants")
    if a.oct_mask and set(names) & set(SPACING) and (io.load_volume(a.oct_mask, a.oct_spacing_um).shape
                                                     != io.load_volume(a.oct, a.oct_spacing_um).shape):
        ap.error("an --oct-mask with the spacing variants must have the OCT's array shape")
    start = read_start(a.start) if a.start else None
    over = json.loads(Path(a.params).read_text(encoding="utf-8") if Path(a.params).is_file() else a.params) if a.params else {}
    turn, load = turned([a.oct] + ([a.oct_mask] if a.oct_mask else []), a.turn, a.oct_spacing_um) if a.turn else (None, None)
    code = code_id()
    for name in names:
        out = a.out / name
        if made_by(out) == code:
            print(f"{name}: done", flush=True)
            continue
        if made_by(out):
            print(f"{name}: made by octreg code {made_by(out)}, run again with {code}", flush=True)
            (out / "code.txt").unlink()
        spacing, to_run = a.oct_spacing_um, turn                  # the voxel size given, the OCT world -> the world run
        if name in SPACING:
            spacing, to_run = spacing_change(a.oct, a.oct_spacing_um, SPACING[name])
        with contextlib.ExitStack() as st:
            if load is not None:
                st.enter_context(mock.patch.object(io, "load_volume", load))
            P = Params.from_dict({**st.enter_context(variant(name, a.oct, a.mri, a.oct_spacing_um, start,
                                                             a.oct_mask)).to_dict(), **over})
            res = Reg.register(a.oct, a.mri, out, spacing, a.oct_mask, None, P, a.device)
        if to_run is not None:
            saved = json.loads((out / "result.json").read_text(encoding="utf-8"))
            vo = io.load_volume(a.oct, a.oct_spacing_um)
            T = np.asarray(saved["T_oct2mri"], float) @ to_run
            T0 = np.asarray(saved["pose"]["T_search"], float) @ to_run
            saved["pose"]["T_search"] = T0
            saved["pose"]["scale_per_oct_axis"] = np.linalg.norm(T[:3, :3] @ (vo.affine[:3, :3] / vo.spacing_mm),
                                                                 axis=0).tolist()
            saved["inputs"]["oct"]["affine"], saved["inputs"]["oct"]["spacing_mm"] = vo.affine, vo.spacing_mm
            saved["inputs"]["oct_spacing_um"] = a.oct_spacing_um
            io.write_json({**saved, "T_oct2mri": T, "turn": to_run}, out / "result.json")
            io.write_transform_txt(T, out / "T_oct2mri.txt")
            for f in ("oct2mri.lta", "oct2mri_itk.txt", "mri_in_oct.nii.gz"):
                (out / f).unlink(missing_ok=True)
            if name == "no_search":
                io.write_transform_txt(T0, out / "T_start.txt")
        (out / "variant.json").write_text(json.dumps({"variant": name, "change": VARIANTS.get(name, "the method"),
                                                      "turn_deg": a.turn, "spacing_um": spacing,
                                                      "turn": None if to_run is None else to_run.tolist(),
                                                      "start_file": None if a.start is None else str(a.start.resolve()),
                                                      "start": None if start is None else start.tolist(),
                                                      "params": P.to_dict()}, indent=1), encoding="utf-8")
        (out / "code.txt").write_text(code + "\n", encoding="utf-8")
        print(f"{name}: search rank {res['pose']['search_rank']}, L {res['pose']['L']:.4f} -> {out}", flush=True)


def _tokens(src):
    """The tokens of src, an f-string as one STRING token as Python 3.11 and older give it."""
    off = [0]
    for line in StringIO(src):                         # line starts as tokenize counts them
        off.append(off[-1] + len(line))
    depth, fs, fe = 0, getattr(tokenize, "FSTRING_START", None), getattr(tokenize, "FSTRING_END", None)
    for t in tokenize.generate_tokens(StringIO(src).readline):
        if fs is not None and t.type == fs:
            depth += 1
            if depth == 1:
                start, line = t.start, t.line
        elif depth:
            if t.type == fe:
                depth -= 1
                if not depth:
                    text = src[off[start[0] - 1] + start[1]:off[t.end[0] - 1] + t.end[1]]
                    yield tokenize.TokenInfo(tokenize.STRING, text, start, t.end, line)
        else:
            yield t


def code_id():
    """Identity of the imported octreg code: the sha256 of its tokens without comments and docstrings, 12 hex digits."""
    h = hashlib.sha256()
    for f in sorted(Path(octreg.__file__).parent.glob("*.py")):
        src = f.read_text(encoding="utf-8")
        docs = {(n.body[0].value.lineno, n.body[0].value.col_offset) for n in ast.walk(ast.parse(src))
                if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body
                and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)
                and isinstance(n.body[0].value.value, str)}
        toks, skip = [], False
        for t in _tokens(src):
            if t.type == tokenize.STRING and (t.start[0], len(t.line[:t.start[1]].encode())) in docs:   # ast counts bytes
                skip = True
            elif not (t.type in (tokenize.COMMENT, tokenize.NL) or skip and t.type == tokenize.NEWLINE):
                toks.append(t.string)
                skip = False
        h.update(f.name.encode() + b"\0" + "\x1f".join(toks).encode() + b"\0")
    return h.hexdigest()[:12]


def made_by(out):
    """code_id in out/code.txt if out holds a finished run with a complete result.json, else None."""
    try:
        res = json.loads((Path(out) / "result.json").read_text(encoding="utf-8"))
        return (Path(out) / "code.txt").read_text(encoding="utf-8").strip() if "seconds" in res else None
    except (OSError, ValueError):
        return None


if __name__ == "__main__":
    main()

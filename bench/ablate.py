#!/usr/bin/env python3
"""Ablations of octreg: the method with one change per variant, run on one pair.

    python bench/ablate.py OCT MRI -o OUT [--oct-mask MASK] [--oct-spacing-um Z,Y,X] [--only A,B] [--turn RX,RY,RZ]
                           [--device cuda]

Each variant changes one function, constant or Params value for its run:

{variants}

OUT/<variant>/ holds the octreg register outputs, variant.json (the change, the Params and the turn) and code.txt (the
code_id of the octreg code that made it). A variant already finished by the same octreg code is skipped.

--turn starts every run from the OCT and its mask turned by the rotation vector RX,RY,RZ in degrees about the centre of the
OCT image box. T_oct2mri.txt and result.json are then rewritten for the original OCT world (T @ turn), the files tied to the
turned header (oct2mri.lta, oct2mri_itk.txt, mri_in_oct.nii.gz) are removed, and no_search also writes its start to T_start.txt.
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
from octreg import deform as D, features as Fe, geometry as G, io, refine as Rf, register as Reg, search as Se  # noqa: E402
from octreg.params import Params                                                                              # noqa: E402

VARIANTS = {
    "no_outline": "the outline never used (outline_min_outside 2): every pose proposed by F",
    "outline_decides": "every pose proposed by S = (F + O) / 2 where the outline is used",
    "no_search": "no orientation search: the refinement from the header orientation with the image box centres aligned",
    "no_refine": "no refinement: the first search pose",
    "first_choice": "only the first search pose refined",
    "no_prior": "no scale and shear prior (lam 0)",
    "signed": "the signed normalised gradient n instead of its tensor n n^T",
    "signed_neg": "the signed normalised gradient n instead of its tensor n n^T, with the OCT intensities negated",
    "tracefree": "the trace-free tensor n n^T - |n|^2 I / 3 instead of n n^T",
    "search_whole_mask": "in the search, the OCT gradients taken inside the whole specimen mask, not its erosion",
    "df_free": "the deformation without the strain bound (deform.STRAIN_WEIGHT 0)",
}
__doc__ = __doc__.replace("{variants}", "\n".join(f"  {k:<18} {v}" for k, v in VARIANTS.items()))
REFINE, TENSOR, PREPARE_OCT = Rf.refine, Fe.orientation_tensor, Reg.prepare_oct


def header_start(mri_affine, mri_shape, oct_affine, oct_shape):
    """Pose with the header orientation and the two image box centres aligned, 4x4 OCT world to MRI world."""
    T = np.eye(4)
    T[:3, 3] = G.box(mri_affine, mri_shape)[0] - G.box(oct_affine, oct_shape)[0]
    return T


def fixed_start(T):
    """Stand-in for search.search that proposes only the pose T."""
    return lambda *args, **kwargs: [dict(T=np.asarray(T, float), F=float("nan"), O=None)]


def search_by_S(mri, mri_mask, mri_affine, oct_, oct_w, oct_valid, oct_affine, params=Params(), device="cuda", outline=True):
    """search.search with the translations and the topk chosen by S = (F + O) / 2 when the outline is used."""
    if not outline:
        return Se.search(mri, mri_mask, mri_affine, oct_, oct_w, oct_valid, oct_affine, params, device, outline)
    P = params
    s = Se.Searcher(mri, mri_mask, mri_affine, oct_, oct_w, oct_valid, oct_affine, P.search_sigma_mm, P.erode_mm, device)
    Rs = G.rotations(P.n_rot)
    found = []
    for i, R in enumerate(Rs):
        Fm, Om = (x.reshape(-1) for x in s.score(R))
        S = (Fm + Om) / 2
        j = int(torch.argmax(S.masked_fill(~s.valid.reshape(-1), float("-inf"))))
        found.append((float(S[j]), i, j, float(Fm[j]), float(Om[j])))
    kept = []
    for _, i, j, f, o in sorted(found, key=lambda e: -e[0]):
        T = s.pose(Rs[i], j)
        centre = G.apply_affine(T, s.c_o)
        if not any(np.linalg.norm(centre - c) < P.nms_mm and Se.angle_deg(Rs[i], R) < P.nms_deg for _, c, R in kept):
            kept.append((dict(T=T, F=f, O=o), centre, Rs[i]))
            if len(kept) == P.topk:
                break
    return [k[0] for k in kept]


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
    (arr, m, A), valid, info = PREPARE_OCT(fine, affine, params, mask)
    return (-arr, m, A), valid, info


def patched(method, *pairs):
    """A copy of a search.Searcher method with source replacements (old, new), each of which must occur once."""
    import inspect
    import textwrap
    src = textwrap.dedent(inspect.getsource(method))
    for old, new in pairs:
        if src.count(old) != 1:
            raise RuntimeError(f"{old!r} does not occur once in search.Searcher.{method.__name__}")
        src = src.replace(old, new)
    ns = {}
    exec(compile(src, f"<ablate: Searcher.{method.__name__}>", "exec"), vars(Se), ns)
    return ns[method.__name__]


def whole_mask_searcher():
    """search.Searcher.__init__ and score with the template gradients and their eps taken inside the whole specimen mask."""
    wb = "(w > 0.5).float()"
    return (patched(Se.Searcher.__init__, ("G.nc_gradient(o * inner, inner,", f"G.nc_gradient(o * {wb}, {wb},"),
                    ("[inner > 0].median()", "[w > 0.5].median()")),
            patched(Se.Searcher.score, ("G.nc_gradient(v * wt, wt, self.sig)", f"G.nc_gradient(v * {wb}, {wb}, self.sig)")))


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
def variant(name, oct_path, mri_path, oct_spacing_um=None):
    """Context with the change of variant `name` in force, yielding the Params of the run.
    oct_path, mri_path and oct_spacing_um give no_search its start."""
    P = Params()
    with contextlib.ExitStack() as st:
        patch = lambda obj, attr, new: st.enter_context(mock.patch.object(obj, attr, new))
        if name == "no_outline":
            P = dataclasses.replace(P, outline_min_outside=2.0)        # a share above 1: never met
        elif name == "outline_decides":
            patch(Reg, "search", search_by_S)
        elif name == "no_search":
            vo, vm = io.load_volume(oct_path, oct_spacing_um), io.load_volume(mri_path)
            patch(Reg, "search", fixed_start(header_start(vm.affine, vm.shape, vo.affine, vo.shape)))
        elif name == "no_refine":
            patch(Reg, "refine", no_refine)
        elif name == "first_choice":
            patch(Reg, "refine", first_choice)
        elif name == "no_prior":
            P = dataclasses.replace(P, lam=0.0)
        elif name in ("signed", "signed_neg"):
            patch(Fe, "orientation_tensor", lambda n: n)
            patch(Se, "orientation_tensor", lambda n: n)
            if name == "signed_neg":
                patch(Reg, "prepare_oct", negated_oct)
        elif name == "tracefree":
            patch(Fe, "orientation_tensor", tracefree)
            patch(Se, "orientation_tensor", tracefree)
        elif name == "search_whole_mask":
            init, score = whole_mask_searcher()
            patch(Se.Searcher, "__init__", init)
            patch(Se.Searcher, "score", score)
        elif name == "df_free":
            patch(D, "STRAIN_WEIGHT", 0.0)
        elif name != "base":
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
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    names = a.only.split(",") if a.only else list(VARIANTS)
    unknown = [n for n in names if n not in VARIANTS and n != "base"]
    if unknown:
        ap.error(f"unknown variants {unknown}")
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
        with contextlib.ExitStack() as st:
            if load is not None:
                st.enter_context(mock.patch.object(io, "load_volume", load))
            P = st.enter_context(variant(name, a.oct, a.mri, a.oct_spacing_um))
            res = Reg.register(a.oct, a.mri, out, a.oct_spacing_um, a.oct_mask, None, P, a.device)
        if turn is not None:
            saved = json.loads((out / "result.json").read_text(encoding="utf-8"))
            T = np.asarray(saved["T_oct2mri"], float) @ turn
            T0 = np.asarray(saved["pose"]["T_search"], float) @ turn
            saved["pose"]["T_search"] = T0
            saved["inputs"]["oct"]["affine"] = io.load_volume(a.oct, a.oct_spacing_um).affine
            io.write_json({**saved, "T_oct2mri": T, "turn": turn}, out / "result.json")
            io.write_transform_txt(T, out / "T_oct2mri.txt")
            for f in ("oct2mri.lta", "oct2mri_itk.txt", "mri_in_oct.nii.gz"):
                (out / f).unlink(missing_ok=True)
            if name == "no_search":
                io.write_transform_txt(T0, out / "T_start.txt")
        (out / "variant.json").write_text(json.dumps({"variant": name, "change": VARIANTS.get(name, "the method"),
                                                      "turn_deg": a.turn, "turn": None if turn is None else turn.tolist(),
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

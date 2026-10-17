#!/usr/bin/env python3
"""Ablations of octreg: the method with one change per variant, each run through octreg.register.register on one pair.

    python bench/ablate.py OCT MRI -o OUT [--oct-mask MASK] [--oct-spacing-um Z,Y,X] [--only A,B] [--turn RX,RY,RZ]
                           [--device cuda]

A variant replaces one function or constant of the package for its run, or one value of Params:

  no_outline     S = F in the search
  outline_refine the refinement on S = (F + O) / 2 instead of F (where the outline is used)
  no_search      no orientation search: the refinement from one start, the header orientation with the centres of the two
                 search grids aligned
  no_refine      no refinement: the best search pose
  first_choice   only the best search pose refined (the others are not)
  no_prior       no scale and shear prior (lam 0)
  signed         the signed normalised gradient n instead of its tensor P(n) (normalised gradient fields), OCT polarity +1
  signed_neg     the same with the OCT intensities negated (polarity -1)
  full_tensor    the full tensor n n^T instead of its trace-free part, so <P, P'> = (n . n')^2 (squared normalised gradient
                 fields)
  df_interior    the deformation on the score of §4: OCT points inside the eroded specimen mask, MRI tensors inside the
                 foreground, so the tissue surface does not enter
  df_free        the deformation without the strain bound

OUT/<variant>/ holds the outputs of octreg register and variant.json (the change). A finished variant is skipped.

--turn starts every run from the OCT world turned by the rotation vector (RX, RY, RZ) in degrees about the centre of the OCT
image box: the OCT and its mask file are read with the turned header. T_oct2mri.txt and T_mri2oct.txt are then rewritten for
the original OCT world (T @ turn), as is T_oct2mri in result.json (with the turn), the files tied to the turned OCT header
(oct2mri.lta, oct2mri_itk.txt, mri_in_oct.nii.gz) are removed, and variant.json holds the turn. The no_search run also writes
T_start.txt, the start it is refined from (its search pose) as a pose of the original OCT world.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg import deform as D, features as Fe, geometry as G, refine as Rf, register as Reg, search as Se   # noqa: E402
from octreg.params import Params                                                                         # noqa: E402

VARIANTS = {
    "no_outline": "S = F in the search",
    "outline_refine": "the refinement on S = (F + O) / 2 instead of F",
    "no_search": "no orientation search: the refinement from the header orientation, search grid centres aligned",
    "no_refine": "no refinement: the best search pose",
    "first_choice": "only the best search pose refined",
    "no_prior": "no scale and shear prior (lam 0)",
    "signed": "signed normalised gradients instead of their tensor, OCT polarity +1",
    "signed_neg": "signed normalised gradients instead of their tensor, OCT polarity -1",
    "full_tensor": "the full tensor n n^T instead of its trace-free part (squared normalised gradient fields)",
    "df_interior": "the deformation on the score of §4, without the tissue surface",
    "df_free": "the deformation without the strain bound",
}


def one_start(mri, mri_mask, mri_affine, oct_, oct_w, oct_valid, oct_affine, params, device, outline):
    """In place of search.search: the header orientation with the centres of the two search grids aligned."""
    T = np.eye(4)
    T[:3, 3] = G.box(mri_affine, np.shape(mri))[0] - G.box(oct_affine, np.shape(oct_))[0]
    nan = float("nan")                                          # no search score
    return [dict(T=T, S=nan, F=nan, O=None)], dict(top1=nan, top2=None, n_orientations=0, seconds=0.0)


def refine_on_S(starts, mri, oct_, params, device):
    """In place of refine.refine: the same passes on S = (F + O) / 2 where the outline is used."""
    if not Reg.outline_used(oct_[1], oct_[2] > 0.5, mri[1], params)[0]:
        return REFINE(starts, mri, oct_, params, device)
    out = Se.Outline(mri[1], mri[2], oct_[1], oct_[2], oct_[3], device)
    with mock.patch.object(Rf, "score", lambda T, tpl, mt: (SCORE(T, tpl, mt) + out(T)) / 2):
        return REFINE(starts, mri, oct_, params, device)


def no_refine(starts, mri, oct_, params, device):
    """In place of refine.refine: the best search pose unchanged."""
    return [{"T": np.asarray(starts[0], float), "L": float("nan"), "F": [], "search_rank": 0}]


def first_choice(starts, mri, oct_, params, device):
    """In place of refine.refine: only the best search pose refined."""
    return REFINE(starts[:1], mri, oct_, params, device)


def full_tensor(n):
    """features.tracefree without the trace removal: six channels of n n^T, <P, P'> = (n . n')^2."""
    r2 = 2.0 ** 0.5
    return Fe.torch.stack([n[0] * n[0], n[1] * n[1], n[2] * n[2], r2 * n[0] * n[1], r2 * n[0] * n[2], r2 * n[1] * n[2]])


def negated_oct(fine, affine, params, mask=None):
    """register.prepare_oct with the OCT intensities of the base grid negated (the masks are those of the method)."""
    (arr, m, A), valid, info = PREPARE_OCT(fine, affine, params, mask)
    return (-arr, m, A), valid, info


def tpl_inside(arr, mask, affine, sigma_mm, points, stride=2, device="cuda", eps_mask=None):
    """deform's OCT template on the points and gradients of §4: inside the specimen mask, eroded by erode_mm."""
    h = float(np.linalg.norm(np.asarray(affine)[:3, 0]))
    return Fe.OCTTemplate(arr, eps_mask, affine, sigma_mm, Fe.eroded(eps_mask, Params().erode_mm, h), stride, device)


def mri_inside(arr, mask, affine, sigma_mm, device="cuda", eps_mask=None):
    """deform's MRI tensors inside the foreground, as in §4."""
    return Fe.MRITensors(arr, eps_mask, affine, sigma_mm, device)


def turned(oct_paths, rotvec_deg):
    """A turn about the OCT image box centre (4x4, OCT world -> turned OCT world) and a stand-in for io.load_volume that reads
    the files of oct_paths with the turned header."""
    from scipy.spatial.transform import Rotation
    from octreg import io
    load = io.load_volume
    vo = load(oct_paths[0], ARGS_SPACING[0])
    c = G.box(vo.affine, vo.shape)[0]
    turn = np.eye(4)
    turn[:3, :3] = Rotation.from_rotvec(np.radians(rotvec_deg)).as_matrix()
    turn[:3, 3] = c - turn[:3, :3] @ c

    def load_turned(path, spacing_um=None):
        v = load(path, spacing_um)
        if str(path) in oct_paths:
            v = dataclasses.replace(v, affine=turn @ v.affine)
        return v
    return turn, load_turned


REFINE, SCORE, PREPARE_OCT = Rf.refine, Rf.score, Reg.prepare_oct
ARGS_SPACING = [None]


@contextlib.contextmanager
def variant(name):
    """The change of variant `name` in force. -> Params of the run."""
    P = Params()
    with contextlib.ExitStack() as st:
        patch = lambda obj, attr, new: st.enter_context(mock.patch.object(obj, attr, new))
        if name == "no_outline":
            P = dataclasses.replace(P, outline_min_outside=2.0)        # a share above 1: never met
        elif name == "no_prior":
            P = dataclasses.replace(P, lam=0.0)
        elif name == "outline_refine":
            patch(Reg, "refine", refine_on_S)
        elif name == "no_search":
            patch(Reg, "search", one_start)
        elif name == "no_refine":
            patch(Reg, "refine", no_refine)
        elif name == "first_choice":
            patch(Reg, "refine", first_choice)
        elif name in ("signed", "signed_neg"):
            patch(Fe, "tracefree", lambda n: n)
            patch(Se, "tracefree", lambda n: n)
            if name == "signed_neg":
                patch(Reg, "prepare_oct", negated_oct)
        elif name == "full_tensor":
            patch(Fe, "tracefree", full_tensor)
            patch(Se, "tracefree", full_tensor)
        elif name == "df_interior":
            patch(D, "OCTTemplate", tpl_inside)
            patch(D, "MRITensors", mri_inside)
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
    ap.add_argument("--only", help="comma-separated variants (default all; 'base' runs the method itself)")
    ap.add_argument("--turn", type=lambda s: [float(x) for x in s.split(",")], help="rotation vector of the start, degrees")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    ARGS_SPACING[0] = a.oct_spacing_um
    names = a.only.split(",") if a.only else list(VARIANTS)
    for name in names:
        out = a.out / name
        if (out / "variant.json").is_file():
            print(f"{name}: done", flush=True)
            continue
        with contextlib.ExitStack() as st:
            P = st.enter_context(variant(name))
            turn = None
            if a.turn:
                from octreg import io
                turn, load = turned([a.oct] + ([a.oct_mask] if a.oct_mask else []), a.turn)
                st.enter_context(mock.patch.object(io, "load_volume", load))
            res = Reg.register(a.oct, a.mri, out, a.oct_spacing_um, a.oct_mask, None, P, a.device)
        if turn is not None:
            from octreg import io
            T = np.asarray(res["T_oct2mri"], float) @ turn
            io.write_transform_txt(T, out / "T_oct2mri.txt")
            io.write_transform_txt(np.linalg.inv(T), out / "T_mri2oct.txt")
            for f in ("oct2mri.lta", "oct2mri_itk.txt", "mri_in_oct.nii.gz"):
                (out / f).unlink(missing_ok=True)
            saved = json.loads((out / "result.json").read_text(encoding="utf-8"))
            io.write_json({**saved, "T_oct2mri": T, "turn": turn}, out / "result.json")
            if name == "no_search":
                io.write_transform_txt(np.asarray(res["pose"]["T_search"], float) @ turn, out / "T_start.txt")
        (out / "variant.json").write_text(json.dumps({"variant": name, "change": VARIANTS.get(name, "the method"),
                                                      "turn_deg": a.turn, "turn": None if turn is None else turn.tolist(),
                                                      "params": P.to_dict()}, indent=1), encoding="utf-8")
        print(f"{name}: S {res['pose']['S']:.4f} -> {out}", flush=True)


if __name__ == "__main__":
    main()

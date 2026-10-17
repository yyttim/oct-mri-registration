#!/usr/bin/env python3
"""Write bench/results/I46/tables.md from the JSON files in bench/results/I46, and for the start rows and the scales of R
from the inputs of bench/i46/prepare.py.

    python bench/i46/summarize.py

    octreg.json                                 the main runs of octreg (run.py collect)
    baselines.json, baselines_search.json       the standard tools at their documented settings and with their own search
    baselines_mask.json, baselines_near.json    the tools with the OCT tissue as moving mask, and from the start near R
    ablation.json, starts.json                  analyses.py collect
    reference_check.json, crops.json, context.json   the reference, the crops and the contrast of the images

The distances come from common.error.
"""
from __future__ import annotations

import json
import re
import sys
import textwrap
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BASIN, CONDITIONS, FLIPS, OCT, PRIMARY, RESULTS, SEEDS, TIERS, crop_file, crop_name, error,  # noqa: E402
                    reference)

INF = float("inf")


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def cp(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return f"{k}/{n} ({100 * lo:.0f} to {100 * hi:.0f} %)"


def med(x):
    x = sorted(x)
    if not x:
        return None
    n = len(x)
    return x[n // 2] if n % 2 else 0.5 * (x[n // 2 - 1] + x[n // 2])


def fmt(v, nd=2):
    return "none" if v is None else "beyond" if v == INF else f"{v:.{nd}f}"


def rng(x, nd=2):
    x = [v for v in x if v is not None]
    if not x:
        return "none"
    lo, hi = min(x), max(x)
    return fmt(lo, nd) if fmt(lo, nd) == fmt(hi, nd) else f"{fmt(lo, nd)} to {fmt(hi, nd)}"


def para(text):
    """text as lines of at most 124 characters, never broken inside "k = 0", and an empty line"""
    return [line.replace(" ", " ") for line in textwrap.wrap(text.replace(" = ", " = "), 124)] + [""]


def dist(x):
    return x["mean_mm"] if "mean_mm" in x else INF


def counts(rows):
    n = len(rows)
    d = [(dist(x), x.get("rotation_deg", 180.0)) for x in rows]
    return [cp(sum(m <= TIERS[0] for m, _ in d), n), cp(sum(m <= TIERS[1] for m, _ in d), n),
            cp(sum(m <= BASIN[0] and a <= BASIN[1] for m, a in d), n)]


def label(image, cond, masked=True):
    m, s = CONDITIONS[cond]
    return (f"{FLIPS[image][1]}° FLASH, {cond} (margin {m:g} mm, shift up to {s:g} mm)"
            + ("" if masked else ", OCT without --oct-mask"))


def conditions():
    c = [(im, "C3", True) for im in FLIPS] + [(PRIMARY, cond, True) for cond in CONDITIONS if cond != "C3"]
    return c + [(PRIMARY, "C3", False)]


def start_pose(crop):
    """The pose before any registration: the header orientation of the OCT with the centres of the two image boxes aligned."""
    o, m = nib.load(str(OCT)), nib.load(str(crop_file(crop)))
    c = lambda img: (img.affine @ np.r_[(np.array(img.shape[:3]) - 1) / 2, 1])[:3]
    T = np.eye(4)
    T[:3, 3] = c(m) - c(o)
    return error(T)


def reference_section(ref, ctx):
    out = ["## Reference", ""]
    lab = (ref or {}).get("labels", {})
    a = lab.get("A12")
    if a:
        th = " / ".join(f"{t['mean_mm']:.2f}" for t in a["by_depth_third"])
        out += para(
            "R is the authors' registration, from the OCT world to the MRI world in the test's OCT frame. The label-only fit "
            f"L (reference_check.py, an affine fitted to the vessels alone) lies {a['mean_mm']:.3f} mm from R on average "
            f"(p95 {a['p95_mm']:.2f} mm, rotation {a['rotation_deg']:.2f}°), {th} mm in the upper, middle and deepest third "
            "of the OCT depth, so distances below 0.30 mm are not interpreted.")
    if ctx:
        out += para("Grey/white contrast-to-noise ratio of the images inside the box of the OCT: " + ", ".join(
                        f"{FLIPS[k][1]}° FLASH {v:.2f}" if k in FLIPS else f"T1 map {v:.2f}"
                        for k, v in ctx.get("grey_white_cnr", {}).items() if k != "box_voxels") + ".")
    return out


def octreg_section(rows):
    out = ["## octreg", ""] + para("octreg at its default parameters.")
    out += ["| condition | <= 0.30 mm | <= 1 mm | <= 2.5 mm and <= 10° | distance from R, median (range) (mm) "
            "| from L, median (mm) | rotation from R (°) |", "|---|---|---|---|---|---|---|"]
    for im, cond, masked in conditions():
        r = [x for x in rows if x["image"] == im and x["cond"] == cond and x["oct_mask"] == masked]
        if not r:
            continue
        d = [x["mean_mm"] for x in r]
        out.append(f"| {label(im, cond, masked)} | " + " | ".join(counts(r))
                   + f" | {fmt(med(d))} ({rng(d)}) | {fmt(med([x.get('mean_L_mm') for x in r if 'mean_L_mm' in x]))} | "
                   f"{rng([x['rotation_deg'] for x in r], 1)} |")
    if rows and "rotation_axis" in rows[0]:
        ax = {}
        for im in FLIPS:
            a = np.array([x["rotation_axis"] for x in rows if x["image"] == im and x["cond"] == "C3" and x["oct_mask"]])
            if len(a):
                m = a.mean(0)
                ax[im] = (m / np.linalg.norm(m)).round(2).tolist()
        sc = np.array([x["scales"] for x in rows if "scales" in x])
        out += [""] + para(
            "Rotation axis of T R^-1, mean over the C3 runs with --oct-mask of each image (MRI world): "
            + ", ".join(f"{FLIPS[k][1]}° FLASH {v}" for k, v in ax.items())
            + f". Scale along the OCT axes over all runs: {rng(sc[:, 0], 3)}, {rng(sc[:, 1], 3)} and {rng(sc[:, 2], 3)}, and "
            f"of R: {', '.join(f'{x:.3f}' for x in np.linalg.norm(reference()[:3, :3], axis=0))}.")
    dv = [x["deform"] for x in rows if x["oct_mask"] and x.get("deform") and x["deform"].get("field_median_mm") is not None]
    if dv:
        out += ([""] if out[-1] else []) + para(
            "The deformation of §5, scored by the vessel check on the labels that the affine and the affine with the field both "
            f"place inside (common.vessel_field): the median distance is lower with the field on {sum(v['field_median_mm'] < v['affine_median_mm'] for v in dv)} "
            f"of the {len(dv)} runs with --oct-mask, median over the runs {fmt(med([v['affine_median_mm'] for v in dv]), 3)} mm through the affine "
            f"and {fmt(med([v['field_median_mm'] for v in dv]), 3)} mm with the field.")
    return trim(out)


def per_crop_section(rows, crops):
    out = ["## octreg per crop: C3 on the 10° FLASH", ""] + para(
        "The distance of octreg's affine from R and from L (mm), and the distance from the centre of the crop to the centre "
        "of the OCT tissue placed by R (crops.json).")
    out += ["| k | crop centre to tissue centre (mm) | from R | from L |", "|---|---|---|---|"]
    for k in range(SEEDS):
        name = crop_name(PRIMARY, "C3", k)
        c = next((x for x in crops if x["name"] == name), None)
        x = next((x for x in rows if x["name"] == name), {})
        out.append(f"| {k} | {fmt(c['crop_centre_to_tissue_centre_mm']) if c else ''} | {fmt(x.get('mean_mm'))} | "
                   f"{fmt(x.get('mean_L_mm'))} |")
    return out


def starts_section(st):
    out = ["## Start test on flip1_C3_k0", ""] + para(
        "The OCT is turned about the centre of its image box, by 90° and 180° about each array axis (i90 to k180) and by 10 "
        "rotations drawn uniformly on SO(3) (rand00 to rand09, seed 4646), and octreg runs from each turned start (analyses.py "
        "starts). The start column gives the distance and rotation from R of the turned OCT with the centres of the two image "
        "boxes aligned.")
    for group, pre in (("turned", ("i", "j", "k")), ("random", ("rand",))):
        r = [x for x in st if x["start"].startswith(pre)]
        if r:
            out += para(f"{group} starts: within 1 mm {cp(sum(x['mean_mm'] <= 1 for x in r), len(r))}, distance from R "
                        f"{rng([x['mean_mm'] for x in r])} mm, rotation {rng([x['rotation_deg'] for x in r], 1)}°.")
    out += ["| start | the start: distance from R (mm), rotation (°) | octreg (mm) |", "|---|---|---|"]
    for x in st:
        out.append(f"| {x['start']} | {x['start_pose']['mean_mm']:.1f}, {x['start_pose']['rotation_deg']:.0f} | {x['mean_mm']:.2f} |")
    return out


def tools_section(rows, bl, bls):
    out = ["## Standard tools", ""] + para(
        "Each tool at its documented setting (bench/baselines/standard_tools.py), and the three tools that have a global "
        "rotation search with that search turned on, on the C3 crops of the 10° FLASH. The start is the pose before any "
        "registration: the header orientation of the OCT with the centres of the two image boxes aligned.")
    out += ["| method | setting | <= 1 mm | <= 2.5 mm and <= 10° | tool errors | distance from R, median (mm) | rotation, "
            "median (°) | time, median (s) |", "|---|---|---|---|---|---|---|---|"]
    names = [crop_name(PRIMARY, "C3", k) for k in range(SEEDS)]
    st = [start_pose(n) for n in names]
    out.append(f"| the start | no registration | {counts(st)[1]} | {counts(st)[2]} | 0 | {fmt(med([x['mean_mm'] for x in st]))} "
               f"({rng([x['mean_mm'] for x in st], 1)}) | {fmt(med([x['rotation_deg'] for x in st]), 0)} | |")
    groups = [("octreg", "its default parameters", [x for x in rows if x["oct_mask"] and x["name"] in names])]
    for tr in (bl or [], bls or []):
        for t in dict.fromkeys(x["tool"] for x in tr):
            r = [x for x in tr if x["tool"] == t and x["crop"] in names]
            groups.append((t, r[0]["setting"], r))
    for method, setting, r in groups:
        ok = [x for x in r if "mean_mm" in x]
        out.append(f"| {method} | {setting} | {counts(r)[1]} | {counts(r)[2]} | {len(r) - len(ok)} | "
                   f"{fmt(med([dist(x) for x in r]))} | {fmt(med([x.get('rotation_deg', INF) for x in r]), 0)} | "
                   f"{fmt(med([x['seconds'] for x in ok if x.get('seconds') is not None]), 0)} |")
    if bl:
        out += [""] + para(f"Over all {len(bl)} runs at the documented settings, on the crops of this table and of the next, the "
                           f"median rotation from R is {fmt(med([x.get('rotation_deg', INF) for x in bl]), 0)}°.")[:-1]
    others = [im for im in FLIPS if im != PRIMARY]
    k0 = [("octreg", {x["name"]: x for x in rows})]
    k0 += [(t, {x["crop"]: x for x in bl if x["tool"] == t}) for t in dict.fromkeys(x["tool"] for x in bl or [])]
    out += [""] + para("The same on the C3 k = 0 crop of the 20° and 30° FLASH: distance from R (mm).")
    out += ["| method | " + " | ".join(f"{FLIPS[im][1]}° FLASH" for im in others) + " |", "|---|" + "---|" * len(others)]
    for method, d in k0:
        out.append(f"| {method} | " + " | ".join(fmt(dist(d[f'{im}_C3_k0'])) if f"{im}_C3_k0" in d else "" for im in others) + " |")
    chk = [x for x in (bl or []) + (bls or []) if "check_r" in x]
    if chk:
        out += [""] + para("Convention check on the k = 0 crops: each converted transform against the tool's own resampling of "
                           "the OCT, Pearson r over the voxels where either is non-zero, and the Dice of the two supports.")
        out += ["| tool | setting | crop | r | Dice | note |", "|---|---|---|---|---|---|"]
        for x in chk:
            out.append(f"| {x['tool']} | {x['setting']} | {x['crop']} | "
                       + ("| |" if x["check_r"] is None else f"{x['check_r']:.2f} | {x['check_dice']:.2f} |")
                       + f" {x.get('check_note', '')} |")
    return out


TOOL_HEAD = ["| method | setting | <= 1 mm | <= 2.5 mm and <= 10° | tool errors | distance from R, median (mm) | rotation, "
             "median (°) | time, median (s) |", "|---|---|---|---|---|---|---|---|"]


def tool_row(method, setting, r, spread=False):
    """One row of TOOL_HEAD over the runs r, the median distance with its range if spread."""
    ok = [x for x in r if "mean_mm" in x]
    d = [dist(x) for x in r]
    return (f"| {method} | {setting} | {counts(r)[1]} | {counts(r)[2]} | {len(r) - len(ok)} | "
            f"{fmt(med(d))}{f' ({rng(d)})' if spread else ''} | {fmt(med([x.get('rotation_deg', INF) for x in r]), 0)} | "
            f"{fmt(med([x['seconds'] for x in ok if x.get('seconds') is not None]), 0)} |")


def check_table(rows):
    chk = [x for x in rows if "check_r" in x]
    if not chk:
        return []
    out = para("Convention check on the k = 0 crop: each converted transform against the tool's own resampling of the OCT "
               "it ran on, Pearson r over the voxels where either is non-zero, and the Dice of the two supports.")
    out += ["| tool | setting | crop | r | Dice | note |", "|---|---|---|---|---|---|"]
    for x in chk:
        out.append(f"| {x['tool']} | {x['setting']} | {x['crop']} | "
                   + ("| |" if x["check_r"] is None else f"{x['check_r']:.2f} | {x['check_dice']:.2f} |")
                   + f" {x.get('check_note', '')} |")
    none = [x["tool"] for x in rows if x["crop"].endswith("_C3_k0") and "mean_mm" not in x]
    return out + ([""] + para(f"No check for {', '.join(none)}: the run on the k = 0 crop ended without a transform.")
                  if none else [])


def trim(lines):
    """lines without the empty lines at their end"""
    while lines and lines[-1] == "":
        lines = lines[:-1]
    return lines


def ks(rows):
    """The crop numbers of rows, a full run of consecutive numbers as "a to b"."""
    k = sorted(int(x["crop"].rsplit("_k", 1)[1]) for x in rows)
    return f"{k[0]} to {k[-1]}" if len(k) > 2 and k == list(range(k[0], k[-1] + 1)) else ", ".join(map(str, k))


def errors_text(rows):
    """The tool errors of rows, by tool: the crops and the log lines that name the error."""
    by = {}
    for x in rows:
        if "mean_mm" not in x:
            by.setdefault(x["tool"], []).append(x)
    parts = []
    for t, r in by.items():
        msgs = list(dict.fromkeys(re.sub(r"^Description:\s*", "", x.get("log_error") or x["error"][0]) for x in r))
        parts.append(f"{t} ended without a transform on k = {ks(r)}, its log: " + " or ".join(f'"{m}"' for m in msgs) + ".")
    return para("Tool errors. " + " ".join(parts)) if parts else []


def mask_section(bl, blm):
    """The tools with the OCT tissue as the moving mask (baselines.py --mask), each under its unmasked row."""
    names = [crop_name(PRIMARY, "C3", k) for k in range(SEEDS)]
    out = ["## Standard tools with a moving mask", ""] + para(
        "Each tool whose options take a mask of the moving image, at its documented setting with the OCT tissue (the voxels "
        "above 0 of the moving image) as that mask, on the C3 crops of the 10° FLASH. Each masked row follows the tool's "
        "unmasked row. mri_synthmorph register has no mask option. Times are wall clock, of the masked runs from another "
        "queue than the unmasked runs, so the two do not compare.")
    out += TOOL_HEAD
    same = {}
    for t in dict.fromkeys(x["tool"] for x in blm):
        base = t.removesuffix("_mask")
        r0 = [x for x in bl if x["tool"] == base and x["crop"] in names]
        r1 = [x for x in blm if x["tool"] == t and x["crop"] in names]
        out += [tool_row(base, r0[0]["setting"], r0), tool_row(t, r1[0]["setting"], r1)]
        d0 = {x["crop"]: x.get("mean_mm") for x in r0}
        same[t] = (sum(x["crop"] in d0 and x.get("mean_mm") == d0[x["crop"]] for x in r1), len(r1))
    out += [""] + para(
        "Masked runs that end at the distance from R of the unmasked run (to 0.1 µm), or in a tool error where it does: "
        + ", ".join(f"{t} {k}/{n}" for t, (k, n) in same.items()) + ".")
    full = [t for t, (k, n) in same.items() if k == n]
    if "reg_aladin_mask" in full:
        out += para("NiftyReg 2.1.2 does not use the floating mask of reg_aladin (bench/baselines/INSTALL.md), so its masked "
                    "runs repeat the unmasked ones.")
    fs = [t for t in ("mri_robust_register_mask", "mri_coreg_mask") if t in full]
    if fs:
        out += para(f"The help of {' and '.join(t.removesuffix('_mask') for t in fs)} describes the option as masking the "
                    "image with the mask, and the moving image is already 0 outside the tissue.")
    out += errors_text([x for x in blm if x["crop"] in names])
    em = [x for x in blm if x["tool"] == "elastix_mask" and x["crop"] in names]
    if em and all("Too many samples map outside moving image buffer" in (x.get("log_error") or "") for x in em):
        out += para("elastix counts a sample that maps outside the moving mask as invalid and stops when fewer than a quarter "
                    "of its samples are valid (RequiredRatioOfValidSamples, 0.25 by default).")
    return trim(out + check_table(blm))


def near_section(bln):
    """The tools from a start near R, given as the header of the moving image (baselines.py --near)."""
    names = [crop_name(PRIMARY, "C3", k) for k in range(SEEDS)]
    bln = [x for x in bln if x["crop"] in names]
    st = list({x["crop"]: x["start_pose"] for x in bln if "start_pose" in x}.values())
    hs = list({x["crop"]: x["start_from_S"] for x in bln if "start_from_S" in x}.values())
    out = ["## Start near R", ""] + para(
        "The tools at their documented settings, started near R, on the C3 crops of the 10° FLASH."
        " The start of crop k is R turned by 10° about a random axis through "
        "the centre of the OCT tissue and shifted by 1 mm in a random direction (numpy default_rng(4700 + k)). The tools get "
        "it as the header of the moving image, with the option that starts from the header alignment "
        "(bench/baselines/standard_tools.py). A NIfTI qform cannot hold the shear of R (ANTs, elastix and greedy read the "
        "qform), so the start is the pose with a rotation and a scale per axis closest to the turned and shifted R over the "
        f"OCT tissue: {rng([h['mean_mm'] for h in hs])} mm from it (mean over the tissue, at most "
        f"{max(h['max_mm'] for h in hs):.2f} mm), {rng([x['mean_mm'] for x in st])} mm and "
        f"{rng([x['rotation_deg'] for x in st], 1)}° from R. The start row is that pose without registration. Times are "
        "wall clock, from another queue than the tool runs above, so the two do not compare.")
    d = [x["mean_mm"] for x in st]
    out += TOOL_HEAD + [f"| the start | no registration | {counts(st)[1]} | {counts(st)[2]} | 0 | {fmt(med(d))} ({rng(d)}) | "
                        f"{fmt(med([x['rotation_deg'] for x in st]), 0)} | |"]
    tools = list(dict.fromkeys(x["tool"] for x in bln))
    for t in tools:
        r = [x for x in bln if x["tool"] == t]
        out.append(tool_row(t, r[0]["setting"], r, spread=True))
    out += [""] + para("Per crop: the distance from R (mm) of the start, with its rotation (°), and of every result, with its "
                       "rotation for the results within 2.5 mm. \"error\" is a tool error.")
    out += ["| k | start | " + " | ".join(tools) + " |", "|---|---|" + "---|" * len(tools)]
    cell = lambda x: ("" if x is None else "error" if "mean_mm" not in x else
                      f"{x['mean_mm']:.2f}" + (f", {x['rotation_deg']:.1f}°" if x["mean_mm"] <= BASIN[0] else ""))
    for k, n in enumerate(names):
        s = next((x["start_pose"] for x in bln if x["crop"] == n and "start_pose" in x), None)
        row = [next((x for x in bln if x["crop"] == n and x["tool"] == t), None) for t in tools]
        out.append(f"| {k} | " + ("" if s is None else f"{s['mean_mm']:.2f}, {s['rotation_deg']:.1f}°") + " | "
                   + " | ".join(cell(x) for x in row) + " |")
    return trim(out + [""] + errors_text(bln) + check_table(bln))


NAMES = {"no_refine": "no refinement (§4): the best search pose",
         "first_choice": "only the best search pose refined (§4)",
         "no_search": "no orientation search (§3): the refinement from the header orientation, search grid centres aligned",
         "no_prior": "no scale and shear prior (§4, λ 0)",
         "signed": "signed normalised gradients instead of their tensor (§2), OCT polarity +1",
         "signed_neg": "signed normalised gradients instead of their tensor (§2), OCT polarity −1",
         "full_tensor": "the full tensor n nᵀ instead of its trace-free part (§2)"}


def ablation_section(abl, rows):
    main = {x["name"]: x for x in rows}
    out = ["## Ablation", ""] + para(
        "bench/ablate.py: the method with one change, on the C3 crops of the 10° FLASH (analyses.py). The difference is the "
        "variant's distance from R minus the method's on the same crop. The outline is unused on these crops (the OCT is "
        "given as tissue only and their MRI is all tissue), so its variants are not run. The vessel columns give the median "
        "distance of the MRI vessel labels to the nearest OCT vessel through the affine alone and through the affine and the "
        "deformation, both on the labels the two place inside (common.vessel_field), median over the crops.")
    out += ["| variant | change | <= 1 mm | distance from R, median (range) (mm) | difference, median (range) (mm) "
            "| vessels, affine (mm) | vessels, with §5 (mm) |", "|---|---|---|---|---|---|---|"]
    crops = sorted({x["crop"] for x in abl})
    base = [main[c] for c in crops if c in main]
    vf = lambda r, k: fmt(med([x["deform"][k] for x in r if x.get("deform") and x["deform"].get(k) is not None]), 3)
    out.append(f"| octreg | the method | {counts(base)[1]} | {fmt(med([x['mean_mm'] for x in base]))} "
               f"({rng([x['mean_mm'] for x in base])}) | | {vf(base, 'affine_median_mm')} | {vf(base, 'field_median_mm')} |")
    for v in NAMES:
        r = [x for x in abl if x["variant"] == v]
        if not r:
            continue
        diff = [dist(x) - main[x["crop"]]["mean_mm"] for x in r if x["crop"] in main and "mean_mm" in x]
        out.append(f"| {v} | {NAMES[v]} | {counts(r)[1]} | {fmt(med([dist(x) for x in r]))} ({rng([dist(x) for x in r])}) | "
                   f"{fmt(med(diff), 3)} ({rng(diff, 3)}) | {vf(r, 'affine_median_mm')} | {vf(r, 'field_median_mm')} |")
    return out


def main():
    rows = load("octreg.json") or []
    out = ["# I46: results", ""] + para(
        "Generated by bench/i46/summarize.py from the JSON files next to this one, and for the start rows and the scales of R "
        "from the inputs of prepare.py. bench/i46/README.md describes the test. Distances are the mean displacement over the "
        "OCT tissue from the authors' registration R, or from the label-only fit L where named. Counts carry their "
        "Clopper-Pearson 95 % interval. Medians are over all runs of a row. A run that ended without a readable transform (a "
        "tool error) counts as beyond every other.")
    out += reference_section(load("reference_check.json"), load("context.json"))
    if rows:
        out += octreg_section(rows) + [""] + per_crop_section(rows, load("crops.json") or []) + [""]
    if load("starts.json"):
        out += starts_section(load("starts.json")) + [""]
    out += tools_section(rows, load("baselines.json"), load("baselines_search.json")) + [""]
    if load("baselines_mask.json") and load("baselines.json"):
        out += mask_section(load("baselines.json"), load("baselines_mask.json")) + [""]
    if load("baselines_near.json"):
        out += near_section(load("baselines_near.json")) + [""]
    if load("ablation.json"):
        out += ablation_section(load("ablation.json"), rows) + [""]
    (RESULTS / "tables.md").write_text("\n".join(out).rstrip("\n") + "\n", encoding="utf-8")
    print("->", RESULTS / "tables.md")


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()

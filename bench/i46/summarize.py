#!/usr/bin/env python3
"""The tables of the I46 test from the JSON files in bench/results/I46, and for the start rows and the scales of R from the
inputs of bench/i46/prepare.py -> bench/results/I46/tables.md

    octreg.json                        the main runs of octreg (run.py collect)
    baselines.json, baselines_search.json   the standard tools at their declared settings and with their own search
    baselines_mask.json, baselines_near.json   the tools with the OCT tissue as moving mask, and from the start near R
    near.json                          octreg without its orientation search from the start near R (analyses.py near)
    ablation.json, starts.json, choices.json   analyses.py collect
    reference_check.json, crops.json, context.json   the reference, the crops and the contrast of the images

Distances are the mean displacement from the reference R over the OCT tissue points (common.error), and from the label-only
fit L where named. Counts carry their Clopper-Pearson 95 % interval, which treats the runs of a row as independent. They are
not, so the interval describes the placement of the crops on this one block, not a rate on other specimens. Medians are over
all runs of a row, a run without a transform counted as beyond every other. The vessel check (common.vessel_score) is a median
distance over the MRI vessel labels inside the OCT tissue, given over the runs with at least 50 such labels, counted in
brackets.

    python bench/i46/summarize.py
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
    a, s = lab.get("A12"), lab.get("S9")
    if a:
        th = " / ".join(f"{t['mean_mm']:.2f}" for t in a["by_depth_third"])
        out += para(
            f"R is the authors' registration, mapping the OCT world to the MRI world in the test's OCT frame. The label-only fit "
            f"L (reference_check.py, an affine fitted to the vessels alone) lies {a['mean_mm']:.3f} mm from R on average "
            f"(p95 {a['p95_mm']:.2f} mm, rotation {a['rotation_deg']:.2f}°), {th} mm in the upper, middle and deepest third of the OCT depth. Fitting half of the "
            f"labels moves L by {a['split_half_spread']['mri']['rms_mm']:.2f} mm (rms over ten splits of the MRI labels) and "
            f"{a['split_half_spread']['oct']['rms_mm']:.2f} mm (of the OCT vessels). The 9-parameter fit S9 lies "
            f"{s['mean_mm']:.3f} mm from R. Distances below 0.30 mm are therefore not interpreted.")
    v = (ref or {}).get("vessels")
    if v:
        out += para(
            f"The vessel check gives a median of {v['reference']['vessel_median_mm']:.3f} mm at R ({v['reference']['vessel_n']} "
            f"labels inside the OCT tissue), {v['shift_0.5mm']['vessel_median_mm']:.3f} mm with R shifted by 0.5 mm and "
            f"{v['shift_1mm']['vessel_median_mm']:.3f} mm shifted by 1 mm (means over {v['directions']} directions). R was "
            "fitted with a vessel term of the same kind, so the check favours R.")
    if ctx:
        out += para("Grey/white contrast-to-noise ratio of the images inside the box of the OCT: " + ", ".join(
                        f"{FLIPS[k][1]}° FLASH {v:.2f}" if k in FLIPS else f"T1 map {v:.2f}"
                        for k, v in ctx.get("grey_white_cnr", {}).items() if k != "box_voxels") + ".")
    return out


def vessel(rows, key=None):
    """Median vessel check over the runs with a value, and their count."""
    v = [(x[key] if key else x).get("vessel_median_mm") for x in rows if "mean_mm" in x]
    v = [x for x in v if x is not None]
    return f"{fmt(med(v), 3)} ({len(v)})"


def octreg_section(rows):
    out = ["## octreg", ""] + para(
        "octreg at its default parameters. Parts of the method, the outline rule of §2, the valley rule of §1, the candidate "
        "suppression of §3 and the score of §5, were designed with runs on these crops, so this is not a held-out test. §4: "
        "the affine before §5, from the same run.")
    out += ["| condition | <= 0.30 mm | <= 1 mm | <= 2.5 mm and <= 10° | distance from R, median (range) (mm) "
            "| from L, median (mm) | §4 from R, median (mm) | rotation from R (°) | vessel check, final / §4, median (runs) (mm) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for im, cond, masked in conditions():
        r = [x for x in rows if x["image"] == im and x["cond"] == cond and x["oct_mask"] == masked]
        if not r:
            continue
        d = [x["mean_mm"] for x in r]
        out.append(f"| {label(im, cond, masked)} | " + " | ".join(counts(r))
                   + f" | {fmt(med(d))} ({rng(d)}) | {fmt(med([x.get('mean_L_mm') for x in r if 'mean_L_mm' in x]))} | "
                   f"{fmt(med([x['s4']['mean_mm'] for x in r]))} | {rng([x['rotation_deg'] for x in r], 1)} | "
                   f"{vessel(r)} / {vessel(r, 's4')} |")
    if rows and "rotation_axis" in rows[0]:
        ax = {}
        for im in FLIPS:
            a = np.array([x["rotation_axis"] for x in rows if x["image"] == im and x["cond"] == "C3" and x["oct_mask"]])
            if len(a):
                m = a.mean(0)
                ax[im] = (m / np.linalg.norm(m)).round(2).tolist()
        sc = np.array([x["scales"] for x in rows if "scales" in x])
        out += [""] + para(
            "Rotation axis of T R^-1, mean over the C3 runs of each image (MRI world): "
            + ", ".join(f"{FLIPS[k][1]}° FLASH {v}" for k, v in ax.items())
            + f". Scale along the OCT axes over all runs: {rng(sc[:, 0], 3)}, {rng(sc[:, 1], 3)} and {rng(sc[:, 2], 3)}, and "
            f"of R: {', '.join(f'{x:.3f}' for x in np.linalg.norm(reference()[:3, :3], axis=0))}.")
    return out


def per_crop_section(rows, crops):
    out = ["## octreg per crop: C3 on the 10° FLASH", ""] + para(
        "Distance from R (mm) of the final affine and of the §4 affine, the distance of the final affine from L, and the "
        "distance from the centre of the crop to the centre of the OCT tissue placed by R (crops.json).")
    out += ["| k | crop centre to tissue centre (mm) | final | §4 | final from L |", "|---|---|---|---|---|"]
    for k in range(SEEDS):
        name = crop_name(PRIMARY, "C3", k)
        c = next((x for x in crops if x["name"] == name), None)
        x = next((x for x in rows if x["name"] == name), {})
        out.append(f"| {k} | {fmt(c['crop_centre_to_tissue_centre_mm']) if c else ''} | {fmt(x.get('mean_mm'))} | "
                   f"{fmt(x.get('s4', {}).get('mean_mm'))} | {fmt(x.get('mean_L_mm'))} |")
    return out


def starts_section(st):
    out = ["## Start test on flip1_C3_k0", ""] + para(
        "The OCT turned through the centre of its image box, by 90° and 180° about each array axis (axis<a>_<deg>) and by 20 "
        "rotations drawn uniformly on SO(3) (rand00 to rand19, seed 4646), then octreg and A14, its §4 and §5 from the "
        "turned start without the orientation search. The start column is the distance of the turned start from R.")
    for group, pre in (("turned", "axis"), ("random", "rand")):
        rows = [x for x in st if x["start"].startswith(pre)]
        if not rows:
            continue
        for kind in ("method", "A14"):
            r = [x for x in rows if x["kind"] == kind]
            out += para(f"{group} starts, {'octreg' if kind == 'method' else 'A14'}: within 1 mm "
                        f"{cp(sum(x['mean_mm'] <= 1 for x in r), len(r))}, distance from R {rng([x['mean_mm'] for x in r])} "
                        f"mm, rotation {rng([x['rotation_deg'] for x in r], 1)}°.")
    out += ["| start | the start: distance from R (mm), rotation (°) | octreg (mm) | A14 (mm) |", "|---|---|---|---|"]
    for s in dict.fromkeys(x["start"] for x in st):
        m = next(x for x in st if x["start"] == s and x["kind"] == "method")
        a = next(x for x in st if x["start"] == s and x["kind"] == "A14")
        out.append(f"| {s} | {m['start_pose']['mean_mm']:.1f}, {m['start_pose']['rotation_deg']:.0f} | {m['mean_mm']:.2f} | "
                   f"{a['mean_mm']:.2f} |")
    return out


def tools_section(rows, bl, bls):
    out = ["## Standard tools", ""] + para(
        "Each tool at the setting declared before any run (bench/baselines/standard_tools.py), and the three tools that have a global "
        "rotation search with that search turned on (one change from the declared setting, declared before those runs), on the "
        "C3 crops of the 10° FLASH. A tool error is a run that ended without a readable transform, counted as beyond. The "
        "start is the pose before any registration: the header orientation of the OCT with the centres of the two image boxes "
        "aligned.")
    out += ["| method | setting | <= 1 mm | <= 2.5 mm and <= 10° | tool errors | distance from R, median (mm) | rotation, "
            "median (°) | vessel check, median (runs) (mm) | time, median (s) |", "|---|---|---|---|---|---|---|---|---|"]
    names = [crop_name(PRIMARY, "C3", k) for k in range(SEEDS)]
    st = [start_pose(n) for n in names]
    out.append(f"| the start | no registration | {counts(st)[1]} | {counts(st)[2]} | 0 | {fmt(med([x['mean_mm'] for x in st]))} "
               f"({rng([x['mean_mm'] for x in st], 1)}) | {fmt(med([x['rotation_deg'] for x in st]), 0)} | | |")
    groups = [("octreg", "its default parameters", [x for x in rows if x["oct_mask"] and x["name"] in names])]
    for tr in (bl or [], bls or []):
        for t in dict.fromkeys(x["tool"] for x in tr):
            r = [x for x in tr if x["tool"] == t and x["crop"] in names]
            groups.append((t, r[0]["setting"], r))
    for method, setting, r in groups:
        ok = [x for x in r if "mean_mm" in x]
        out.append(f"| {method} | {setting} | {counts(r)[1]} | {counts(r)[2]} | {len(r) - len(ok)} | "
                   f"{fmt(med([dist(x) for x in r]))} | {fmt(med([x.get('rotation_deg', INF) for x in r]), 0)} | "
                   f"{vessel(r)} | {fmt(med([x['seconds'] for x in ok if x.get('seconds') is not None]), 0)} |")
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
             "median (°) | vessel check, median (mm) | time, median (s) |", "|---|---|---|---|---|---|---|---|---|"]


def tool_row(method, setting, r, spread=False):
    """One row of TOOL_HEAD over the runs r, the median distance with its range if spread."""
    ok = [x for x in r if "mean_mm" in x]
    d = [dist(x) for x in r]
    return (f"| {method} | {setting} | {counts(r)[1]} | {counts(r)[2]} | {len(r) - len(ok)} | "
            f"{fmt(med(d))}{f' ({rng(d)})' if spread else ''} | {fmt(med([x.get('rotation_deg', INF) for x in r]), 0)} | "
            f"{fmt(med([x['vessel_median_mm'] for x in ok if x.get('vessel_median_mm') is not None]), 3)} | "
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
    """The tools with the OCT tissue as the moving mask (baselines.py --mask), each under its declared row."""
    names = [crop_name(PRIMARY, "C3", k) for k in range(SEEDS)]
    out = ["## Standard tools with a moving mask", ""] + para(
        "Each tool whose options take a mask of the moving image, at its declared setting with the OCT tissue as that mask "
        "(the voxels above 0 of the moving image), on the C3 crops of the 10° FLASH, declared on 2026-10-16 before these "
        "runs. Each masked row follows the tool's declared row on the same crops. mri_synthmorph register has no mask option. "
        "Times are wall clock, of the masked runs with up to six tool runs at a time on 24 cores and of the declared runs one "
        "at a time, so the two do not compare.")
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
        "Masked runs that end at the distance from R of the declared run (to 0.1 µm), or in a tool error where it does: "
        + ", ".join(f"{t} {k}/{n}" for t, (k, n) in same.items()) + ".")
    full = [t for t, (k, n) in same.items() if k == n]
    if "reg_aladin_mask" in full:
        out += para("NiftyReg 2.1.2 does not use the floating mask of reg_aladin (bench/baselines/INSTALL.md), so its masked "
                    "runs repeat the declared ones.")
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


def near_section(nr, bln):
    """The tools from the header alignment and octreg's A14, all from the near start 10 deg and 1 mm from R (baselines.py
    --near, analyses.py near)."""
    names = [crop_name(PRIMARY, "C3", k) for k in range(SEEDS)]
    nr, bln = [x for x in nr if x["crop"] in names], [x for x in bln if x["crop"] in names]
    st = list({x["crop"]: x["start_pose"] for x in nr + bln if "start_pose" in x}.values())
    hs = list({x["crop"]: x["start_from_S"] for x in nr + bln if "start_from_S" in x}.values())
    out = ["## Start near R", ""] + para(
        "The tools at their declared settings, started from the header alignment of the two images, and octreg without its "
        "orientation search (A14: §4 from the start at both polarities, then §5), all from a start 10° and 1 mm from R, on the "
        "C3 crops of the 10° FLASH, declared on 2026-10-16 before these runs. The start of crop k is R turned by 10° about a "
        "random axis through the centre of the OCT tissue and shifted by 1 mm in a random direction (numpy "
        "default_rng(4700 + k)). The tools get it as the header of the moving image, with the option that starts from the "
        "header alignment (bench/baselines/standard_tools.py), and octreg as its start pose. R has a shear, which the qform of a "
        "NIfTI header cannot hold (ANTs, elastix and greedy read the qform), so the start is the pose closest to it over the "
        f"OCT tissue with a rotation and a scale per axis: {rng([h['mean_mm'] for h in hs])} mm from the turned and shifted R "
        f"(mean over the tissue, at most {max(h['max_mm'] for h in hs):.2f} mm), {rng([x['mean_mm'] for x in st])} mm and "
        f"{rng([x['rotation_deg'] for x in st], 1)}° from R. The start row is that pose without registration. A tool error is "
        "a run that ended without a readable transform, counted as beyond. Times are wall clock, of the tools with up to six "
        "tool runs at a time on 24 cores, and of octreg for §4 and §5 on the GPU, without the preprocessing.")
    d = [x["mean_mm"] for x in st]
    out += TOOL_HEAD + [f"| the start | no registration | {counts(st)[1]} | {counts(st)[2]} | 0 | {fmt(med(d))} ({rng(d)}) | "
                        f"{fmt(med([x['rotation_deg'] for x in st]), 0)} | | |"]
    if nr:
        out.append(tool_row("octreg", "§4 and §5 from the start, no orientation search", nr, spread=True))
    tools = list(dict.fromkeys(x["tool"] for x in bln))
    for t in tools:
        r = [x for x in bln if x["tool"] == t]
        out.append(tool_row(t, r[0]["setting"], r, spread=True))
    out += [""] + para("Per crop: the distance from R (mm) of the start, with its rotation (°), and of every result, with its "
                       "rotation for the results within 2.5 mm. \"error\" is a tool error.")
    cols = (["octreg"] if nr else []) + tools
    out += ["| k | start | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    cell = lambda x: ("" if x is None else "error" if "mean_mm" not in x else
                      f"{x['mean_mm']:.2f}" + (f", {x['rotation_deg']:.1f}°" if x["mean_mm"] <= BASIN[0] else ""))
    for k, n in enumerate(names):
        s = next((x["start_pose"] for x in nr + bln if x["crop"] == n and "start_pose" in x), None)
        row = [next((x for x in nr if x["crop"] == n), None)] if nr else []
        row += [next((x for x in bln if x["crop"] == n and x["tool"] == t), None) for t in tools]
        out.append(f"| {k} | " + ("" if s is None else f"{s['mean_mm']:.2f}, {s['rotation_deg']:.1f}°") + " | "
                   + " | ".join(cell(x) for x in row) + " |")
    return trim(out + [""] + errors_text(bln) + check_table(bln))


NAMES = {"base": "the method", "A12": "no fine-structure refinement (§5): the §4 pose", "A4": "standardised intensities instead of "
         "the two-class maps (§2)", "A6p": "no scale and shear penalty (§4)", "A14": "no orientation search (§3), from the header "
         "orientation with the image centres aligned", "A9": "no outline term (§2)", "A10": "two-sided outline (§2)",
         "A13": "the outline alone (§2)"}


def ablation_section(abl, rows):
    main = {x["name"]: x for x in rows}
    a13 = rng([x["s4"]["mean_mm"] for x in abl if x["variant"] == "A13" and "s4" in x])
    a9 = max(abs(dist(x) - main[x["crop"]]["mean_mm"]) for x in abl if x["variant"] == "A9" and x["crop"] in main)
    outside = max(x["outside_fraction"]["oct"] for x in rows if x["oct_mask"])
    out = ["## Ablation", ""] + para(
        "bench/ablate.py: the method with one element removed or replaced, everything else unchanged, on the C3 crops of the "
        "10° FLASH, and A4 also on the C3 crops of the 20° and 30° FLASH and the C12 crops of the 10° FLASH. The difference is "
        "the variant's distance from R minus the method's on the same crop. On I46 the outline is unused in every run (the "
        f"masked OCT has almost no measured outside, at most {100 * outside:.2f} % of its measured voxels). A13 then scores 0 "
        "for every pose, so §3 and §4 keep one search candidate in the "
        f"header orientation, {a13} mm from R, and §5 then moves it. A9 changes only the OCT weights of the search, so its "
        f"differences show the numerical spread of the method, up to {a9:.2f} mm. A10 adds a two-sided outline to §4 alone, "
        "as the search keeps the outline off.")
    out += ["| crops | variant | change | <= 1 mm | distance from R, median (range) (mm) | difference, median (range) (mm) "
            "| from L, median (mm) | §4 from R, median (mm) |", "|---|---|---|---|---|---|---|---|"]
    groups = [("10° FLASH, C3", f"{PRIMARY}_C3_"), ("20° FLASH, C3", "flip2_C3_"), ("30° FLASH, C3", "flip3_C3_"),
              ("10° FLASH, C12", f"{PRIMARY}_C12_")]
    for gname, pre in groups:
        rows = [x for x in abl if x["crop"].startswith(pre)]
        base = [main[c] for c in sorted({x["crop"] for x in rows}) if c in main]
        if not rows:
            continue
        out.append(f"| {gname} | octreg | {NAMES['base']} | {counts(base)[1]} | {fmt(med([x['mean_mm'] for x in base]))} "
                   f"({rng([x['mean_mm'] for x in base])}) | | {fmt(med([x.get('mean_L_mm') for x in base if 'mean_L_mm' in x]))} "
                   f"| {fmt(med([x['s4']['mean_mm'] for x in base]))} |")
        for v in ["A12", "A4", "A6p", "A14", "A9", "A10", "A13"]:
            r = [x for x in rows if x["variant"] == v]
            if not r:
                continue
            na = v in ("A9", "A13")
            diff = [dist(x) - main[x["crop"]]["mean_mm"] for x in r if x["crop"] in main and "mean_mm" in x]
            out.append(f"| | {v} | {NAMES[v]}{' (the outline is unused on I46)' if na else ''} | {counts(r)[1]} | "
                       f"{fmt(med([dist(x) for x in r]))} ({rng([dist(x) for x in r])}) | "
                       f"{fmt(med(diff), 3)} ({rng(diff, 3)}) | {fmt(med([x.get('mean_L_mm') for x in r if 'mean_L_mm' in x]))} | "
                       f"{fmt(med([x['s4']['mean_mm'] for x in r if 's4' in x]))} |")
    return out


CHOICES = {"score": "§5 scored by the weighted mean of (n_O · n_M)², as Haber and Modersitzki do", "outline": "the outline used whatever "
           "the outside (outline_min_outside 0)", "valley": "every deep MRI valley taken (medium_fraction 1)",
           "nms": "candidate suppression 10° (nms_deg 10)", "topk48": "48 poses refined instead of 24 (topk 48)"}


def choices_section(ch, rows):
    out = ["## Design choices", ""] + para(
        f"The method with one design choice replaced at a time (analyses.py choices): on all {len(rows)} runs, the valley "
        "rule only on the runs where the method passes a valley over, and 48 refined poses on the runs whose result came from "
        "the 18th to 24th of the 24 search candidates. The rows pool the conditions, or take only the runs where a choice "
        "applies, so their counts are given without an interval.")
    main = {x["name"]: x for x in rows}
    out += ["| replaced by | runs | <= 1 mm, the method | <= 1 mm, replaced | runs beyond 1 mm when replaced "
            "| distance from R, median, the method / replaced (mm) |", "|---|---|---|---|---|---|"]
    for c in CHOICES:
        r = [x for x in ch if x["choice"] == c]
        if not r:
            continue
        base = [main[x["run"]] for x in r]
        lost = [x["run"] for x in r if dist(x) > 1]
        within = lambda d: f"{sum(x <= 1 for x in d)}/{len(d)}"
        out.append(f"| {CHOICES[c]} | {len(r)} | {within([x['mean_mm'] for x in base])} | {within([dist(x) for x in r])} | "
                   f"{', '.join(lost) or 'none'} | "
                   f"{fmt(med([x['mean_mm'] for x in base]))} / {fmt(med([dist(x) for x in r]))} |")
    return out


def main():
    rows = load("octreg.json") or []
    out = ["# I46: results", ""] + para(
        "Generated by bench/i46/summarize.py from the JSON files next to this one, and for the start rows and the scales of R "
        "from the inputs of prepare.py. Distances are the mean displacement over "
        "the OCT tissue from the authors' registration R, or from the label-only fit L where named. They measure agreement "
        "with these registrations, which are not ground truth. Counts carry their Clopper-Pearson 95 % interval, which treats "
        "the runs of a row as independent. They are not, so the interval describes the placement of the crops on this one "
        "block, not a rate on other specimens. Medians are over all runs of a row, a run without a transform counted as beyond "
        "every other. The vessel check is a median over the runs with at least 50 MRI vessel labels inside the OCT tissue, "
        "counted in brackets.")
    out += reference_section(load("reference_check.json"), load("context.json"))
    if rows:
        out += octreg_section(rows) + [""] + per_crop_section(rows, load("crops.json") or []) + [""]
    if load("starts.json"):
        out += starts_section(load("starts.json")) + [""]
    out += tools_section(rows, load("baselines.json"), load("baselines_search.json")) + [""]
    if load("baselines_mask.json") and load("baselines.json"):
        out += mask_section(load("baselines.json"), load("baselines_mask.json")) + [""]
    if load("near.json") or load("baselines_near.json"):
        out += near_section(load("near.json") or [], load("baselines_near.json") or []) + [""]
    if load("ablation.json"):
        out += ablation_section(load("ablation.json"), rows) + [""]
    if load("choices.json"):
        out += choices_section(load("choices.json"), rows) + [""]
    (RESULTS / "tables.md").write_text("\n".join(out).rstrip("\n") + "\n", encoding="utf-8")
    print("->", RESULTS / "tables.md")


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()

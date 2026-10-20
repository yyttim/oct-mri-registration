#!/usr/bin/env python3
"""Write bench/results/I46/tables.md from the JSON files in bench/results/I46, and the start rows, the scales of R and the
residual of L from the inputs of bench/i46/prepare.py.

    python bench/i46/summarise.py

    octreg.json                                 the octreg runs (run.py collect)
    baselines.json, baselines_search.json       the standard tools at their documented settings and with their own search
    baselines_mask.json, baselines_near.json    the tools with the test's specimen mask as moving mask, and from a start near R
    baselines_near_mask.json, baselines_near_zeroed.json, baselines_atR.json, baselines_atR_mask.json
                                                the controls on the start near R (baselines.py --near-mask and the others)
    ablation.json, starts.json                  analyses.py collect
    reference_check.json, context.json          the reference and the contrast of the images
"""
from __future__ import annotations

import itertools
import json
import re
import sys
import textwrap
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CONDITIONS, FLIPS, OCT, PRIMARY, RESULTS, SEEDS, TIERS, crop_file, crop_name, distance,  # noqa: E402
                    error, label_fit, reference, rigid_fit, rotation)
from analyses import N_RANDOM, SEED  # noqa: E402
from baselines import NEAR_DEG, NEAR_MM, NEAR_SEED  # noqa: E402
from bench.ablate import VARIANTS as CHANGES, header_start  # noqa: E402

INF = float("inf")
NBSP = " "


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


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
    return [line.replace(NBSP, " ") for line in textwrap.wrap(text.replace(" = ", f"{NBSP}={NBSP}"), 124)] + [""]


def trim(lines):
    while lines and lines[-1] == "":
        lines = lines[:-1]
    return lines


def dist(x):
    return x["mean_mm"] if "mean_mm" in x else INF


def counts(rows):
    """The runs of rows within 1 mm of R, as "k/n"."""
    d = [dist(x) for x in rows]
    return f"{sum(m <= TIERS[1] for m in d)}/{len(d)}"


def tails(rows):
    """p95 / max (mm) of the distance from R, medians over rows."""
    return f"{fmt(med([x['p95_mm'] for x in rows]))} / {fmt(med([x['max_mm'] for x in rows]))}"


def vessels(rows):
    """The vessel check, median over rows."""
    return fmt(med([x["vessel_median_mm"] for x in rows if x.get("vessel_median_mm") is not None]), 3)


def ranks(rows):
    """The winning search ranks of rows, each with its number of runs."""
    return ", ".join(f"{r} ({n})" for r, n in sorted(Counter(x["search_rank"] for x in rows).items()))


def between(a, b):
    """Mean displacement (mm) over the tissue and rotation (deg) between two affines."""
    return distance(a, b), rotation(a, b)[0]


def label(image, cond, masked=False):
    m, s = CONDITIONS[cond]
    return (f"{FLIPS[image][1]}° FLASH, {cond} (margin {m:g} mm, shift up to {s:g} mm)"
            + (", the test's specimen mask as --oct-mask" if masked else ""))


def conditions():
    """(image, condition, with --oct-mask) of the rows of the octreg table."""
    c = [(im, "C3", False) for im in FLIPS] + [(PRIMARY, cond, False) for cond in CONDITIONS if cond != "C3"]
    return c + [(PRIMARY, "C3", True)]


def start_pose(crop):
    """common.error of the pose before any registration: bench/ablate.py header_start of the OCT and the crop."""
    from octreg import io
    vo, vm = io.load_volume(OCT), io.load_volume(crop_file(crop))
    return error(header_start(vm.affine, vm.shape, vo.affine, vo.shape))


def vessel_text(v):
    """The vessel check of R, of L and of R moved with the labels R places inside held (reference_check.py vessels)."""
    score = lambda x: f"{x['vessel_median_mm']:.3f} mm (mean {x['vessel_mean_mm']:.3f} mm)"
    m = v["moved"]
    shifts = ", ".join(f"{k} mm {x['mean']['vessel_median_mm']:.3f}" for k, x in m["shift_mm"].items())
    turns = ", ".join(f"{k}° {x['z']['vessel_median_mm']:.3f} about the depth axis and {x['x']['vessel_median_mm']:.3f} "
                      f"about an in-plane axis" for k, x in m["rotation_deg"].items())
    scales = ", ".join(f"{k} {x['vessel_median_mm']:.3f}" for k, x in m["depth_scale"].items())
    return ("Vessel check (common.vessel_score): the distance of the MRI vessel labels placed in the OCT tissue to the nearest "
            f"OCT vessel, median over the labels, is {score(v['reference'])} for R and {score(v['A12'])} for L. With the "
            f"{m['labels']} labels R places inside held fixed, the median through R moved is: shifted along an OCT axis "
            f"(mean over the six directions) {shifts}, turned {turns}, and with the depth scaled by {scales}.")


def reference_section(ref, ctx, runs):
    """runs: every scored run of the tables, counted against the floor TIERS[0]."""
    a = ref["labels"]["A12"]
    th = " / ".join(f"{t['mean_mm']:.2f}" for t in a["by_depth_third"])
    close = Counter(x.get("tool") or x.get("variant") or x.get("start") or "octreg" for x in runs if dist(x) <= TIERS[0])
    n = sum(close.values())
    who = f"all {next(iter(close))}" if len(close) == 1 else ", ".join(f"{k} {t}" for t, k in close.most_common())
    out = ["## Reference", ""] + para(
        "R is the authors' registration, from the OCT world to the MRI world in the test's OCT frame. The label-only fit "
        f"L (reference_check.py, an affine fitted to the vessels alone) lies {a['mean_mm']:.3f} mm from R on average "
        f"(p95 {a['p95_mm']:.2f} mm, rotation {a['rotation_deg']:.2f}°), {th} mm in the upper, middle and deepest third "
        f"of the OCT depth, so distances below {TIERS[0]:.2f} mm are not interpreted. "
        + (f"No run in these tables comes within {TIERS[0]:.2f} mm of R (0 of {len(runs)})." if n == 0 else
           f"{n} of the {len(runs)} runs in these tables come within {TIERS[0]:.2f} mm of R, {who}."))
    if "vessels" in ref:
        out += para(vessel_text(ref["vessels"]))
    return out + para("Grey/white contrast-to-noise ratio of the images inside the box of the OCT: " + ", ".join(
        f"{FLIPS[k][1]}° FLASH {v:.2f}" if k in FLIPS else f"T1 map {v:.2f}"
        for k, v in ctx["grey_white_cnr"].items() if k != "box_voxels") + ".")


def octreg_section(rows, st):
    out = ["## octreg", ""] + para(
        "octreg at its default parameters on the released OCT, with its own specimen mask and MRI foreground (§1). The last "
        "row repeats the first with the test's specimen mask as --oct-mask in place of octreg's own. p95 / max and vessels "
        "are medians over the runs of the row. The vessel check is the median distance of the MRI vessel labels to the "
        "nearest OCT vessel through the affine (common.vessel_score), the search rank the rank of the search pose that won "
        "the refinement, with its number of runs.")
    out += ["| condition | <= 1 mm | distance from R, median (range) (mm) | p95 / max (mm) | from L, median (mm) "
            "| rotation from R (°) | vessels (mm) | search rank |", "|---|---|---|---|---|---|---|---|"]
    for im, cond, masked in conditions():
        r = [x for x in rows if x["image"] == im and x["cond"] == cond and x["oct_mask"] == masked]
        d = [x["mean_mm"] for x in r]
        out.append(f"| {label(im, cond, masked)} | {counts(r)} | {fmt(med(d))} ({rng(d)}) | {tails(r)} | "
                   f"{fmt(med([x['mean_L_mm'] for x in r if 'mean_L_mm' in x]))} | {rng([x['rotation_deg'] for x in r], 1)} | "
                   f"{vessels(r)} | {ranks(r)} |")
    main = [x for x in rows if not x["oct_mask"]]
    near = [x for x in main if x["mean_mm"] <= TIERS[1]]
    axes = []
    for im in FLIPS:
        a = np.array([x["rotation_axis"] for x in near if x["image"] == im and x["cond"] == "C3"]).reshape(-1, 3)
        axes.append(f"{FLIPS[im][1]}° FLASH " + (str((a.mean(0) / np.linalg.norm(a.mean(0))).round(2).tolist()) if len(a)
                                                  else "none"))
    sc = np.array([x["scales"] for x in near]).reshape(-1, 3)
    out += [""] + para(
        f"Over the {len(near)} runs without --oct-mask that end within 1 mm of R: the rotation axis of T R^-1 (MRI world), "
        "mean over the C3 runs of each image, " + ", ".join(axes) + f". Over the same runs the scale along the OCT axes is "
        f"{rng(sc[:, 0], 3)}, "
        f"{rng(sc[:, 1], 3)} and {rng(sc[:, 2], 3)}, and that of R "
        f"{', '.join(f'{x:.3f}' for x in np.linalg.norm(reference()[:3, :3], axis=0))}.")
    return trim(out + precision_text(rows, st) + residual_text(main) + search_text(rows))


def short(im, cond, masked):
    return f"{FLIPS[im][1]}° FLASH {cond}" + (" with --oct-mask" if masked else "")


def precision_text(rows, st):
    """The spread of the affines across crops, contrasts and starts, and the rigid transform closest to R."""
    T = lambda x: np.array(x["T_oct2mri"], float)
    cross = []
    for im, cond, masked in conditions():
        r = [x for x in rows if x["image"] == im and x["cond"] == cond and x["oct_mask"] == masked]
        cross.append(f"{short(im, cond, masked)} "
                     f"{rng([distance(T(a), T(b)) for a, b in itertools.combinations(r, 2)], 3)} mm")
    by = {x["name"]: T(x) for x in rows}
    pairs = []
    for a, b in itertools.combinations(FLIPS, 2):
        p = [between(by[crop_name(a, "C3", k)], by[crop_name(b, "C3", k)]) for k in range(SEEDS)]
        pairs.append(f"{FLIPS[a][1]}° vs {FLIPS[b][1]}° {rng([d for d, _ in p], 3)} mm and {rng([g for _, g in p], 2)}°")
    starts = [between(T(a), T(b)) for a, b in itertools.combinations(st, 2)]
    R, main = reference(), [x for x in rows if not x["oct_mask"]]
    KR = rigid_fit(R)
    ours, theirs = [], []
    for x in main:
        KT = rigid_fit(T(x))
        ours.append(distance(KR @ np.linalg.inv(KT) @ T(x), R))
        theirs.append(distance(KT @ np.linalg.inv(KR) @ R, R))
    return para(
        "Precision. The mean displacement over the tissue between the ten affines of a condition is " + ", ".join(cross)
        + ", between the contrasts on the same C3 crop k " + ", ".join(pairs)
        + (f", and between the {len(st)} affines of the start test {rng([d for d, _ in starts], 3)} mm and "
           f"{rng([g for _, g in starts], 2)}°. " if starts else ". ")
        + f"The rigid transform closest to R over the tissue points (least squares) lies {distance(KR, R):.3f} mm from R. "
        f"Over the {len(main)} runs without --oct-mask, R's rotation and translation (that rigid transform) with octreg's "
        f"stretch (T with its own closest rigid transform removed) lies {fmt(med(ours), 3)} mm ({rng(ours, 3)}) from R, "
        f"and octreg's rotation and translation with R's stretch {fmt(med(theirs), 3)} mm ({rng(theirs, 3)}).")


def residual_text(main):
    """The residual against R split into rotation, stretch and shift, by depth third and by direction, and the same for L."""
    m = lambda k, nd=3: fmt(med([x[k] for x in main]), nd)
    thirds = lambda k: " / ".join(fmt(med([x[k][i] for x in main])) for i in range(3))
    L = error(label_fit())
    return para(
        f"Residual. Over the {len(main)} runs without --oct-mask, T R^-1 about the centroid of the tissue (medians): its "
        f"rotation alone moves the tissue by {m('rotation_only_mm')} mm, its stretch alone by {m('strain_only_mm')} mm, "
        f"and the centroid shifts by {m('centroid_mm')} mm. The |cos| of the rotation axis with the OCT depth axis placed by "
        f"R is {m('rotation_axis_cos_depth', 2)} ({rng([x['rotation_axis_cos_depth'] for x in main])}). By thirds of the "
        f"OCT depth the distance from R is {thirds('by_depth_third_mm')} mm and from L {thirds('by_depth_third_L_mm')} mm. "
        f"Along the depth axis placed by R the displacement from R is {m('along_depth_mm')} mm, in the plane "
        f"{m('in_plane_mm')} mm. "
        f"For L against R: rotation alone {L['rotation_only_mm']:.3f} mm, stretch alone {L['strain_only_mm']:.3f} mm, "
        f"centroid {L['centroid_mm']:.3f} mm, |cos| {L['rotation_axis_cos_depth']:.2f}, along the depth axis "
        f"{L['along_depth_mm']:.3f} mm, in the plane {L['in_plane_mm']:.3f} mm.")


def search_text(rows):
    """The ranks of the search poses that won the refinement, where rank 0 lost, and the poses proposed by S = (F + O) / 2
    where the outline was used."""
    rank0 = lambda x: next(c["L"] for c in x["candidates"] if c["search_rank"] == 0)
    lost = [x for x in rows if x["search_rank"] != 0]
    outl = [x for x in rows if x["outline_used"]]
    text = (f"Search. The search pose of rank 0 won the refinement in {len(rows) - len(lost)} of the {len(rows)} runs, the "
            f"winning ranks {ranks(rows)}.")
    if lost:
        text += (f" Where it lost, it ended at a refinement score L of {rng([rank0(x) for x in lost], 3)} against the "
                 f"winner's {rng([x['L'] for x in lost], 3)}.")
    text += f" The outline was used in {len(outl)} runs"
    if outl:
        half = "/".join(str(h) for h in sorted({x["search_topk"] // 2 for x in outl}))
        won_s = sum(x["search_rank"] < x["search_topk"] // 2 for x in outl)
        by_s = [c["L"] for x in outl for c in x["candidates"] if c["search_rank"] < x["search_topk"] // 2]
        text += (f" ({', '.join(x['name'] for x in outl)}), where the first {half} search poses are proposed by "
                 f"S = (F + O) / 2 and the rest by F. Their winners, of rank {ranks(outl)}, ended at L "
                 f"{rng([x['L'] for x in outl], 3)}, {won_s} of them proposed by S, whose poses ended at L {rng(by_s, 3)}")
    return para(text + ".")


def starts_section(st):
    out = ["## Start test on flip1_C3_k0", ""] + para(
        "The OCT is turned about the centre of its image box, by 90° and 180° about each array axis (i90 to k180) and by "
        f"{N_RANDOM} rotations drawn uniformly on SO(3) (rand00 to rand{N_RANDOM - 1:02d}, seed {SEED}), and octreg runs from "
        "each turned start (analyses.py starts). The start column gives the distance and rotation from R of the turned OCT "
        "with the image box centres aligned, the search column the same for the search pose that won the refinement.")
    for group, pre in (("turned", ("i", "j", "k")), ("random", ("rand",))):
        r = [x for x in st if x["start"].startswith(pre)]
        out += para(f"{group} starts: within 1 mm {counts(r)}, distance from R {rng([x['mean_mm'] for x in r])} mm, "
                    f"rotation {rng([x['rotation_deg'] for x in r], 1)}°.")
    out += ["| start | the start: distance from R (mm), rotation (°) | the search pose: distance from R (mm), rotation (°) "
            "| octreg (mm) | p95 / max (mm) | vessels (mm) | search rank |", "|---|---|---|---|---|---|---|"]
    for x in st:
        s, q = x["start_pose"], x["search"]
        out.append(f"| {x['start']} | {s['mean_mm']:.1f}, {s['rotation_deg']:.0f} | {q['mean_mm']:.2f}, "
                   f"{q['rotation_deg']:.1f} | {x['mean_mm']:.2f} | {tails([x])} | {vessels([x])} | {x['search_rank']} |")
    return out


TOOL_HEAD = ["| method | setting | <= 1 mm | tool errors | distance from R, median (mm) | rotation, median (°) | "
             "vessels (mm) |", "|---|---|---|---|---|---|---|"]


def tool_row(method, setting, r, spread=False):
    """One row of TOOL_HEAD over the runs r, the median distance with its range if spread."""
    d = [dist(x) for x in r]
    return (f"| {method} | {setting} | {counts(r)} | {sum('mean_mm' not in x for x in r)} | "
            f"{fmt(med(d))}{f' ({rng(d)})' if spread else ''} | {fmt(med([x.get('rotation_deg', INF) for x in r]), 0)} | "
            f"{vessels(r) if any('vessel_median_mm' in x for x in r) else ''} |")


def tools_section(rows, bl, bls):
    names = [crop_name(PRIMARY, "C3", k) for k in range(SEEDS)]
    out = ["## Standard tools", ""] + para(
        "Each tool at its documented setting (bench/baselines/standard_tools.py), and the three tools that have a global "
        "rotation search with that search turned on, on the C3 crops of the 10° FLASH. The moving image is the released OCT "
        "box-averaged to 0.15 mm, the fixed image the crop. The start is the pose before any registration: the header "
        "orientation of the OCT with the image box centres aligned.")
    octreg = [x for x in rows if not x["oct_mask"] and x["name"] in names]
    out += TOOL_HEAD + [tool_row("the start", "no registration", [start_pose(n) for n in names], spread=True),
                        tool_row("octreg", "its default parameters", octreg)]
    for tr in (bl, bls):
        for t in dict.fromkeys(x["tool"] for x in tr):
            r = [x for x in tr if x["tool"] == t and x["crop"] in names]
            out.append(tool_row(t, r[0]["setting"], r))
    out += [""] + para(f"Over all {len(bl)} runs at the documented settings, on the crops of this table and of the next, the "
                       f"median rotation from R is {fmt(med([x.get('rotation_deg', INF) for x in bl]), 0)}°.")
    others = [im for im in FLIPS if im != PRIMARY]
    k0 = [("octreg", {x["name"]: x for x in rows})]
    k0 += [(t, {x["crop"]: x for x in bl if x["tool"] == t}) for t in dict.fromkeys(x["tool"] for x in bl)]
    out += para("The same on the C3 k = 0 crop of the 20° and 30° FLASH: distance from R (mm).")
    out += ["| method | " + " | ".join(f"{FLIPS[im][1]}° FLASH" for im in others) + " |", "|---|" + "---|" * len(others)]
    for method, d in k0:
        out.append(f"| {method} | " + " | ".join(fmt(dist(d[f"{im}_C3_k0"])) if f"{im}_C3_k0" in d else "" for im in others)
                   + " |")
    return out


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
        msgs = list(dict.fromkeys(re.sub(r"\b\d+ / \d+\b", "n / m", re.sub(r"^Description:\s*", "",
                                                                             x.get("log_error") or x["error"][0]))
                                  for x in r))
        parts.append(f"{t} ended without a transform on k = {ks(r)}, its log: " + " or ".join(f'"{m}"' for m in msgs) + ".")
    return para("Tool errors. " + (" ".join(parts) if parts else "None."))


def mask_section(bl, blm):
    """The tools with the test's specimen mask as the moving mask (baselines.py --mask), each under its unmasked row."""
    names = [crop_name(PRIMARY, "C3", k) for k in range(SEEDS)]
    out = ["## Standard tools with a moving mask", ""] + para(
        "Each tool whose options take a mask of the moving image, at its documented setting with the test's specimen mask on "
        "the grid of the moving image as that mask, on the C3 crops of the 10° FLASH. Each masked row follows the tool's "
        "unmasked row. mri_synthmorph register has no mask option, and NiftyReg 2.1.2 does not use the floating mask of "
        "reg_aladin (bench/baselines/INSTALL.md)."
        + (" elastix counts a sample that maps outside the moving mask as invalid and stops when fewer than a quarter of its "
           "samples are valid (RequiredRatioOfValidSamples, 0.25 by default)."
           if any(x["tool"] == "elastix_mask" and "mean_mm" not in x for x in blm) else ""))
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
    return trim(out + errors_text([x for x in blm if x["crop"] in names]))


def near_section(bln):
    """The tools from a start near R, given as the header of the moving image (baselines.py --near)."""
    names = [crop_name(PRIMARY, "C3", k) for k in range(SEEDS)]
    bln = [x for x in bln if x["crop"] in names]
    st = list({x["crop"]: x["start_pose"] for x in bln if "start_pose" in x}.values())
    hs = list({x["crop"]: x["start_from_S"] for x in bln if "start_from_S" in x}.values())
    out = ["## Start near R", ""] + para(
        "The tools at their documented settings, started near R, on the C3 crops of the 10° FLASH. The start of crop k is R "
        f"turned by {NEAR_DEG:g}° about a random axis through the centre of the OCT tissue and shifted by {NEAR_MM:g} mm in a "
        f"random direction (numpy default_rng({NEAR_SEED} + k)). The tools get it as the header of the moving image, with the "
        "option that starts from the header alignment (bench/baselines/standard_tools.py). A NIfTI qform cannot hold the "
        "shear of R (ANTs, elastix and greedy read the qform), so the start is the pose with a rotation and a scale per axis "
        f"closest to the turned and shifted R over the OCT tissue: {rng([h['mean_mm'] for h in hs])} mm from it (mean over "
        f"the tissue, at most {fmt(max([h['max_mm'] for h in hs], default=None))} mm), {rng([x['mean_mm'] for x in st])} mm "
        f"and {rng([x['rotation_deg'] for x in st], 1)}° from R. The start row is that pose without registration.")
    out += TOOL_HEAD + [tool_row("the start", "no registration", st, spread=True)]
    for t in dict.fromkeys(x["tool"] for x in bln):
        r = [x for x in bln if x["tool"] == t]
        out.append(tool_row(t, r[0]["setting"], r, spread=True))
    return trim(out + [""] + errors_text(bln))


CONTROLS = (("near_mask", "the near start with the test's specimen mask as the moving mask, for the tools that take one, the "
                          "mask written with the OCT header of the start"),
            ("near_zeroed", "the near start on the moving OCT zeroed outside the specimen mask, every tool"),
            ("atR", "the start at R itself, the pose with a rotation and a scale per axis closest to R"),
            ("atR_mask", "the start at R with the specimen mask as the moving mask"))


def controls_section(groups):
    """The controls on the start near R (baselines.py --near-mask, --near-zeroed, --atR, --atR-mask), one table each."""
    names = [crop_name(PRIMARY, "C3", k) for k in range(SEEDS)]
    out = ["## Controls on the start near R", ""] + para(
        "The tools at their documented settings on the C3 crops of the 10° FLASH, from the start near R of the previous "
        "table or from R itself, with the test's specimen mask as the moving mask or with the moving OCT zeroed outside it "
        "(bench/i46/README.md). The start row of each table is its start without registration.")
    for kind, what in CONTROLS:
        rows = [x for x in groups.get(kind) or [] if x["crop"] in names]
        if not rows:
            continue
        st = list({x["crop"]: x["start_pose"] for x in rows if "start_pose" in x}.values())
        out += para(f"baselines.py --{kind.replace('_', '-')}: {what}.")
        out += TOOL_HEAD + [tool_row("the start", "no registration", st, spread=True)]
        for t in dict.fromkeys(x["tool"] for x in rows):
            r = [x for x in rows if x["tool"] == t]
            out.append(tool_row(t, r[0]["setting"], r, spread=True))
        out += [""] + errors_text(rows)
    return trim(out)


def check_section(*groups):
    """One convention-check table over the tool runs of every file."""
    rows = [x for g in groups for x in g]
    out = ["## Convention check", ""] + para(
        "Each converted transform against the tool's own resampling of the OCT it ran on (bench/baselines/standard_tools.py "
        "check), on the k = 0 crops of the first three tables and on every run from a header start: the Pearson r over the "
        "voxels where either is non-zero, the Dice of the two supports, and the number of non-zero voxels of each. Where few "
        "voxels of the registered OCT fall inside the crop, r and Dice are over those few. r is none where neither image has "
        "a voxel inside the crop and nan where one of them is zero on every voxel compared. elastix resamples with a cubic "
        "B-spline (FinalBSplineInterpolationOrder 3), which overshoots at the hard edge of the zeroed OCT and leaves small "
        "non-zero values around the tissue, so on that OCT its own resampling has more non-zero voxels than the converted one.")
    out += ["| tool | setting | crop | r | Dice | voxels: the tool's, the converted |", "|---|---|---|---|---|---|"]
    for x in rows:
        if "check_r" in x:
            v = x.get("check_voxels")
            out.append(f"| {x['tool']} | {x['setting']} | {x['crop']} | {fmt(x['check_r'])} | {fmt(x['check_dice'])} | "
                       + ("" if v is None else f"{v['tool']}, {v['converted']}") + " |")
    none = [f"{x['tool']} on {x['crop']}" for x in rows if x["crop"].endswith("_C3_k0") and "mean_mm" not in x]
    return trim(out + [""] + para("Runs on a k = 0 crop without a transform, so without a check: "
                                  + (", ".join(none) if none else "none") + "."))


def ablation_section(abl, rows):
    main = {x["name"]: x for x in rows if not x["oct_mask"]}
    out = ["## Ablation", ""] + para(
        "bench/ablate.py: the method with one change, on the C3 crops of the 10° FLASH, on the released OCT with octreg's own "
        "masks as in the main runs (analyses.py). The difference is the variant's distance from R minus the method's on the "
        "same crop. The variants of the outline are shown on I58 (bench/README.md). The vessel column gives the median "
        "distance of the MRI vessel labels to the nearest OCT vessel through the affine (common.vessel_score), median over "
        "the crops.")
    out += ["| variant | change | <= 1 mm | distance from R, median (range) (mm) | difference, median (range) (mm) "
            "| vessels (mm) |", "|---|---|---|---|---|---|"]
    crops = sorted({x["crop"] for x in abl})
    base = [main[c] for c in crops if c in main]
    out.append(f"| octreg | the method | {counts(base)} | {fmt(med([x['mean_mm'] for x in base]))} "
               f"({rng([x['mean_mm'] for x in base])}) | | {vessels(base)} |")
    for v in dict.fromkeys(x["variant"] for x in abl):
        r = [x for x in abl if x["variant"] == v]
        d = [dist(x) for x in r]
        diff = [dist(x) - main[x["crop"]]["mean_mm"] for x in r if x["crop"] in main and "mean_mm" in x]
        change = CHANGES[v].replace("|", r"\|")                      # | would end the table cell
        out.append(f"| {v} | {change} | {counts(r)} | {fmt(med(d))} ({rng(d)}) | {fmt(med(diff), 3)} ({rng(diff, 3)}) | "
                   f"{vessels(r)} |")
    return out


def main():
    missing = [f for f in ("reference_check.json", "context.json") if not (RESULTS / f).is_file()]
    if missing:
        sys.exit(f"missing in {RESULTS}: {', '.join(missing)} (prepare.py and reference_check.py write them)")
    rows = load("octreg.json") or []
    codes = {(x.get("version"), x.get("code")) for f in ("octreg.json", "ablation.json", "starts.json") for x in load(f) or []}
    if len(codes) > 1:
        sys.exit(f"octreg.json, ablation.json and starts.json mix octreg versions and codes {sorted(map(str, codes))}")
    st, abl = load("starts.json") or [], load("ablation.json") or []
    bl, bls, blm, bln = (load(f"baselines{s}.json") for s in ("", "_search", "_mask", "_near"))
    ctl = {kind: load(f"baselines_{kind}.json") for kind, _ in CONTROLS}
    ctl["near_zeroed"] = [{**x, "tool": x["tool"] + "_zeroed"} for x in ctl["near_zeroed"] or []]   # the near tools by name
    groups = [g for g in (bl, bls, blm, bln, *ctl.values()) if g]
    out = ["# I46: results", ""] + para(
        "Generated by bench/i46/summarise.py from the JSON files next to this one, and for the start rows, the scales of R "
        "and the residual of L from the inputs of prepare.py. bench/i46/README.md describes the test. Distances are the mean "
        "displacement over the OCT tissue from the authors' registration R, or from the label-only fit L where named. Counts "
        "are numbers of runs: the ten crops of a condition are shifted boxes around the same block. Medians are over all runs "
        "of a row. A run that ended without a readable transform (a tool error) counts as beyond every other.")
    out += reference_section(load("reference_check.json"), load("context.json"),
                             rows + st + abl + [x for g in groups for x in g])
    if rows:
        out += octreg_section(rows, st) + [""]
    if st:
        out += starts_section(st) + [""]
    if bl and bls:
        out += tools_section(rows, bl, bls) + [""]
    if bl and blm:
        out += mask_section(bl, blm) + [""]
    if bln:
        out += near_section(bln) + [""]
    if any(ctl.values()):
        out += controls_section(ctl) + [""]
    if groups:
        out += check_section(*groups) + [""]
    if abl:
        out += ablation_section(abl, rows) + [""]
    (RESULTS / "tables.md").write_text("\n".join(out).rstrip("\n") + "\n", encoding="utf-8")
    print("->", RESULTS / "tables.md")


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()

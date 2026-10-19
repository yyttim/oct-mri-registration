#!/usr/bin/env python3
"""Write bench/results/I46/tables.md from the JSON files in bench/results/I46, and the start rows and the scales of R from the
inputs of bench/i46/prepare.py.

    python bench/i46/summarize.py

    octreg.json                                 the octreg runs (run.py collect)
    baselines.json, baselines_search.json       the standard tools at their documented settings and with their own search
    baselines_mask.json, baselines_near.json    the tools with the test's specimen mask as moving mask, and from a start near R
    ablation.json, starts.json                  analyses.py collect
    reference_check.json, context.json          the reference and the contrast of the images
"""
from __future__ import annotations

import json
import re
import sys
import textwrap
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CONDITIONS, FLIPS, OCT, PRIMARY, RESULTS, SEEDS, TIERS, crop_file, crop_name, error,  # noqa: E402
                    reference)
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
    """The runs of rows within each distance of TIERS, as "k/n"."""
    d = [dist(x) for x in rows]
    return [f"{sum(m <= t for m in d)}/{len(d)}" for t in TIERS]


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


def reference_section(ref, ctx):
    a = ref["labels"]["A12"]
    th = " / ".join(f"{t['mean_mm']:.2f}" for t in a["by_depth_third"])
    out = ["## Reference", ""] + para(
        "R is the authors' registration, from the OCT world to the MRI world in the test's OCT frame. The label-only fit "
        f"L (reference_check.py, an affine fitted to the vessels alone) lies {a['mean_mm']:.3f} mm from R on average "
        f"(p95 {a['p95_mm']:.2f} mm, rotation {a['rotation_deg']:.2f}°), {th} mm in the upper, middle and deepest third "
        f"of the OCT depth, so distances below {TIERS[0]:.2f} mm are not interpreted.")
    return out + para("Grey/white contrast-to-noise ratio of the images inside the box of the OCT: " + ", ".join(
        f"{FLIPS[k][1]}° FLASH {v:.2f}" if k in FLIPS else f"T1 map {v:.2f}"
        for k, v in ctx["grey_white_cnr"].items() if k != "box_voxels") + ".")


def octreg_section(rows):
    out = ["## octreg", ""] + para(
        "octreg at its default parameters on the released OCT, with its own specimen mask and MRI foreground (§1). The last "
        "row repeats the first with the test's specimen mask as --oct-mask in place of octreg's own.")
    out += ["| condition | <= 0.30 mm | <= 1 mm | distance from R, median (range) (mm) | from L, median (mm) "
            "| rotation from R (°) |", "|---|---|---|---|---|---|"]
    for im, cond, masked in conditions():
        r = [x for x in rows if x["image"] == im and x["cond"] == cond and x["oct_mask"] == masked]
        d = [x["mean_mm"] for x in r]
        out.append(f"| {label(im, cond, masked)} | " + " | ".join(counts(r)) + f" | {fmt(med(d))} ({rng(d)}) | "
                   f"{fmt(med([x['mean_L_mm'] for x in r if 'mean_L_mm' in x]))} | {rng([x['rotation_deg'] for x in r], 1)} |")
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
    return trim(out)


def starts_section(st):
    out = ["## Start test on flip1_C3_k0", ""] + para(
        "The OCT is turned about the centre of its image box, by 90° and 180° about each array axis (i90 to k180) and by "
        f"{N_RANDOM} rotations drawn uniformly on SO(3) (rand00 to rand{N_RANDOM - 1:02d}, seed {SEED}), and octreg runs from "
        "each turned start (analyses.py starts). The start column gives the distance and rotation from R of the turned OCT "
        "with the centres of the two image boxes aligned.")
    for group, pre in (("turned", ("i", "j", "k")), ("random", ("rand",))):
        r = [x for x in st if x["start"].startswith(pre)]
        out += para(f"{group} starts: within 1 mm {counts(r)[1]}, distance from R {rng([x['mean_mm'] for x in r])} mm, "
                    f"rotation {rng([x['rotation_deg'] for x in r], 1)}°.")
    out += ["| start | the start: distance from R (mm), rotation (°) | octreg (mm) |", "|---|---|---|"]
    for x in st:
        s = x["start_pose"]
        out.append(f"| {x['start']} | {s['mean_mm']:.1f}, {s['rotation_deg']:.0f} | {x['mean_mm']:.2f} |")
    return out


TOOL_HEAD = ["| method | setting | <= 0.30 mm | <= 1 mm | tool errors | distance from R, median (mm) | rotation, median (°) |",
             "|---|---|---|---|---|---|---|"]


def tool_row(method, setting, r, spread=False):
    """One row of TOOL_HEAD over the runs r, the median distance with its range if spread."""
    d = [dist(x) for x in r]
    return (f"| {method} | {setting} | " + " | ".join(counts(r)) + f" | {sum('mean_mm' not in x for x in r)} | "
            f"{fmt(med(d))}{f' ({rng(d)})' if spread else ''} | {fmt(med([x.get('rotation_deg', INF) for x in r]), 0)} |")


def tools_section(rows, bl, bls):
    names = [crop_name(PRIMARY, "C3", k) for k in range(SEEDS)]
    out = ["## Standard tools", ""] + para(
        "Each tool at its documented setting (bench/baselines/standard_tools.py), and the three tools that have a global "
        "rotation search with that search turned on, on the C3 crops of the 10° FLASH. The moving image is the released OCT "
        "box-averaged to 0.15 mm, the fixed image the crop. The start is the pose before any registration: the header "
        "orientation of the OCT with the centres of the two image boxes aligned.")
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
        msgs = list(dict.fromkeys(re.sub(r"^Description:\s*", "", x.get("log_error") or x["error"][0]) for x in r))
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


def check_section(*groups):
    """One convention-check table over the tool runs of every file."""
    rows = [x for g in groups for x in g]
    out = ["## Convention check", ""] + para(
        "On the k = 0 crops, each converted transform against the tool's own resampling of the OCT it ran on "
        "(bench/baselines/standard_tools.py check): the Pearson r over the voxels where either is non-zero, the Dice of the "
        "two supports, and the number of non-zero voxels of each.")
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
    vf = lambda r: fmt(med([x["vessel_median_mm"] for x in r if x.get("vessel_median_mm") is not None]), 3)
    out.append(f"| octreg | the method | {counts(base)[1]} | {fmt(med([x['mean_mm'] for x in base]))} "
               f"({rng([x['mean_mm'] for x in base])}) | | {vf(base)} |")
    for v in dict.fromkeys(x["variant"] for x in abl):
        r = [x for x in abl if x["variant"] == v]
        d = [dist(x) for x in r]
        diff = [dist(x) - main[x["crop"]]["mean_mm"] for x in r if x["crop"] in main and "mean_mm" in x]
        change = CHANGES[v].replace("|", r"\|")                      # | would end the table cell
        out.append(f"| {v} | {change} | {counts(r)[1]} | {fmt(med(d))} ({rng(d)}) | {fmt(med(diff), 3)} ({rng(diff, 3)}) | "
                   f"{vf(r)} |")
    return out


def main():
    missing = [f for f in ("reference_check.json", "context.json") if not (RESULTS / f).is_file()]
    if missing:
        sys.exit(f"missing in {RESULTS}: {', '.join(missing)} (prepare.py and reference_check.py write them)")
    rows = load("octreg.json") or []
    codes = {(x.get("version"), x.get("code")) for f in ("octreg.json", "ablation.json", "starts.json") for x in load(f) or []}
    if len(codes) > 1:
        sys.exit(f"octreg.json, ablation.json and starts.json mix octreg versions and codes {sorted(map(str, codes))}")
    bl, bls, blm, bln = (load(f"baselines{s}.json") for s in ("", "_search", "_mask", "_near"))
    out = ["# I46: results", ""] + para(
        "Generated by bench/i46/summarize.py from the JSON files next to this one, and for the start rows and the scales of R "
        "from the inputs of prepare.py. bench/i46/README.md describes the test. Distances are the mean displacement over the "
        "OCT tissue from the authors' registration R, or from the label-only fit L where named. Counts are numbers of runs: "
        "the ten crops of a condition are shifted boxes around the same block. Medians are over all runs of a row. A run that "
        "ended without a readable transform (a tool error) counts as beyond every other.")
    out += reference_section(load("reference_check.json"), load("context.json"))
    if rows:
        out += octreg_section(rows) + [""]
    if load("starts.json"):
        out += starts_section(load("starts.json")) + [""]
    if bl and bls:
        out += tools_section(rows, bl, bls) + [""]
    if bl and blm:
        out += mask_section(bl, blm) + [""]
    if bln:
        out += near_section(bln) + [""]
    if any((bl, bls, blm, bln)):
        out += check_section(*(g for g in (bl, bls, blm, bln) if g)) + [""]
    if load("ablation.json"):
        out += ablation_section(load("ablation.json"), rows) + [""]
    (RESULTS / "tables.md").write_text("\n".join(out).rstrip("\n") + "\n", encoding="utf-8")
    print("->", RESULTS / "tables.md")


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()

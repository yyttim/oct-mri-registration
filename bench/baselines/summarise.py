#!/usr/bin/env python3
"""Tables of the baseline comparison from the summary JSON files (bench only, not part of the package).

    python bench/baselines/summarise.py [--affine SUMMARY.json] [--deform SUMMARY.json] [--notes NOTES.json] [--out DIR]

Writes DIR/affine.md and DIR/deform.md (default bench/results/I58/baselines/) and copies the two summary files and
NOTES.json next to them. The summaries are those of evaluate_runs.py and evaluate_deform.py. NOTES.json maps each run of
each stage (keys affine and deform) to the tool and its version, the setting and the start, written from the run's command.
Each table has a main section, one row per tool at the setting its documentation gives for this kind of registration, FLIRT with
both headers scaled x10 as its schedule works in mm (MAIN_AFFINE, MAIN_DEFORM, the panels of bench/figures.py), and a
supplement with every stored run. The columns are properties that need no reference: for an affine the scale along each OCT
array axis and the determinant, for a field the grid, the minimum of the Jacobian determinant J, the share of voxels with J <= 0
and the standard deviation of log J, and for both the wall time and the hardware.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("OCTREG_PROJECT_ROOT") or REPO)

# panel of bench/figures.py -> run, in the order of its figures
MAIN_AFFINE = [("start", "reference_centres"), ("octreg", "octreg"), ("reg_aladin", "niftyreg_center"),
               ("greedy", "greedy_centers"), ("elastix", "elastix_center"),
               ("mri_robust_register", "mri_robust_register_default_com"), ("ants", "ants_com"),
               ("mri_coreg", "mri_coreg_default"), ("flirt", "flirt_search_x10"), ("synthmorph", "synthmorph_affine")]
MAIN_DEFORM = [("affine", "none"), ("octreg", "octreg"), ("ants_syn", "deform_ants_syn_cc"),
               ("elastix", "deform_elastix_default"), ("reg_f3d", "deform_niftyreg_default"),
               ("convexadam", "deform_convexadam_default"), ("greedy", "deform_greedy_wncc")]


def wall(s):
    if s is None:
        return ""
    return f"{s:.0f} s" if s < 120 else f"{s / 60:.0f} min" if s < 7200 else f"{s / 3600:.1f} h"


def opt(v, fmt):
    return "" if v is None else fmt.format(v)


def note(notes, run):
    n = notes.get(run)
    if n is None:
        raise KeyError(f"no entry for run {run} in the notes file")
    return n


def affine_line(r, n):
    sc = " / ".join(f"{x:.2f}" for x in r["scale_per_oct_axis"])
    return (f"| {n['tool']} | `{r['run']}` | {n['setting']} | {n['start']} | {sc} | {r['det']:.2f} | {wall(r.get('wall_s'))} | "
            f"{r.get('hardware', '')} |")


def folded(fraction):
    """The share of folded voxels in per cent, '< 0.01 %' for a share that is not 0 but rounds to it."""
    if fraction is None:
        return ""
    return "< 0.01 %" if 0 < 100 * fraction < 0.005 else f"{100 * fraction:.2f} %"


def deform_line(r, n):
    return (f"| {n['tool']} | `{r['run']}` | {n['setting']} | {n['start']} | {opt(r.get('field_grid_mm'), '{:.2f} mm')} | "
            f"{opt(r.get('jac_min'), '{:.2f}')} | {folded(r.get('jac_folding_fraction'))} | "
            f"{opt(r.get('sd_log_jac'), '{:.2f}')} | {wall(r.get('wall_s'))} | {r.get('hardware', '')} |")


def table(rows, notes, main, head, line, title, intro):
    missing = [run for _, run in main if run not in rows]
    if missing:
        raise KeyError(f"main runs without a summary row: {missing}")
    first = ("reference_centres", "none")
    rest = sorted(rows.values(), key=lambda r: (r["run"] not in first, r["run"] != "octreg", r["tool"], r["run"]))
    intro = "\n".join(textwrap.wrap(intro, 124, break_on_hyphens=False, break_long_words=False))
    out = [f"# {title}", "", intro, "", "## Main: each tool at the setting its documentation gives (FLIRT with both headers scaled x10)", "", *head,
           *[line(rows[run], note(notes, run)) for _, run in main], "", "## Supplement: every stored run", "", *head,
           *[line(r, note(notes, r["run"])) for r in rest]]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--affine", type=Path, default=ROOT / "baselines" / "summary_affine.json")
    ap.add_argument("--deform", type=Path, default=ROOT / "baselines" / "summary_deform.json")
    ap.add_argument("--notes", type=Path, default=ROOT / "baselines" / "notes.json")
    ap.add_argument("--out", type=Path, default=REPO / "bench" / "results" / "I58" / "baselines")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    notes = json.loads(a.notes.read_text(encoding="utf-8"))
    ra = json.loads(a.affine.read_text(encoding="utf-8"))
    rd = json.loads(a.deform.read_text(encoding="utf-8"))
    head_a = ["| method | run | setting | start | scale per OCT axis | det | wall time | hardware |",
              "|---|---|---|---|---|---|---|---|"]
    head_d = ["| method | run | setting | start | field grid | Jacobian min | folded (J <= 0) | SD log J | wall time | "
              "hardware |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    intro_a = ("No ground truth exists for this pair, and the table holds no accuracy column: the overlays of BASELINES.md "
               "are the result. Scale per OCT axis: the length of the transform applied to a unit step along each array axis "
               "of the OCT header (1 = voxel size kept). det: the determinant of its 3x3 (1 = volume kept). Wall times as each "
               "run recorded them, on a 24-core machine shared with other jobs, and only indicative. octreg's is the sum of "
               "its §1 to §5 step times in result.json. FLIRT's main run has both headers scaled x10, as its schedule works in mm, "
               "and its run on the unscaled headers is flirt_search_native.")
    intro_d = ("Every tool starts from octreg's affine: the OCT resampled through it onto the 0.08 mm MRI grid, and the "
               "identity as initial transform. No ground truth exists for this pair, and the overlays of BASELINES.md are the "
               "result. J is the Jacobian determinant of x + u(x), u the field, by finite differences on the field grid, "
               "over the MRI foreground of octreg's §1 (a mask of the MRI alone, on the 0.15 mm grid) sampled onto that grid. "
               "folded: the share of those voxels with J <= 0. SD log J: the standard deviation of log J, with J clipped at 1e-6 "
               "where it is not positive (Learn2Reg, Hering "
               "et al., IEEE Trans. Med. Imaging 42:697, 2023). Wall times as each run recorded them, on a machine shared with "
               "other jobs, and only indicative. octreg's is its §6 step time in result.json.")
    for name, rows, main_rows, head, line, title, intro in (
            ("affine", ra, MAIN_AFFINE, head_a, affine_line, "I58, affine stage", intro_a),
            ("deform", rd, MAIN_DEFORM, head_d, deform_line, "I58, deformable stage", intro_d)):
        (a.out / f"{name}.md").write_text(table(rows, notes[name], main_rows, head, line, title, intro), encoding="utf-8",
                                          newline="\n")
        src = a.affine if name == "affine" else a.deform
        if src.resolve() != (a.out / f"summary_{name}.json").resolve():
            shutil.copy(src, a.out / f"summary_{name}.json")
        print(f"{name}: {len(rows)} rows -> {a.out / (name + '.md')}")
    if a.notes.resolve() != (a.out / "notes.json").resolve():
        shutil.copy(a.notes, a.out / "notes.json")


if __name__ == "__main__":
    main()

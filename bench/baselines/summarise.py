#!/usr/bin/env python3
"""The table of the I58 baseline comparison from the summary JSON file.

    python bench/baselines/summarise.py [--affine SUMMARY.json] [--notes NOTES.json] [--out DIR]

Writes DIR/affine.md (default bench/results/I58/baselines/) and copies the summary of evaluate_runs.py next to it.
NOTES.json (default bench/results/I58/baselines/notes.json) gives the tool, version, setting and start of each run (key
affine). The table has the main runs, one per tool at the setting its documentation gives (the panels of AFFINE in
bench/figures.py), and, when there are more, a supplement with every stored run.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.figures import AFFINE  # noqa: E402
from bench.paths import PROJECT_ROOT as ROOT, REPO  # noqa: E402

MAIN = ["octreg" if sub.startswith("RUN/") else "_".join(sub.split("/")[1:]) for _, sub in AFFINE]
HEAD = ["| method | run | setting | start | scale per OCT axis | det | wall time | hardware |",
        "|---|---|---|---|---|---|---|---|"]
INTRO = ("The pair has no reference, so the table has no accuracy column, and BASELINES.md shows the overlays. "
         "Scale per OCT axis: the length of the transform applied to a unit step along each array axis "
         "of the OCT header (1 = voxel size kept). det: the determinant of its 3x3 (1 = volume kept). Wall times as each "
         "run recorded them. octreg's is the sum of "
         "its §1 to §4 step times in result.json. FLIRT's main run has both headers scaled x10, because its schedule "
         "resamples the images to fixed voxel sizes in mm, and its run on the unscaled headers is flirt_search_native.")
MAIN_TITLE = "Main: each tool at the setting its documentation gives (FLIRT with both headers scaled x10)"


def wall(s):
    if s is None:
        return ""
    return f"{s:.0f} s" if s < 120 else f"{s / 60:.0f} min" if s < 7200 else f"{s / 3600:.1f} h"


def note(notes, run):
    n = notes.get(run)
    if n is None:
        raise KeyError(f"no entry for run {run} in the notes file")
    return n


def line(r, n):
    sc = " / ".join(f"{x:.2f}" for x in r["scale_per_oct_axis"])
    return (f"| {n['tool']} | `{r['run']}` | {n['setting']} | {n['start']} | {sc} | {r['det']:.2f} | {wall(r.get('wall_s'))} | "
            f"{r.get('hardware', '')} |")


def table(rows, notes):
    missing = [run for run in MAIN if run not in rows]
    if missing:
        raise KeyError(f"main runs without a summary row: {missing}")
    rest = sorted(rows.values(), key=lambda r: (r["run"] != "reference_centres", r["run"] != "octreg", r["tool"], r["run"]))
    intro = "\n".join(textwrap.wrap(INTRO, 124, break_on_hyphens=False, break_long_words=False))
    out = ["# I58 baselines", "", intro, "", f"## {MAIN_TITLE}", "", *HEAD,
           *[line(rows[run], note(notes, run)) for run in MAIN]]
    if len(rows) > len(MAIN):
        out += ["", "## Supplement: every stored run", "", *HEAD, *[line(r, note(notes, r["run"])) for r in rest]]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--affine", type=Path, default=ROOT / "baselines" / "summary_affine.json")
    ap.add_argument("--notes", type=Path, default=REPO / "bench" / "results" / "I58" / "baselines" / "notes.json")
    ap.add_argument("--out", type=Path, default=REPO / "bench" / "results" / "I58" / "baselines")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    notes = json.loads(a.notes.read_text(encoding="utf-8"))
    rows = json.loads(a.affine.read_text(encoding="utf-8"))
    (a.out / "affine.md").write_text(table(rows, notes["affine"]), encoding="utf-8", newline="\n")
    if a.affine.resolve() != (a.out / "summary_affine.json").resolve():
        shutil.copy(a.affine, a.out / "summary_affine.json")
    print(f"affine: {len(rows)} rows -> {a.out / 'affine.md'}")


if __name__ == "__main__":
    main()

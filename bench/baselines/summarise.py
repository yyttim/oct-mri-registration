#!/usr/bin/env python3
"""The tables of the I58 results: the baseline comparison from the summary JSON file, and the ablation runs.

    python bench/baselines/summarise.py [--affine SUMMARY.json] [--notes NOTES.json] [--out DIR] [--ablation DIR]

Writes DIR/affine.md (default bench/results/I58/baselines/) and copies the summary of evaluate_runs.py next to it.
NOTES.json (default bench/results/I58/baselines/notes.json) gives the tool, version, setting and start of each run (key
affine). The table has the main runs, one per tool at the setting its documentation gives (the panels of AFFINE in
bench/figures.py), and, when there are more, a supplement with every stored run.
From the bench/ablate.py runs under --ablation (default <project root>/bench_runs/I58/ablation, turned/* included), when
that directory exists: bench/results/I58/ablation.json, the pose and the search candidates of every run, no accuracy, and
ablation/no_outline.json, the result.json of that run with the input paths cut to file names. Then, from result.json and
ablation.json in bench/results/I58, search.md: the candidates of the orientation search in the I58 run and in the
no_outline and outline_decides variants.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import textwrap
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.ablate import VARIANTS, code_id, made_by  # noqa: E402
from bench.figures import AFFINE  # noqa: E402
from bench.paths import BENCH_RUNS, PROJECT_ROOT as ROOT, REPO  # noqa: E402

I58 = REPO / "bench" / "results" / "I58"
MAIN = ["octreg" if sub.startswith("RUN/") else "_".join(sub.split("/")[1:]) for _, sub in AFFINE]
HEAD = ["| method | run | setting | start | scale per OCT axis | det | rotation from the header (°) | wall time | hardware |",
        "|---|---|---|---|---|---|---|---|---|"]
INTRO = ("The pair has no reference, so the table has no accuracy column, and BASELINES.md shows the overlays. "
         "Scale per OCT axis: the length of the transform applied to a unit step along each array axis "
         "of the OCT header (1 = voxel size kept). det: the determinant of its 3x3 (1 = volume kept). Rotation from the "
         "header: the angle of the rotation part of the transform from the header orientation, that of the first row. The "
         "summary JSON next to this table also holds each transform, the shift of the OCT box centre from the start (mm) "
         "and the correlation and support Dice of its check (rundir.py). Wall times as each run recorded them. octreg's is "
         "the sum of its §1 to §4 step times in result.json. FLIRT's main run has both headers scaled x10, because its "
         "schedule resamples the images to fixed voxel sizes in mm, and its run on the unscaled headers is "
         "flirt_search_native.")
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
    return (f"| {n['tool']} | `{r['run']}` | {n['setting']} | {n['start']} | {sc} | {r['det']:.2f} | "
            f"{r['rotation_from_header_deg']:.1f} | {wall(r.get('wall_s'))} | {r.get('hardware', '')} |")


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


def ablation_rows(abl):
    """One row per bench/ablate.py run under abl, turned/* included: the pose and the search candidates, no accuracy."""
    code, rows = code_id(), []
    for run in sorted(p.parent for p in abl.glob("**/variant.json")):
        if made_by(run) != code:
            sys.exit(f"{run}: incomplete or not made by the octreg code {code}")
        res = json.loads((run / "result.json").read_text(encoding="utf-8"))
        v = json.loads((run / "variant.json").read_text(encoding="utf-8"))
        T, start = np.loadtxt(run / "T_oct2mri.txt"), run / "T_start.txt"
        rows.append({"run": run.relative_to(abl).as_posix(), "variant": v["variant"], "turn_deg": v["turn_deg"],
                     "version": res["octreg_version"], "code": code, "T_oct2mri": np.round(T, 10).tolist(),
                     **({"T_start": np.round(np.loadtxt(start), 10).tolist()} if start.is_file() else {}),
                     "scale_per_oct_axis": res["pose"]["scale_per_oct_axis"], "det": float(np.linalg.det(T[:3, :3])),
                     **{k: res["pose"][k] for k in ("search_rank", "search_F", "search_O", "L", "outline_used")},
                     "search_topk": res["params"]["topk"], "candidates": res["candidates"]})
    return rows


def cut_paths(res):
    """A result.json with the input paths cut to file names, as the stored copy of the I58 run."""
    for v in res["inputs"].values():
        if isinstance(v, dict):
            v["path"] = Path(v["path"]).name
    return res


def num(x, nd):
    return "" if x is None else f"{x:.{nd}f}"


def proposed_by_s(variant, topk, outline_used):
    """The number of first ranks proposed by S: none without the outline, all in outline_decides, else half."""
    return 0 if not outline_used else topk if variant == "outline_decides" else topk // 2


def search_table(cands, winner, by_s):
    """The candidates of one run in the order of result.json. The first by_s ranks were proposed by S, the rest by F."""
    out = ["| rank | proposed by | search F | search O | final L | winner |", "|---|---|---|---|---|---|"]
    for c in cands:
        out.append(f"| {c['search_rank']} | {'S' if c['search_rank'] < by_s else 'F'} | {num(c['search_F'], 4)} | "
                   f"{num(c['search_O'], 4)} | {num(c['L'], 5)} | {'yes' if c['search_rank'] == winner else ''} |")
    return out


def search_md(res, abl):
    """search.md from the stored result.json of the I58 run and the rows of ablation_rows."""
    k, pose = res["params"]["topk"], res["pose"]
    runs = [("the I58 run", res["candidates"], pose["search_rank"], proposed_by_s("base", k, pose["outline_used"]))]
    runs += [(f"{r['variant']}: {VARIANTS[r['variant']]}", r["candidates"], r["search_rank"],
              proposed_by_s(r["variant"], r["search_topk"], r["outline_used"]))
             for r in abl if r["variant"] in ("no_outline", "outline_decides")]
    intro = textwrap.wrap(
        "Generated by bench/baselines/summarise.py from result.json and ablation.json next to this file: the poses the "
        "orientation search keeps (docs/METHOD.md §3) in the order of result.json, by final L after the refinement (§4), "
        "the poses within 10⁻⁵ of the lowest first by rank. rank: the order in which the search kept the pose. proposed "
        f"by: S = (F + O) / 2 for the first {k // 2} ranks and F for the rest where the outline is used, F for all {k} in "
        f"no_outline and S for all {k} in outline_decides. search F and search O: at the search pose. final L: after the "
        "refinement. The winner is the pose with the lowest final L, among equals the one proposed first.", 124)
    out = ["# I58: the search candidates", "", *intro]
    for title, cands, winner, b in runs:
        out += ["", f"## {title}", "", *search_table(cands, winner, b)]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--affine", type=Path, default=ROOT / "baselines" / "summary_affine.json")
    ap.add_argument("--notes", type=Path, default=I58 / "baselines" / "notes.json")
    ap.add_argument("--out", type=Path, default=I58 / "baselines")
    ap.add_argument("--ablation", type=Path, default=BENCH_RUNS / "I58" / "ablation",
                    help="the bench/ablate.py runs of the I58 pair, for ablation.json and search.md")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    notes = json.loads(a.notes.read_text(encoding="utf-8"))
    rows = json.loads(a.affine.read_text(encoding="utf-8"))
    (a.out / "affine.md").write_text(table(rows, notes["affine"]), encoding="utf-8", newline="\n")
    if a.affine.resolve() != (a.out / "summary_affine.json").resolve():
        shutil.copy(a.affine, a.out / "summary_affine.json")
    print(f"affine: {len(rows)} rows -> {a.out / 'affine.md'}")
    if a.ablation.is_dir():
        abl = ablation_rows(a.ablation)
        (I58 / "ablation.json").write_text(json.dumps(abl, indent=1), encoding="utf-8")
        (I58 / "ablation").mkdir(exist_ok=True)
        res = cut_paths(json.loads((a.ablation / "no_outline" / "result.json").read_text(encoding="utf-8")))
        (I58 / "ablation" / "no_outline.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"ablation: {len(abl)} runs -> {I58 / 'ablation.json'}")
    if (I58 / "ablation.json").is_file():
        res, abl = (json.loads((I58 / f).read_text(encoding="utf-8")) for f in ("result.json", "ablation.json"))
        (I58 / "search.md").write_text(search_md(res, abl), encoding="utf-8", newline="\n")
        print(f"search: -> {I58 / 'search.md'}")


if __name__ == "__main__":
    main()

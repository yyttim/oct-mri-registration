#!/usr/bin/env python3
"""Markdown tables of the baseline comparison from the summary JSON files (bench only, not part of the package).

    python bench/baselines/summarise.py [--affine SUMMARY.json] [--deform SUMMARY.json] [--out DIR] [--notes NOTES.json]

Writes DIR/affine.md and DIR/deform.md (default bench/results/I58/baselines/) and copies the two summary files next to
them. NOTES.json, optional, maps a run name to a short description for the note column (the variant and its outcome).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("OCTREG_PROJECT_ROOT") or REPO)


def f(v, nd=2, unit=""):
    return "n/a" if v is None else f"{v:.{nd}f}{unit}"


def affine_table(rows, notes):
    order = sorted(rows.values(), key=lambda r: (r["tool"], r["variant"]))
    head = ("| tool | variant | wall s | scale per OCT axis | rim OCT to MRI / MRI to OCT (mm) | OCT boundary p90 (mm) | Dice | "
            "to octreg: mean mm / deg | frame check | note |")
    out = [head, "|---|---|---|---|---|---|---|---|---|---|"]
    for r in order:
        sc = r.get("scale_per_oct_axis")
        sc = " / ".join(f"{x:.3f}" for x in sc) if sc else "n/a"
        out.append(f"| {r['tool']} | {r['variant']} | {f(r.get('wall_s'), 0)} | {sc} | {f(r.get('rim_fwd_mm'))} / {f(r.get('rim_rev_mm'))} | "
                   f"{f(r.get('fwd_all_p90_mm'))} | {f(r.get('dice'), 3)} | {f(r.get('to_octreg_mean_mm'), 1)} / "
                   f"{f(r.get('to_octreg_rotation_deg'), 1)} | {'pass' if r.get('frame_ok') else 'fail'} | {notes.get(r['run'], '')} |")
    return "\n".join(out)


def deform_table(rows, notes):
    order = sorted(rows.values(), key=lambda r: (r["tool"] != "none", r["tool"] != "octreg", r["tool"], r["variant"]))
    head = ("| tool | variant | wall s | interior matches (mm) | surface-edge offsets (mm) | within 0.3 mm | F | two-class, core | "
            "field median / p95 / max (mm) | Jacobian min | folded voxels | SD log J | note |")
    out = [head, "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in order:
        fld = "0" if not r.get("field_max_mm") else f"{f(r['field_median_mm'])} / {f(r.get('field_p95_mm'))} / {f(r['field_max_mm'])}"
        out.append(f"| {r['tool']} | {r['variant']} | {f(r.get('wall_s'), 0)} | {f(r['interior_mm'], 3)} | {f(r['boundary_mm'], 3)} | "
                   f"{100 * r['within']:.0f} % | {f(r['F'], 4)} | {f(r['two_class_core'], 4)} | {fld} | "
                   f"{f(r.get('jac_min'), 2) if r.get('field_max_mm') else '1'} | "
                   f"{f(100 * r['jac_folding_fraction'], 2, ' %') if r.get('field_max_mm') else '0 %'} | "
                   f"{f(r.get('sd_log_jac'), 3) if r.get('field_max_mm') else '0'} | {notes.get(r['run'], '')} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--affine", type=Path, default=ROOT / "baselines" / "summary_affine.json")
    ap.add_argument("--deform", type=Path, default=ROOT / "baselines" / "summary_deform.json")
    ap.add_argument("--notes", type=Path, default=ROOT / "baselines" / "notes.json")
    ap.add_argument("--out", type=Path, default=REPO / "bench" / "results" / "I58" / "baselines")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    notes = json.loads(a.notes.read_text(encoding="utf-8")) if a.notes.is_file() else {}
    for src, fn, name in ((a.affine, affine_table, "affine"), (a.deform, deform_table, "deform")):
        if not src.is_file():
            continue
        rows = json.loads(src.read_text(encoding="utf-8"))
        with open(a.out / f"{name}.md", "w", encoding="utf-8", newline="\n") as fh:
            fh.write(fn(rows, notes) + "\n")
        shutil.copy(src, a.out / f"summary_{name}.json")
        print(f"{name}: {len(rows)} rows -> {a.out / (name + '.md')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Write bench/BENCHMARK.md for the I58 pair from the bench outputs (formatting only, no computation).

    python bench/report.py --main RUN --ablate ABL [--logs RUN_logs] [-o bench/BENCHMARK.md] [--figures bench/figures]
        [--store bench/results/I58]

RUN: the CLI run (result.json, and eval.json of bench/evaluate.py). ABL: the output of bench/ablate.py (ablations.json, and
starts.json with --starts). RUN_logs: the step logs of bench/run_i58.py (register.time and register.gpu_mib), for the wall
clock and the peak GPU memory of the device. Missing values print n/a. A rewrite keeps the hand-written '## Visual result'
section and the text after the READING marker.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path, PureWindowsPath

READING = "<!-- reading: written by hand below this line, and bench/report.py keeps it when it rewrites the file -->"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if path and Path(path).exists() else {}


def portable(o):
    """o with every absolute path cut to its part from bench_runs/ on or, outside the runs, to its file name."""
    if isinstance(o, dict):
        return {k: portable(v) for k, v in o.items()}
    if isinstance(o, list):
        return [portable(v) for v in o]
    if isinstance(o, str) and (PureWindowsPath(o).is_absolute() or o.startswith("/")):
        parts = PureWindowsPath(o).parts                           # splits on / and on \
        return "/".join(parts[parts.index("bench_runs"):]) if "bench_runs" in parts else parts[-1]
    return o


def wrap(lines, width=128):
    """Prose lines (not tables, headings, comments, images or fenced code) longer than width, wrapped at width."""
    import textwrap
    out, code = [], False
    for line in lines:
        if line.startswith("```"):
            code = not code
        if code or line.startswith(("```", "<img", "![")):
            out.append(line)
            continue
        if len(line) > width and not line.startswith(("|", "#", "<!--")):
            out += textwrap.wrap(line, width, break_on_hyphens=False, break_long_words=False)
        else:
            out.append(line)
    return out


def num(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def wall_seconds(path):
    """Wall-clock seconds from a NAME.time file of bench/run_i58.py (the wall-clock line of GNU time -v), or None."""
    t = Path(path).read_text(encoding="utf-8") if path and Path(path).exists() else ""
    wall = re.search(r"Elapsed \(wall clock\) time.*?: ([\d:.]+)", t)
    return sum(float(p) * 60 ** i for i, p in enumerate(reversed(wall.group(1).split(":")))) if wall else None


def gpu_peak_gb(path):
    vals = [float(v) for v in Path(path).read_text(encoding="utf-8").split()] if path and Path(path).exists() else []
    return max(vals) / 1024 if vals else None


def rims(b):
    """Rim medians forward / reverse at the pose."""
    r = (b or {}).get("rim_median_mm")
    return f"{num(r[0])} / {num(r[1])} mm" if r else "n/a"


def arrow(pair, scale=1.0, nd=2):
    """'before -> after' of a [before, after] read-out."""
    return " -> ".join(num(None if x is None else scale * x, nd) for x in (pair or [None, None]))


def deform_rows(d):
    """Table rows of the smooth deformation (§6) read-outs (result.json 'deform'), none when the run has no such block."""
    if not d:
        return []
    r, f, cv = d.get("residual", {}), d.get("field", {}), d.get("cv", {})
    model = f"control lattice {num(d.get('grid_mm'), 0)} mm, λ {num(d.get('lam'), 2)}" if d.get("status") == "applied" else "no field"
    return [("smooth deformation (§6): status, model, interior matches and surface-edge offsets used by the fit",
             f"{d.get('status', 'n/a')}, {model}, {d.get('n_interior', 'n/a')} and {d.get('n_boundary', 'n/a')}"),
            ("§6 held-out median error, interior matches + surface-edge offsets: no deformation -> the field",
             f"{arrow([cv['none'][0], cv['field'][0]], nd=3)} mm + {arrow([cv['none'][1], cv['field'][1]], nd=3)} mm"
             if cv.get("none") and cv.get("field") else "n/a"),
            ("§6 residuals affine -> deformed, measured again: interior matches, surface-edge offsets (within 0.3 mm)",
             f"{arrow(r.get('interior_mm'), nd=3)} mm, {arrow(r.get('boundary_mm'), nd=3)} mm ({arrow(r.get('boundary_within_0.3mm'), 100, 0)} %)"),
            ("§6 field over the MRI foreground, median / p95 / max, and max strain",
             f"{num(f.get('median_mm'))} / {num(f.get('p95_mm'))} / {num(f.get('max_mm'))} mm, {num(d.get('max_strain'), 3)}")]


def main_section(run, logs):
    res, ev = load(run / "result.json"), load(run / "eval.json")
    pose, srch = res.get("pose", {}), res.get("search", {})
    fc, b, mk = ev.get("frame_check", {}), ev.get("boundary", {}), ev.get("mask", {})
    wall = wall_seconds(logs / "register.time" if logs else None)
    gpu = gpu_peak_gb(logs / "register.gpu_mib" if logs else None)
    sec = res.get("seconds", {})
    ngf_s = res.get("ngf", {}).get("seconds")
    steps = ", ".join((f"search {num(srch.get('seconds'), 0)}, refinement {num(res['refine']['seconds'], 0)}"
                       + (f", fine-structure refinement {num(ngf_s, 0)}" if ngf_s is not None else "")
                       if k == "search_refine_ngf" and "refine" in res else
                       f"smooth deformation {num(v, 0)}" if k == "deform" else f"{k.replace('_', ' ')} {num(v, 0)}")
                      for k, v in sec.items() if k != "total")
    shifts = [v["spearman"] for k, v in fc.get("variants", {}).items() if k.startswith("shift") and v["spearman"] is not None]
    shift_max = max(shifts) if shifts else None
    rows = [
        ("raw-data frame check (Spearman)", f"{num(fc.get('spearman_export'), 3)} at the pose, axis flips <= "
                                            f"{num(fc.get('max_flip_spearman'), 3)}, 2 mm shifts <= {num(shift_max, 3)}, "
                                            f"{('pass' if fc['ok'] else 'fail') if fc else 'n/a'}"),
        ("score S of the §4 pose = (2 polarity S_class + S_outline) / 3, with S_class, S_outline and the polarity",
         f"{num(pose.get('S'), 4)}, {num(pose.get('S_class'), 4)}, {num(pose.get('S_outline'), 4)}, {pose.get('polarity', 'n/a')}"),
        ("search top-1 / top-2", f"{num(srch.get('top1'), 4)} / {num(srch.get('top2'), 4)}"),
        ("score F of §5, at the §4 pose -> at the result",
         f"{num(pose.get('NGF_start'), 4)} -> {num(pose.get('NGF'), 4)}"),
        ("pose change of §5 (block-corner mean)", f"{num(pose.get('ngf_shift_mm'))} mm"),
        ("scale per OCT array axis", " / ".join(f"{v:.3f}" for v in pose["scale_per_oct_axis"]) if pose else "n/a"),
        *deform_rows(res.get("deform")),
        ("flags", "n/a" if "flags" not in res else ", ".join(res["flags"]) or "none"),
        ("rim outline agreement forward / reverse, method masks", rims(b)),
        ("OCT specimen mask (base grid, intersected with the measured OCT)", f"{num(mk.get('volume_cm3'))} cm3"),
        ("registration time in process, peak RAM, peak GPU memory allocated by torch",
         f"{num(sec['total'] / 60 if 'total' in sec else None, 1)} min, {num(res.get('peak_rss_gb'), 1)} GiB, "
         f"{num(res.get('gpu_peak_gb'), 2)} GiB" + (f" (wall clock {num(wall / 60, 1)} min, GPU memory in use on the device during the run, all processes, "
                                                 f"{num(gpu, 1)} GiB)" if wall else "")),
        ("time per step (s)", steps or "n/a"),
    ]
    note = [] if ev else ["bench/evaluate.py has not been run on this run yet.", ""]
    return ["## Main result", "", "| | |", "|---|---|"] + [f"| {a} | {v} |" for a, v in rows] + [""] + note


def det3(T):
    """Determinant of the 3 x 3 part of a 4 x 4 nested list."""
    (a, b, c), (d, e, f), (g, h, i) = (row[:3] for row in T[:3])
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def ablation_row(n, r):
    """One row of the ablation table: the variant's own values. Every read-out is optional, so a partial table still prints."""
    cut = (r.get("spec") or {}).get("cut_axis") is not None             # the mask it ran with is cut, its volume not stored
    vol = "" if cut else num((r.get("mask") or {}).get("volume_cm3"))
    if "error" in r:
        return f"| {n} | {r['change'].replace('|', chr(92) + '|')} | failed: {r['error']} | | | | | {vol} | {num(r.get('seconds'), 0)} |"
    det = num(det3(r["T_oct2mri"]), 3) if r.get("T_oct2mri") else "n/a"
    return (f"| {n} | {r['change'].replace('|', chr(92) + '|')} | {num(r.get('S'), 4)} | {num(r.get('L'), 4)} | "
            f"{r.get('polarity', 'n/a')} | {' / '.join(f'{s:.3f}' for s in r.get('scales', [])) or 'n/a'} | {det} | "
            f"{vol} | {num(r.get('seconds'), 0)} |")


def ablation_section(abl):
    if not abl:
        return ["## Ablations", "", "Not run yet (bench/ablate.py).", ""]
    V = abl["variants"]
    cols = ("| variant | change | S (§4 pose) | L (§4 pose) | polarity | scale per OCT axis | determinant | OCT mask cm3 "
            "| time (s) |")
    rule = "|---|---|---|---|---|---|---|---|---|"
    head = ["## Ablations", "",
            "Each variant is the method with one change, run from the same preprocessed grids (one per OCT mask source). "
            "L is the loss of §4, 1 - S plus the scale and shear penalty (docs/METHOD.md). "
            "A11 and A11b cut the specimen mask, so their mask column is empty. No variant runs the smooth deformation (§6), "
            "which does not change the affine. Its ablation is in bench/results/I58/deform_ablation.md.", "",
            cols, rule]
    rows = [ablation_row(n, r) for n, r in V.items()]
    tail = [""]
    dc = abl.get("driver_check")
    if dc:
        tail += [f"Base, run through bench/ablate.py, lies {num(dc['base_vs_main']['mean_mm'])} mm (corners max "
                 f"{num(dc['base_vs_main']['corners_max_mm'])} mm) from the CLI run of the Main result.", ""]
    return head + rows + tail


def starts_section(st):
    """The start test of bench/ablate.py --starts: the method and the variant without the orientation search, run from turned
    starts."""
    if not st:
        return []
    out = ["## Start test", "",
           "The OCT is turned by 90 and 180 degrees about each of its array axes, through the centre of its image box, and the "
           "method runs from the image-centre start and from each turned start. Each cell gives the mean displacement from the "
           "base result (the method from the file orientation) over the base specimen mask, and the angle of the rotation "
           "between the two poses. A14, the variant without the orientation search, also runs from each start, and "
           "README.md shows it from the start turned by 90° about the sectioning axis.", "",
           "| start | start pose to base: mm / deg | method to base: mm / deg |", "|---|---|---|"]
    f = lambda d: f"{num(d.get('mean_mm'))} / {num(d.get('rotation_deg'), 1)}" if d else "n/a"
    for name, r in st["starts"].items():
        label = "image centres, header orientation" if r["axis"] is None else f"{r['degrees']} deg about OCT array axis {r['axis']}"
        out.append(f"| {label} | {f(r.get('start_to_base'))} | {f(r.get('method_to_base'))} |")
    return out + [""]


def runtime_section(abl):
    if not abl:
        return []
    out = ["## Runtime and memory of bench/ablate.py", "",
           "| step | seconds | peak RAM (GiB) |", "|---|---|---|"]
    grid = next((p["grid"] for p in abl["prep"].values() if "grid" in p), None)
    if grid:
        out.append(f"| OCT fine grid {'x'.join(map(str, grid['fine_shape']))} (streamed once) | {num(grid['seconds'], 0)} | "
                   f"{num(grid['peak_rss_gb'], 1)} |")
    for k, p in abl["prep"].items():
        out.append(f"| prep {k} | {num(p.get('mask_seconds', p.get('seconds')), 0)} | {num(p.get('peak_rss_gb'), 1)} |")
    gpu = [r["gpu_peak_gb"] for r in abl["variants"].values() if r.get("gpu_peak_gb") is not None]
    out += [f"| all variants and preprocessing | {num(abl.get('seconds'), 0)} | {num(abl.get('peak_rss_gb'), 1)} |", "",
            f"Peak GPU memory allocated by torch over the variants: {num(max(gpu) if gpu else None, 1)} GiB.", ""]
    return out


def figures(run, out):
    """Three figures of the I58 report from the run's own qc images, not tracked by git:
    out/fig_qc_I58.png is RUN/qc.png without the white margin, outline colours kept, and out/fig_qc_montage_I58.png and
    out/fig_qc_deform_I58.png are RUN/qc_montage.png and RUN/qc_deform.png as they are."""
    import shutil
    from PIL import Image, ImageOps
    out.mkdir(parents=True, exist_ok=True)
    qc = Image.open(run / "qc.png").convert("RGB")
    x0, y0, x1, y1 = ImageOps.invert(qc.convert("L")).getbbox()
    qc.crop((max(x0 - 8, 0), max(y0 - 8, 0), min(x1 + 8, qc.width), min(y1 + 8, qc.height))).save(out / "fig_qc_I58.png", optimize=True)
    for src, dst in (("qc_montage.png", "fig_qc_montage_I58.png"), ("qc_deform.png", "fig_qc_deform_I58.png")):
        if (run / src).exists():
            shutil.copyfile(run / src, out / dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", type=Path, required=True)
    ap.add_argument("--ablate", type=Path, default=None)
    ap.add_argument("--logs", type=Path, default=None)
    ap.add_argument("-o", "--out", type=Path, default=Path("bench/BENCHMARK.md"))
    ap.add_argument("--figures", type=Path, default=None, help="also write three qc figures of the run into this dir")
    ap.add_argument("--store", type=Path, default=None,
                    help="also copy result.json, eval.json, ablations.json and starts.json into this dir (bench/results/I58), "
                         "absolute paths cut to a file name or a bench_runs/ path")
    a = ap.parse_args()
    abl = load(a.ablate / "ablations.json") if a.ablate else {}
    st = load(a.ablate / "starts.json") if a.ablate and (a.ablate / "starts.json").exists() else {}
    phash = load(a.main / "result.json").get("params_hash") or abl.get("params_hash")
    stored = ["result.json", "eval.json", "ablations.json"] + (["starts.json"] if st else [])
    intro = ["# Benchmark: the I58 brainstem pair", "",
             "octreg registered the two original files as given (OCT 1457×2013×1595 at 20 µm, header LPI, and MRI crop 343×489×495 at "
             "0.08 mm, header RIA) with `octreg register OCT MRI -o OUT` and default parameters"
             + (f" (Params hash {phash})" if phash else "") + ". "
             f"The numbers come from {', '.join(stored[:-1])} and {stored[-1]}, stored in bench/results/I58/"
             + (", except the wall clock and the GPU memory of the device, which come from the step logs of bench/run_i58.py"
                if a.logs and (a.logs / "register.time").exists() else "") + ".", "",
             "Commands: `python bench/run_i58.py` (see bench/README.md).", ""]
    old = a.out.read_text(encoding="utf-8") if a.out.exists() else ""
    visual = re.search(r"^## Visual result\n.*?(?=^## Main result)", old, re.S | re.M)       # hand-written, kept
    text = "\n".join(intro + ([visual.group(0).rstrip("\n"), ""] if visual else []) + main_section(a.main, a.logs)
                     + ablation_section(abl) + starts_section(st) + runtime_section(abl))
    if READING in old:                                   # the hand-written reading at the end survives a rewrite
        text += "\n" + READING + old.split(READING, 1)[1]
    text = "\n".join(wrap(text.split("\n")))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(text.encode("utf-8"))             # UTF-8 and LF on every OS
    print(f"wrote {a.out}")
    if a.figures:
        figures(a.main, a.figures)
        print(f"wrote the figures of {a.out.name} into {a.figures}")
    if a.store:
        a.store.mkdir(parents=True, exist_ok=True)
        for src in [a.main / "result.json", a.main / "eval.json"] + ([a.ablate / "ablations.json", a.ablate / "starts.json"]
                                                                       if a.ablate else []):
            if src.exists():                             # the format of octreg.io.write_json, LF on every OS
                (a.store / src.name).write_bytes(json.dumps(portable(load(src)), indent=1, allow_nan=False).encode("utf-8"))
                print(f"stored {a.store / src.name}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Side-by-side section panels of the baseline comparison (bench only, not part of the package).

    python bench/baselines/composites.py --shots DIR --out DIR/../composites --methods mri octreg ants_com ...

Reads one 800x800 freeview screenshot DIR/<section>_<method>.png per method at each shared section
(baselines/shots/sections.txt). Writes one image per section with the methods side by side, up to --per-row per row. Each
panel carries the method name, or the label of a --label name=Label pair. A missing screenshot gives a dark panel marked
(missing).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SECTIONS = ["sag_m8", "sag_m3", "sag_p2", "sag_p7", "cor_m10", "cor_m3", "cor_p4", "cor_p11", "ax_10", "ax_18", "ax_26",
            "ax_34", "ax_40", "top_sag", "top_cor", "top_sag0", "top_cor35"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--label", nargs="*", default=[])
    ap.add_argument("--per-row", type=int, default=4)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--sections", nargs="*", default=SECTIONS)
    a = ap.parse_args()
    labels = dict(kv.split("=", 1) for kv in a.label)
    a.out.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("arial.ttf", int(28 * a.scale * 2))
    except OSError:
        font = ImageFont.load_default()
    size = int(800 * a.scale)
    for sec in a.sections:
        panels = []
        for m in a.methods:
            f = a.shots / f"{sec}_{m}.png"
            im = Image.open(f).convert("RGB").resize((size, size), Image.LANCZOS) if f.is_file() else Image.new("RGB", (size, size), (30, 30, 30))
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, size, int(34 * a.scale * 2)], fill=(0, 0, 0))
            d.text((6, 2), labels.get(m, m) + ("" if f.is_file() else " (missing)"), fill=(255, 255, 255), font=font)
            panels.append(im)
        rows = [panels[i:i + a.per_row] for i in range(0, len(panels), a.per_row)]
        W = size * min(a.per_row, len(panels)) + 4 * (min(a.per_row, len(panels)) - 1)
        H = size * len(rows) + 4 * (len(rows) - 1)
        out = Image.new("RGB", (W, H), (60, 60, 60))
        for r, row in enumerate(rows):
            for c, im in enumerate(row):
                out.paste(im, (c * (size + 4), r * (size + 4)))
        out.save(a.out / f"{sec}.png")
    print(f"{len(a.sections)} composites in {a.out}")


if __name__ == "__main__":
    main()

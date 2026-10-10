#!/usr/bin/env python3
"""Zoomed crops of the superior-end sections for the artefact check (bench only).

    python bench/baselines/zooms.py --shots DIR --out DIR --methods a b c [--label a=Label ...] [--per-row 4]

The top_* sections of baselines/shots/sections.txt are rendered at 2.4x by freeview; this crops the band of the block end
where the section stripes and the cut edge of the block are visible (a fixed window per section) and enlarges it 2x, one
panel per method, so that bent stripes, bowed block edges and smeared tissue show at a glance.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WINDOWS = {"top_cor35": (0, 240, 480, 720), "top_cor": (150, 250, 630, 730), "top_sag": (200, 250, 680, 730),
           "top_sag0": (200, 250, 680, 730)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--label", nargs="*", default=[])
    ap.add_argument("--per-row", type=int, default=4)
    a = ap.parse_args()
    labels = dict(kv.split("=", 1) for kv in a.label)
    a.out.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
    for sec, box in WINDOWS.items():
        w, h = (box[2] - box[0]) * 2, (box[3] - box[1]) * 2
        panels = []
        for m in a.methods:
            f = a.shots / f"{sec}_{m}.png"
            im = (Image.open(f).convert("RGB").crop(box).resize((w, h), Image.NEAREST) if f.is_file()
                  else Image.new("RGB", (w, h), (30, 30, 30)))
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, w, 40], fill=(0, 0, 0))
            d.text((6, 4), labels.get(m, m) + ("" if f.is_file() else " (missing)"), fill=(255, 255, 255), font=font)
            panels.append(im)
        rows = [panels[i:i + a.per_row] for i in range(0, len(panels), a.per_row)]
        n = min(a.per_row, len(panels))
        out = Image.new("RGB", (w * n + 6 * (n - 1), h * len(rows) + 6 * (len(rows) - 1)), (60, 60, 60))
        for r, row in enumerate(rows):
            for c, im in enumerate(row):
                out.paste(im, (c * (w + 6), r * (h + 6)))
        out.save(a.out / f"zoom_{sec}.png")
    print(f"{len(WINDOWS)} zoom composites in {a.out}")


if __name__ == "__main__":
    main()

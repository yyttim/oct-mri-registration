"""python -m tilestack BLOCK_JSON OUT_DIR [--full] [--layout LATTICE_JSON] [--zero LOG10]"""
import argparse
import os

ap = argparse.ArgumentParser(prog="python -m tilestack", description="Internal registration of the slabs of a serial-sectioning OCT block.")
ap.add_argument("block", help="JSON with the block's slabs and settings (see tilestack/data.py)")
ap.add_argument("out", help="output folder")
ap.add_argument("--full", action="store_true", help="also write the registered volume at the raw voxel")
ap.add_argument("--layout", help="JSON with the tile lattice from the acquisition metadata: X0, Y0, PX, PY, SX, SY, W, H in pixels, rows, cols")
ap.add_argument("--zero", type=float, help="the embedding gel's level (log10 mu_s) in place of the lowest mode of the level map")
a = ap.parse_args()
os.environ["TILESTACK_BLOCK"] = os.path.abspath(a.block)
from .run import run  # noqa: E402  (the package reads the block when it is imported)

run(a.out, full=a.full, layout=a.layout, zero=a.zero)

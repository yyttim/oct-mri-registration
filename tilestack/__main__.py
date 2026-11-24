"""python -m tilestack BLOCK_JSON OUT_DIR [--full] [--layout LATTICE_JSON] [--mri MRI]"""
import argparse
import os

ap = argparse.ArgumentParser(prog="python -m tilestack", description="Internal registration of the slabs of a serial-sectioning OCT block.")
ap.add_argument("block", metavar="BLOCK_JSON", help="JSON with the block's slabs and settings (docs/TILESTACK.md, Usage)")
ap.add_argument("out", metavar="OUT_DIR", help="output folder")
ap.add_argument("--full", action="store_true", help="also write the registered volume at the raw voxel")
ap.add_argument("--layout", metavar="LATTICE_JSON", help="JSON with the tile lattice from the acquisition metadata: X0, Y0, PX, PY, SX, SY, W along x, H along y, in pixels of the slab files, rows, cols")
ap.add_argument("--mri", help="MRI NIfTI cropped around the block: also register the volume to it with octreg (OUT_DIR/mri)")
a = ap.parse_args()
os.environ["TILESTACK_BLOCK"] = os.path.abspath(a.block)
from .run import run  # noqa: E402  (the package reads the block when it is imported)

run(a.out, full=a.full, layout=a.layout, mri=a.mri)

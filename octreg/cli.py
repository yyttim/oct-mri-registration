"""Command line: `octreg register` and `octreg apply`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _zyx(text):
    try:
        v = tuple(float(x) for x in text.split(","))
    except ValueError:
        v = ()
    if len(v) != 3:
        raise argparse.ArgumentTypeError(f"expected Z,Y,X in um, got {text!r}")
    return v


def main(argv=None) -> int:
    """Run a subcommand. -> exit code: 0 done, 2 input refused with the message on stderr."""
    ap = argparse.ArgumentParser(prog="octreg", description="Label-free affine registration of an OCT block to an MRI crop.")
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("register", help="register OCT to MRI and write the transform, overlays, visual QC and result.json")
    r.add_argument("oct", help="OCT volume: NIfTI, or TIFF / OME-TIFF / NPY with spacing")
    r.add_argument("mri", help="MRI NIfTI, already cropped to a region containing the block")
    r.add_argument("-o", "--out", required=True, help="output directory")
    r.add_argument("--oct-spacing-um", type=_zyx, metavar="Z,Y,X",
                   help="OCT voxel spacing (um) along the numpy axes of a TIFF / NPY (NIfTI axes k, j, i), also used for an "
                   "OCT mask of the same array shape")
    r.add_argument("--oct-mask", help="OCT specimen mask file (voxels > 0, in the OCT world) instead of octreg's own")
    r.add_argument("--mri-mask", help="MRI foreground mask file (voxels > 0, in the MRI world) instead of octreg's own")
    r.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="cuda (default) or cpu")
    r.add_argument("--params", type=Path, metavar="F", help="JSON file with Params overrides, e.g. the params block of a result.json")
    a = sub.add_parser("apply", help="resample a volume through a registration")
    a.add_argument("--run", required=True, help="output directory of octreg register")
    a.add_argument("--moving", required=True, help="volume in the OCT frame (MRI frame with --inverse)")
    a.add_argument("--reference", required=True, help="volume whose grid the output takes")
    a.add_argument("-o", "--out", required=True, help="output NIfTI")
    a.add_argument("--inverse", action="store_true", help="MRI -> OCT through the inverse affine")
    a.add_argument("--oct-spacing-um", type=_zyx, metavar="Z,Y,X",
                   help="as for register, for the file in the OCT frame (default: the run's, for its own OCT file "
                   "at its original path)")
    args = ap.parse_args(argv)
    from .params import Params
    from .register import apply, pose_label, register
    try:
        if args.command == "register":
            if args.params is not None and not args.params.is_file():
                raise ValueError(f"{args.params}: no such file")
            params = Params.from_dict(json.loads(args.params.read_text())) if args.params else Params()
            res = register(args.oct, args.mri, args.out, args.oct_spacing_um, args.oct_mask, args.mri_mask, params, args.device)
            print(f"octreg: {pose_label(res['pose'])} -> {args.out}")
        else:
            print(f"octreg: wrote {apply(args.run, args.moving, args.reference, args.out, args.inverse, args.oct_spacing_um)}")
    except ValueError as e:
        print(f"octreg: error: {e}", file=sys.stderr)
        return 2
    return 0

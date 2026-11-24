"""tilestack: internal registration of the slabs of a serial-sectioning OCT block (docs/TILESTACK.md).

Command line: python -m tilestack BLOCK_JSON OUT_DIR. The block is read when a module other than mri is imported, from the
JSON named by the environment variable TILESTACK_BLOCK (tilestack/data.py).
"""
BELOW = 1000    # source-map offset of the voxels below the reliable planes
dev = "cuda"

"""§2: the outline is used only when both images have a measured outside (register.outline_used)."""
import numpy as np

from octreg.params import Params
from octreg.register import outline_used


def _ball(n, r):
    z, y, x = np.mgrid[:n, :n, :n] - n // 2
    return x * x + y * y + z * z <= r * r


def test_outline_needs_an_outside_in_both_images():
    tissue, measured, mri_fg = _ball(40, 12), np.ones((40,) * 3, bool), _ball(48, 18)
    use, out = outline_used(tissue, measured, mri_fg, Params())
    assert use and out["oct"] > 0.5 and out["mri"] > 0.3                          # embedded block, MRI with medium
    assert not outline_used(tissue, tissue, mri_fg, Params())[0]                    # OCT given as tissue only
    assert not outline_used(tissue, measured, np.ones((48,) * 3, bool), Params())[0]  # MRI crop that is all tissue
    shell = tissue | (_ball(40, 13) & ~tissue)                                      # a one-voxel resampling shell is no outside
    assert not outline_used(tissue, shell, mri_fg, Params())[0]

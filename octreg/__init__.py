"""octreg: label-free registration of a serial-sectioning OCT block to an ex-vivo MRI cropped around it: an affine and a
small smooth deformation on top, both fitted to one polarity-free structure similarity F (docs/METHOD.md).

1. Masks: the OCT specimen as the voxels with signal and isotropic texture, the MRI foreground by a histogram valley.
2. F: the correlation of gradient-orientation tensors (octreg.features); the outline O compares the two masks.
3. FFT orientation search over a near-uniform rotation set inside the MRI crop: poses proposed by F and, where both images
   show medium around the tissue, by (F + O) / 2.
4. Every proposed pose refined on F under a scale and shear prior, coarse to fine; the lowest final L wins.
5. A smooth displacement field on top of the affine, fitted to F under a strain bound.

Python use: `from octreg.register import register, apply`; command line: `octreg register`, `octreg apply`.
"""
from .params import Params

__version__ = "3.1.0"

__all__ = ["Params", "__version__"]

"""octreg: label-free registration of a serial-sectioning OCT block to an ex-vivo MRI cropped around it: an affine and a
small smooth deformation on top, both fitted to one polarity-free structure score.

1. Specimen mask from isotropic texture: agarose has the intensity of tissue, but its artefacts are anisotropic.
2. Score: the correlation of gradient-orientation tensors (octreg.features), with the outline of the specimen mask against
   the MRI foreground where both images show an outside.
3. FFT orientation search over a near-uniform rotation set inside the MRI crop.
4. Every search pose refined on the structure score under a scale and shear prior, coarse to fine, and the lowest
   final L wins.
5. A smooth displacement field on top of the affine, fitted to the same score under a strain bound.

Method constants are in octreg.params.Params (numerical guards and fixed rule literals are documented where they are used).
Python use: `from octreg.register import register, apply`; command line: `octreg register`, `octreg apply`.
"""
from .params import Params

__version__ = "3.0.0"

__all__ = ["Params", "__version__"]

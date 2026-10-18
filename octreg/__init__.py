"""octreg: label-free registration of a serial-sectioning OCT block to an ex-vivo MRI cropped around it (docs/METHOD.md).

Python: `from octreg.register import register, apply`. Command line: `octreg register`, `octreg apply`.
"""
from .params import Params

__version__ = "3.1.0"

__all__ = ["Params", "__version__"]

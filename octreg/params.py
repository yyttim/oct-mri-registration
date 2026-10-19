"""The method constants of octreg in one frozen dataclass. Lengths in mm."""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    """Method constants. Variants via dataclasses.replace or from_dict."""
    # grids
    search_mm: float = 0.6                    # orientation search
    base_mm: float = 0.15                     # refinement
    fine_mm: float = 0.04                     # OCT specimen mask
    # masks (§1)
    valley_ratio: float = 0.5                 # MRI histogram valley / smaller neighbouring peak must be below this
    min_component: float = 0.01               # components below this fraction of the mask are dropped
    texture_bandpass_mm: float = 0.08         # Gaussian sigma of the smoothing before the texture
    texture_window_mm: float = 0.36           # window of the local coefficient of variation
    texture_grid_mm: float = 0.16             # cell size of the texture grid
    texture_smooth_mm: float = 1.2            # Gaussian sigma of the log texture before the threshold
    # similarity (§2)
    erode_mm: float = 0.8                     # depth of the OCT points of F inside the specimen mask
    outline_min_outside: float = 0.01         # measured share outside the tissue both images need for the outline
    # orientation search (§3)
    search_sigma_mm: float = 1.2              # Gaussian sigma of the search gradients
    n_rot: int = 8000                         # Super-Fibonacci rotations
    topk: int = 24                            # poses kept after non-maximum suppression
    nms_mm: float = 3.0                       # same pose if centres closer than nms_mm and rotations closer than nms_deg
    nms_deg: float = 20.0
    # affine refinement (§4)
    sigmas_mm: tuple = (0.6, 0.4, 0.3)        # one pass per sigma, coarse to fine
    iters: int = 150                          # Adam iterations per pass
    lam: float = 2.0                          # L = 1 - F + lam (sum log_scale^2 + sum shear^2)
    lr_mm: float = 0.06                       # Adam step in mm of OCT box corner motion

    def __post_init__(self):
        real = lambda x: isinstance(x, float) and math.isfinite(x)
        for f in dataclasses.fields(self):                    # JSON gives ints for floats and lists for tuples
            given = v = getattr(self, f.name)
            if isinstance(f.default, float) and type(v) is int:
                v = float(v)
            elif isinstance(f.default, tuple) and isinstance(v, (list, tuple)):
                v = tuple(float(x) if type(x) is int else x for x in v)
            object.__setattr__(self, f.name, v)
            if isinstance(f.default, float):
                ok, want = real(v), "a finite number"
            elif isinstance(f.default, tuple):
                ok, want = isinstance(v, tuple) and len(v) > 0 and all(map(real, v)), "a non-empty list of finite numbers"
            else:
                ok, want = isinstance(v, int) and not isinstance(v, bool), "an integer"
            if not ok:
                raise ValueError(f"Params: {f.name} must be {want}, got {given!r}")
        if not 0 < self.fine_mm <= self.base_mm < self.search_mm:
            raise ValueError("Params: need 0 < fine_mm <= base_mm < search_mm")
        if min(self.sigmas_mm) <= 0:
            raise ValueError(f"Params: sigmas_mm must be positive, got {self.sigmas_mm!r}")
        for names, ok, want in (
                # outline_min_outside above 1 turns the outline off
                (("texture_bandpass_mm", "texture_window_mm", "texture_grid_mm", "texture_smooth_mm", "erode_mm",
                  "search_sigma_mm", "lr_mm", "valley_ratio", "outline_min_outside"), lambda x: x > 0, "positive"),
                (("lam", "nms_mm", "nms_deg"), lambda x: x >= 0, "at least 0"),
                (("iters", "n_rot", "topk"), lambda x: x >= 1, "at least 1"),
                (("min_component",), lambda x: 0 <= x < 1, "in [0, 1)")):
            for name in names:
                if not ok(getattr(self, name)):
                    raise ValueError(f"Params: {name} must be {want}, got {getattr(self, name)!r}")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        """Params from a partial dict. Unknown keys raise ValueError."""
        if not isinstance(d, dict):
            raise ValueError(f"Params: overrides must be a JSON object, not {type(d).__name__}")
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Params: unknown keys {sorted(unknown)}")
        return cls(**d)

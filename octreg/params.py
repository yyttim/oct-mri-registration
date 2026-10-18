"""The method constants of octreg in one frozen dataclass. Lengths in mm."""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    """Method constants (no switches). A variant via dataclasses.replace or from_dict (a partial dict)."""
    # grids
    search_mm: float = 0.6                    # isotropic grid of the orientation search
    base_mm: float = 0.15                     # isotropic grid of the refinement and the deformation
    fine_mm: float = 0.04                     # OCT grid of the specimen mask
    # masks (§1)
    valley_ratio: float = 0.5                 # MRI histogram valley / smaller neighbouring peak must be below this
    min_component: float = 0.01               # mask components smaller than this fraction of the mask are dropped
    texture_bandpass_mm: float = 0.08         # sigma of the normalised Gaussian smoothing before the directional variation
    texture_window_mm: float = 0.36           # window of the directional local coefficient of variation
    texture_grid_mm: float = 0.16             # block grid of the texture field
    texture_smooth_mm: float = 1.2            # Gaussian sigma of the log texture field before its threshold
    # score (§2)
    erode_mm: float = 0.8                     # the OCT points of F lie farther than this inside the specimen mask
    outline_min_outside: float = 0.01         # the outline is used when both images have this measured share outside the tissue
    # orientation search (§3)
    search_sigma_mm: float = 1.2              # Gaussian sigma of the gradients in the search
    n_rot: int = 8000                         # Super-Fibonacci rotations
    topk: int = 24                            # poses kept after non-maximum suppression, each one refined (§4)
    nms_mm: float = 3.0                       # same pose if centres closer than nms_mm ...
    nms_deg: float = 20.0                     # ... and rotations closer than nms_deg
    # affine refinement (§4)
    sigmas_mm: tuple = (0.6, 0.4, 0.3)        # Gaussian sigmas of the passes, coarse to fine
    iters: int = 150                          # Adam iterations per pass
    lam: float = 2.0                          # L = 1 - F + lam (sum log_scale^2 + sum shear^2), the scale and shear prior
    lr_mm: float = 0.06                       # Adam step, in mm of motion of the OCT box corners
    # smooth deformation (§5)
    df_grid_mm: float = 5.0                   # spacing of the control lattice
    df_max_strain: float = 0.08               # largest first difference of the lattice / spacing
    df_sigmas_mm: tuple = (0.6, 0.3)          # Gaussian sigmas of the deformation passes
    df_evals: int = 250                       # score evaluations after which an L-BFGS pass stops

    def __post_init__(self):
        real = lambda x: isinstance(x, float) and math.isfinite(x)
        for f in dataclasses.fields(self):                    # ints given for floats, lists for tuples (JSON)
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
        if not (self.sigmas_mm and self.df_sigmas_mm and min(self.sigmas_mm + self.df_sigmas_mm) > 0):
            raise ValueError("Params: sigmas_mm and df_sigmas_mm must be non-empty and positive")
        for names, ok, want in (
                (("texture_bandpass_mm", "texture_window_mm", "texture_grid_mm", "texture_smooth_mm", "erode_mm",
                  "search_sigma_mm", "lr_mm", "df_grid_mm", "df_max_strain", "valley_ratio"), lambda x: x > 0, "positive"),
                # outline_min_outside above 1 is a share never met: the outline is then never used
                (("lam", "nms_mm", "nms_deg", "outline_min_outside"), lambda x: x >= 0, "at least 0"),
                (("iters", "df_evals", "n_rot", "topk"), lambda x: x >= 1, "at least 1"),
                (("min_component",), lambda x: 0 <= x < 1, "in [0, 1)")):
            for name in names:
                if not ok(getattr(self, name)):
                    raise ValueError(f"Params: {name} must be {want}, got {getattr(self, name)!r}")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        """Params from a partial dict (JSON). Unknown keys raise ValueError."""
        if not isinstance(d, dict):
            raise ValueError(f"Params: overrides must be a JSON object, not {type(d).__name__}")
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Params: unknown keys {sorted(unknown)}")
        return cls(**d)

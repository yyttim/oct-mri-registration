"""The method constants of octreg in one frozen dataclass. Lengths in mm."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    """Method constants (no switches); a variant via dataclasses.replace or from_dict (a partial dict)."""
    # grids
    search_mm: float = 0.6                    # isotropic grid of the orientation search
    base_mm: float = 0.15                     # isotropic grid of the refinement and the deformation
    fine_mm: float = 0.04                     # OCT grid of the specimen mask
    # masks (§1)
    valley_ratio: float = 0.5                 # MRI histogram valley / smaller neighbouring peak must be below this
    min_component: float = 0.01               # mask components smaller than this fraction of the mask are dropped
    texture_bandpass_mm: float = 0.08         # Gaussian sigma of the band-pass before the directional variation
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
    df_band_mm: float = 0.6                   # the OCT points of §5 lie within this of the specimen mask
    df_sigmas_mm: tuple = (0.6, 0.3)          # Gaussian sigmas of the deformation passes
    df_evals: int = 250                       # most score evaluations of L-BFGS per pass

    def __post_init__(self):
        for f in dataclasses.fields(self):                    # ints given for floats, lists for tuples (JSON)
            v = getattr(self, f.name)
            if isinstance(f.default, float) and type(v) is int:
                object.__setattr__(self, f.name, float(v))
            elif isinstance(f.default, tuple) and isinstance(v, (list, tuple)):
                object.__setattr__(self, f.name, tuple(float(x) for x in v))
        if not 0 < self.fine_mm <= self.base_mm < self.search_mm:
            raise ValueError("Params: need 0 < fine_mm <= base_mm < search_mm")
        if not (self.sigmas_mm and self.df_sigmas_mm and min(self.sigmas_mm + self.df_sigmas_mm) > 0):
            raise ValueError("Params: sigmas_mm and df_sigmas_mm must be non-empty and positive")
        if not min(self.erode_mm, self.search_sigma_mm, self.df_grid_mm, self.df_max_strain, self.df_band_mm) > 0 or min(
                self.iters, self.df_evals, self.n_rot, self.topk) < 1:
            raise ValueError("Params: lengths and the strain bound must be positive, iterations, n_rot and topk >= 1")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        """Params from a partial dict (JSON); unknown keys raise ValueError."""
        if not isinstance(d, dict):
            raise ValueError(f"Params: overrides must be a JSON object, not {type(d).__name__}")
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Params: unknown keys {sorted(unknown)}")
        return cls(**d)

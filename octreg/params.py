"""The method constants of octreg in one frozen dataclass. Lengths in mm."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    """Method constants (no switches); a variant via dataclasses.replace or from_dict (a partial dict)."""
    # grids
    search_mm: float = 0.6                    # isotropic grid of the orientation search, an integer multiple of base_mm
    base_mm: float = 0.15                     # isotropic grid of the refinement and the deformation
    fine_mm: float = 0.04                     # OCT grid of the specimen mask
    # masks (§1)
    valley_ratio: float = 0.5                 # MRI histogram valley / smaller neighbouring peak must be below this
    medium_fraction: float = 0.5              # ... and its darker peak below this x the brightest peak (medium, not tissue)
    min_component: float = 0.01               # keep MRI foreground components >= this fraction of the foreground volume
    texture_bandpass_mm: float = 0.08         # Gaussian sigma of the band-pass before the directional variation
    texture_window_mm: float = 0.36           # window of the directional local coefficient of variation
    texture_grid_mm: float = 0.16             # block grid of the texture field
    texture_smooth_mm: float = 1.2            # Gaussian sigma of the log texture field before its two-class (Otsu) threshold
    texture_close_mm: float = 0.48            # closing radius of the specimen mask
    # score (§2)
    erode_mm: float = 0.8                     # erosion of the OCT points of F, in round(mm / h) steps of the 6-neighbourhood
    outline_min_outside: float = 0.01         # the outline is used when both images have this measured share outside the tissue
    # orientation search (§3)
    search_sigma_mm: float = 1.2              # Gaussian sigma of the gradients in the search
    n_rot: int = 8000                         # rotations (identity + Super-Fibonacci)
    topk: int = 24                            # poses kept after non-maximum suppression, each one refined (§4)
    nms_mm: float = 3.0                       # same pose if centres closer than nms_mm ...
    nms_deg: float = 20.0                     # ... and rotations closer than nms_deg
    # affine refinement (§4)
    sigmas_mm: tuple = (0.6, 0.4, 0.3)        # Gaussian sigmas of the passes, coarse to fine
    iters: int = 150                          # Adam iterations per pass
    lam: float = 2.0                          # L = 1 - F + lam (sum log_scale^2 + sum shear^2), the scale and shear prior
    lr_rot: float = 0.004                     # rad
    lr_t: float = 0.06                        # mm
    lr_ls: float = 0.002                      # log-scale
    lr_sh: float = 0.002                      # shear
    # smooth deformation (§5)
    df_grid_mm: float = 5.0                   # spacing of the control lattice
    df_max_strain: float = 0.08               # largest first difference of the lattice / spacing
    df_band_mm: float = 0.6                   # dilation of the OCT points of §5, in round(mm / h) 6-neighbourhood steps
    df_sigmas_mm: tuple = (0.6, 0.3)          # Gaussian sigmas of the deformation passes
    df_iters: int = 200                       # most L-BFGS iterations per pass

    def __post_init__(self):
        for f in dataclasses.fields(self):                    # ints given for float fields become floats (equality = hash)
            v = getattr(self, f.name)
            if isinstance(f.default, float) and type(v) is int:
                object.__setattr__(self, f.name, float(v))
            elif isinstance(f.default, tuple) and isinstance(v, tuple):
                object.__setattr__(self, f.name, tuple(float(x) if type(x) is int else x for x in v))
        ok = 0 < self.fine_mm <= self.base_mm < self.search_mm
        if not (ok and abs(self.search_mm / self.base_mm - round(self.search_mm / self.base_mm)) < 1e-6):
            raise ValueError("Params: need 0 < fine_mm <= base_mm < search_mm, search_mm an integer multiple of base_mm")
        for name in ("sigmas_mm", "df_sigmas_mm"):
            v = getattr(self, name)
            if not (isinstance(v, tuple) and v and all(isinstance(x, float) and 0 < x < float("inf") for x in v)):
                raise ValueError(f"Params: {name} must be a non-empty tuple of positive finite floats")
        if not min(self.erode_mm, self.search_sigma_mm, self.df_grid_mm, self.df_max_strain, self.df_band_mm) > 0 or min(
                self.iters, self.df_iters, self.n_rot, self.topk) < 1:
            raise ValueError("Params: lengths and the strain bound must be positive, iterations, n_rot and topk >= 1")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        """Params from a partial dict (JSON); ints are accepted for floats; unknown keys or wrong types raise ValueError."""
        if not isinstance(d, dict):
            raise ValueError(f"Params: overrides must be a JSON object, not {type(d).__name__}")
        base, kw = cls(), {}
        for k, v in d.items():
            dv = getattr(base, k, None)
            kw[k] = v = float(v) if isinstance(dv, float) and type(v) is int else v
            if isinstance(dv, tuple) and isinstance(v, (list, tuple)):
                kw[k] = v = tuple(float(x) if type(x) is int else x for x in v)
            if k not in base.__dataclass_fields__ or type(v) is not type(dv):
                raise ValueError(f"Params {k}={v!r}: unknown key or not a {type(dv).__name__}")
        return cls(**kw)

    def hash(self) -> str:
        """16 hex characters of the sha256 of the sorted-key JSON of all fields (process and platform independent)."""
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]

"""NFW Dark Matter Halo computational engine for AstroGrid."""

import numpy as np
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class HaloProfileType(str, Enum):
    NFW = "nfw"
    ISOTHERMAL = "isothermal"
    PLUMMER = "plummer"


class NFWHaloParameters(BaseModel):
    """Parameters for an NFW dark matter halo."""
    mass: float = Field(..., gt=0, description="Total halo mass in solar masses")
    concentration: float = Field(..., gt=0, description="Concentration parameter c")
    r_vir: float = Field(..., gt=0, description="Virial radius in kpc")
    profile_type: HaloProfileType = Field(default=HaloProfileType.NFW)


class NFWHaloResult(BaseModel):
    """Result from NFW halo calculations."""
    density_profile: List[float] = Field(default_factory=list)
    enclosed_mass: List[float] = Field(default_factory=list)
    circular_velocity: List[float] = Field(default_factory=list)
    radii: List[float] = Field(default_factory=list)
    parameters: NFWHaloParameters


class NFWDarkMatterHaloEngine:
    """Computational engine for NFW dark matter halo profiles."""

    def __init__(self, parameters: NFWHaloParameters):
        self.params = parameters
        self.r_s = parameters.r_vir / parameters.concentration
        self.rho_0 = self._compute_rho_0()

    def _compute_rho_0(self) -> float:
        """Compute the characteristic density rho_0 from mass and concentration."""
        c = self.params.concentration
        mass = self.params.mass
        r_vir = self.params.r_vir

        # f(c) = ln(1+c) - c/(1+c)
        f_c = np.log(1 + c) - c / (1 + c)
        # rho_0 = M / (4*pi*r_vir^3 * f(c))
        rho_0 = mass / (4.0 * np.pi * r_vir**3 * f_c)
        return float(rho_0)

    def density(self, r: np.ndarray) -> np.ndarray:
        """Compute NFW density profile rho(r) = rho_0 / (r/r_s * (1 + r/r_s)^2)."""
        r = np.asarray(r, dtype=float)
        r_safe = np.where(r > 0, r, 1e-10)
        x = r_safe / self.r_s
        rho = self.rho_0 / (x * (1.0 + x)**2)
        return rho

    def enclosed_mass(self, r: np.ndarray) -> np.ndarray:
        """Compute enclosed mass M(r) = 4*pi*rho_0*r_s^3 * (ln(1+x) - x/(1+x))."""
        r = np.asarray(r, dtype=float)
        r_safe = np.where(r > 0, r, 1e-10)
        x = r_safe / self.r_s
        f = np.log(1.0 + x) - x / (1.0 + x)
        m = 4.0 * np.pi * self.rho_0 * self.r_s**3 * f
        return m

    def circular_velocity(self, r: np.ndarray) -> np.ndarray:
        """Compute circular velocity v_c(r) = sqrt(G*M(r)/r)."""
        r = np.asarray(r, dtype=float)
        r_safe = np.where(r > 0, r, 1e-10)
        m = self.enclosed_mass(r_safe)
        g = 4.30091e-6  # G in units of (kpc * (km/s)^2 / Msun)
        v_c = np.sqrt(g * m / r_safe)
        return v_c

    def compute_profile(self, n_points: int = 100, r_max: Optional[float] = None) -> NFWHaloResult:
        """Compute full halo profile over a radial grid."""
        if r_max is None:
            r_max = self.params.r_vir
        radii = np.linspace(1e-3, r_max, n_points)
        rho = self.density(radii)
        m_enc = self.enclosed_mass(radii)
        v_c = self.circular_velocity(radii)

        return NFWHaloResult(
            density_profile=[float(x) for x in rho],
            enclosed_mass=[float(x) for x in m_enc],
            circular_velocity=[float(x) for x in v_c],
            radii=[float(x) for x in radii],
            parameters=self.params,
        )

    def virial_radius_from_mass(self, mass: float, concentration: float) -> float:
        """Compute virial radius from mass and concentration."""
        c = concentration
        f_c = np.log(1 + c) - c / (1 + c)
        r_vir = (mass / (4.0 * np.pi * self.rho_0 * f_c)) ** (1.0 / 3.0)
        return float(r_vir)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize engine state to dictionary."""
        return {
            "parameters": self.params.model_dump(),
            "r_s": self.r_s,
            "rho_0": self.rho_0,
        }
"""
Supernova Arnett Model Engine.

Implements the Arnett (1982) model for Type II supernova light curves,
using the grey diffusion approximation for radiative transfer.
"""

import numpy as np
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union
from enum import Enum


class OpacityModel(str, Enum):
    """
    Defines the opacity treatment in the Arnett model.
    """
    GREY = "grey"


class ArnettModelParams(BaseModel):
    """
    Parameters for the Arnett supernova model.
    """
    ejecta_mass: float = Field(..., gt=0, description="Ejecta mass in solar masses")
    kinetic_energy: float = Field(..., gt=0, description="Kinetic energy in ergs")
    nickel_mass: float = Field(..., ge=0, description="Mass of Ni-56 in solar masses")
    opacity: float = Field(..., gt=0, description="Grey opacity in cm^2/g")
    diffusion_timescale: Optional[float] = Field(None, gt=0, description="Diffusion timescale in days")
    peak_luminosity: Optional[float] = Field(None, gt=0, description="Peak luminosity in erg/s")


class ArnettModelResult(BaseModel):
    """
    Results from the Arnett model computation.
    """
    time_days: List[float] = Field(..., description="Time array in days")
    luminosity: List[float] = Field(..., description="Luminosity in erg/s")
    peak_time: float = Field(..., description="Time of peak luminosity in days")
    peak_luminosity: float = Field(..., description="Peak luminosity in erg/s")
    diffusion_timescale: float = Field(..., description="Diffusion timescale in days")


class SupernovaArnettModel:
    """
    Computational engine for the Arnett (1982) supernova model.
    
    Uses the grey diffusion approximation to compute the bolometric
    light curve of a Type II supernova.
    """

    def __init__(self, params: ArnettModelParams):
        self.params = params
        self._compute()

    def _compute(self):
        """Compute the Arnett model light curve."""
        p = self.params

        # Physical constants
        c = 2.998e10  # cm/s
        G = 6.674e-8  # cm^3 g^-1 s^-2
        M_sun = 1.989e33  # g
        day = 86400.0  # s

        # Convert to cgs
        M_ej = p.ejecta_mass * M_sun
        E_kin = p.kinetic_energy
        M_Ni = p.nickel_mass * M_sun
        kappa = p.opacity

        # Diffusion timescale: t_d = sqrt(2 * kappa * M_ej / (beta * c * v))
        # where v = sqrt(2 * E_kin / M_ej)
        v = np.sqrt(2.0 * E_kin / M_ej)
        beta = 13.8  # Arnett's constant for diffusion
        t_d = np.sqrt(2.0 * kappa * M_ej / (beta * c * v))
        t_d_days = t_d / day

        # Peak luminosity from Arnett's relation:
        # L_peak = E_Ni / t_d, where E_Ni is the total Ni-56 decay energy
        # Ni-56 -> Co-56 -> Fe-56, total energy ~ 1.72e43 erg per solar mass of Ni
        E_Ni = M_Ni * 1.72e43 / M_sun
        L_peak = E_Ni / t_d

        # Peak time is approximately t_d
        t_peak = t_d_days

        # Generate light curve using Arnett's analytic approximation
        # L(t) = L_peak * (t/t_d) * exp(-t/t_d) for t > 0
        # More precisely, use the diffusion solution
        n_points = 200
        t_max = 5.0 * t_d_days
        time_days = np.linspace(0.01, t_max, n_points)

        # Arnett's light curve: L(t) = L_peak * (t/t_d) * exp(-t/t_d)
        # This is a simplified form; the full solution involves the
        # diffusion equation with a delta-function source
        t_ratio = time_days / t_d_days
        luminosity = L_peak * t_ratio * np.exp(-t_ratio)

        # Ensure no negative values
        luminosity = np.maximum(luminosity, 0.0)

        self.time_days = time_days.tolist()
        self.luminosity = luminosity.tolist()
        self.peak_time = float(t_peak)
        self.peak_luminosity = float(L_peak)
        self.diffusion_timescale = float(t_d_days)

    def get_result(self) -> ArnettModelResult:
        """Return the computed model result."""
        return ArnettModelResult(
            time_days=self.time_days,
            luminosity=self.luminosity,
            peak_time=self.peak_time,
            peak_luminosity=self.peak_luminosity,
            diffusion_timescale=self.diffusion_timescale,
        )

    def get_luminosity_at_time(self, t_days: float) -> float:
        """
        Interpolate luminosity at a given time in days.
        
        Args:
            t_days: Time in days.
            
        Returns:
            Luminosity in erg/s.
        """
        if t_days <= 0:
            return 0.0
        t_ratio = t_days / self.diffusion_timescale
        L = self.peak_luminosity * t_ratio * np.exp(-t_ratio)
        return float(max(L, 0.0))
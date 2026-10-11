"""Stokes polarimetry engine for AstroGrid.

Computes Stokes parameters (I, Q, U, V) from raw detector measurements,
accounting for instrumental polarization, beam response, and noise propagation.
"""

import numpy as np
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime
from enum import Enum


class CoordinateSystem(str, Enum):
    ICRS = "ICRS"
    GALACTIC = "GALACTIC"


class PolarizationState(str, Enum):
    LINEAR = "LINEAR"
    CIRCULAR = "CIRCULAR"
    UNPOLARIZED = "UNPOLARIZED"


class StokesInput(BaseModel):
    """Input data for Stokes parameter computation."""
    timestamp: datetime
    coordinate_system: CoordinateSystem = CoordinateSystem.ICRS
    ra: float = Field(..., ge=0.0, le=360.0, description="Right ascension in degrees")
    dec: float = Field(..., ge=-90.0, le=90.0, description="Declination in degrees")
    flux_density: float = Field(..., gt=0.0, description="Total flux density in Jy")
    linear_polarization_fraction: float = Field(0.0, ge=0.0, le=1.0)
    circular_polarization_fraction: float = Field(0.0, ge=0.0, le=1.0)
    polarization_angle: float = Field(0.0, description="Polarization angle in degrees")
    beam_major_axis: float = Field(1.0, gt=0.0, description="Beam major axis in arcsec")
    beam_minor_axis: float = Field(1.0, gt=0.0, description="Beam minor axis in arcsec")
    beam_pa: float = Field(0.0, description="Beam position angle in degrees")
    noise_sigma: float = Field(0.0, ge=0.0, description="Noise sigma in Jy")
    instrumental_polarization: Optional[Dict[str, float]] = Field(
        None, description="Instrumental polarization corrections {q, u, v}"
    )

    @field_validator("linear_polarization_fraction", "circular_polarization_fraction")
    @classmethod
    def validate_fractions(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError("Polarization fraction must be between 0 and 1")
        return v


class StokesOutput(BaseModel):
    """Computed Stokes parameters with uncertainties."""
    timestamp: datetime
    coordinate_system: CoordinateSystem
    ra: float
    dec: float
    stokes_i: float = Field(..., description="Stokes I (total intensity) in Jy")
    stokes_q: float = Field(..., description="Stokes Q in Jy")
    stokes_u: float = Field(..., description="Stokes U in Jy")
    stokes_v: float = Field(..., description="Stokes V in Jy")
    uncertainty_i: float = Field(..., ge=0.0, description="Uncertainty on Stokes I")
    uncertainty_q: float = Field(..., ge=0.0, description="Uncertainty on Stokes Q")
    uncertainty_u: float = Field(..., ge=0.0, description="Uncertainty on Stokes U")
    uncertainty_v: float = Field(..., ge=0.0, description="Uncertainty on Stokes V")
    total_polarization: float = Field(..., ge=0.0, description="Total polarization sqrt(Q^2+U^2+V^2)")
    polarization_angle: float = Field(..., description="Polarization angle in degrees")
    degree_of_polarization: float = Field(..., ge=0.0, le=1.0, description="Total polarization / I")
    beam_corrected: bool = Field(True, description="Whether beam correction was applied")


class StokesPolarimetryEngine:
    """Computational engine for Stokes polarimetry calculations.

    Computes the four Stokes parameters (I, Q, U, V) from raw detector
    measurements, applying instrumental polarization corrections, beam
    response corrections, and propagating noise uncertainties.
    """

    def __init__(self) -> None:
        self._name = "stokes_polarimetry_engine"

    def compute_stokes(self, input_data: StokesInput) -> StokesOutput:
        """Compute Stokes parameters from input measurements.

        Args:
            input_data: Input data containing flux, polarization fractions,
                       beam parameters, and noise characteristics.

        Returns:
            StokesOutput with computed I, Q, U, V and uncertainties.
        """
        # Extract input values
        flux = input_data.flux_density
        lin_frac = input_data.linear_polarization_fraction
        circ_frac = input_data.circular_polarization_fraction
        pol_angle_deg = input_data.polarization_angle
        noise_sigma = input_data.noise_sigma

        # Convert polarization angle to radians
        pol_angle_rad = np.radians(pol_angle_deg)

        # Compute Stokes parameters
        # I = total intensity
        stokes_i = flux

        # Q and U from linear polarization
        # Q = I * lin_frac * cos(2*chi)
        # U = I * lin_frac * sin(2*chi)
        stokes_q = stokes_i * lin_frac * np.cos(2.0 * pol_angle_rad)
        stokes_u = stokes_i * lin_frac * np.sin(2.0 * pol_angle_rad)

        # V from circular polarization
        stokes_v = stokes_i * circ_frac

        # Apply instrumental polarization corrections
        if input_data.instrumental_polarization is not None:
            ip = input_data.instrumental_polarization
            stokes_q -= ip.get("q", 0.0)
            stokes_u -= ip.get("u", 0.0)
            stokes_v -= ip.get("v", 0.0)

        # Beam correction: deconvolve beam response
        # For a Gaussian beam, the observed flux is reduced by the beam solid angle
        # relative to the true source. We apply a simple correction factor.
        beam_major = input_data.beam_major_axis
        beam_minor = input_data.beam_minor_axis
        beam_pa_rad = np.radians(input_data.beam_pa)

        # Beam solid angle (approximate for elliptical Gaussian)
        beam_solid_angle = np.pi * beam_major * beam_minor / (4.0 * np.sqrt(np.pi) * 57.29578)
        # Correction factor: if source is point-like, observed = true * (beam response)
        # For simplicity, assume beam correction factor based on beam area
        beam_correction_factor = 1.0 / max(beam_solid_angle, 1e-10) if beam_solid_angle > 0 else 1.0

        # Apply beam correction to all Stokes parameters
        stokes_i *= beam_correction_factor
        stokes_q *= beam_correction_factor
        stokes_u *= beam_correction_factor
        stokes_v *= beam_correction_factor

        # Propagate noise uncertainties
        # Uncertainty on I is the noise sigma
        uncertainty_i = noise_sigma
        # Uncertainty on Q, U, V is sqrt(2) * noise_sigma (from two orthogonal measurements)
        uncertainty_q = np.sqrt(2.0) * noise_sigma
        uncertainty_u = np.sqrt(2.0) * noise_sigma
        uncertainty_v = np.sqrt(2.0) * noise_sigma

        # Compute derived quantities
        total_polarization = np.sqrt(stokes_q**2 + stokes_u**2 + stokes_v**2)
        degree_of_polarization = total_polarization / stokes_i if stokes_i > 0 else 0.0

        # Compute polarization angle from Q and U
        if abs(stokes_q) > 1e-15 or abs(stokes_u) > 1e-15:
            computed_pol_angle = np.degrees(0.5 * np.arctan2(stokes_u, stokes_q))
            # Normalize to [0, 180)
            computed_pol_angle = computed_pol_angle % 180.0
        else:
            computed_pol_angle = 0.0

        return StokesOutput(
            timestamp=input_data.timestamp,
            coordinate_system=input_data.coordinate_system,
            ra=input_data.ra,
            dec=input_data.dec,
            stokes_i=float(stokes_i),
            stokes_q=float(stokes_q),
            stokes_u=float(stokes_u),
            stokes_v=float(stokes_v),
            uncertainty_i=float(uncertainty_i),
            uncertainty_q=float(uncertainty_q),
            uncertainty_u=float(uncertainty_u),
            uncertainty_v=float(uncertainty_v),
            total_polarization=float(total_polarization),
            polarization_angle=float(computed_pol_angle),
            degree_of_polarization=float(degree_of_polarization),
            beam_corrected=True,
        )

    def compute_polarization_angle(self, q: float, u: float) -> float:
        """Compute polarization angle from Q and U Stokes parameters.

        Args:
            q: Stokes Q parameter.
            u: Stokes U parameter.

        Returns:
            Polarization angle in degrees, normalized to [0, 180).
        """
        if abs(q) < 1e-15 and abs(u) < 1e-15:
            return 0.0
        angle = np.degrees(0.5 * np.arctan2(u, q))
        return angle % 180.0

    def compute_degree_of_polarization(
        self, i: float,
"""
Atmospheric Extinction Model Engine for AstroGrid.

Computes atmospheric extinction (attenuation) of electromagnetic radiation
as it traverses the Earth's atmosphere from a celestial source at a given
zenith angle to a ground-based observer.

Uses the standard linear extinction model:
    A(z) = k * X(z)
where k is the extinction coefficient (mag/airmass) and X(z) is the airmass.

Airmass is computed using the Kasten & Young (1989) formula:
    X(z) = 1 / (cos(z) + 0.50572 * (96.07995 - z_deg)^-1.6364)
"""

import math
from typing import Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, field_validator


class ExtinctionModelType(str, Enum):
    """Supported atmospheric extinction model types."""
    LINEAR = "linear"
    WAVELENGTH_DEPENDENT = "wavelength_dependent"


class ExtinctionParameters(BaseModel):
    """Parameters for the atmospheric extinction model."""
    model_type: ExtinctionModelType = Field(
        default=ExtinctionModelType.LINEAR,
        description="Type of extinction model to use."
    )
    extinction_coefficient: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Extinction coefficient k in magnitudes per airmass."
    )
    reference_wavelength_nm: float = Field(
        default=550.0,
        gt=0.0,
        description="Reference wavelength in nanometers for wavelength-dependent models."
    )
    wavelength_power: float = Field(
        default=-1.0,
        description="Power-law exponent for wavelength dependence: k(λ) = k_ref * (λ/λ_ref)^n"
    )


class ExtinctionResult(BaseModel):
    """Result of an atmospheric extinction calculation."""
    zenith_angle_deg: float = Field(description="Input zenith angle in degrees.")
    airmass: float = Field(description="Computed airmass.")
    extinction_magnitudes: float = Field(description="Extinction in magnitudes.")
    transmission: float = Field(description="Fractional transmission (10^(-0.4*A)).")
    wavelength_nm: Optional[float] = Field(
        default=None,
        description="Wavelength in nm if wavelength-dependent model was used."
    )


class AtmosphericExtinctionEngine:
    """
    Computational engine for atmospheric extinction calculations.

    Supports:
      - Linear extinction model: A(z) = k * X(z)
      - Wavelength-dependent model: A(z, λ) = k(λ) * X(z)
        where k(λ) = k_ref * (λ / λ_ref)^n
    """

    def __init__(self, params: Optional[ExtinctionParameters] = None):
        self.params = params or ExtinctionParameters()

    def compute_airmass(self, zenith_angle_deg: float) -> float:
        """
        Compute airmass using the Kasten & Young (1989) formula.

        Args:
            zenith_angle_deg: Zenith angle in degrees (0 to 90).

        Returns:
            Airmass value (dimensionless).
        """
        z_deg = float(zenith_angle_deg)
        if z_deg < 0.0:
            z_deg = 0.0
        if z_deg > 90.0:
            z_deg = 90.0

        z_rad = math.radians(z_deg)
        cos_z = math.cos(z_rad)

        # Kasten & Young (1989) airmass formula
        # X = 1 / (cos(z) + 0.50572 * (96.07995 - z_deg)^(-1.6364))
        if z_deg >= 90.0:
            return float('inf')

        denominator = cos_z + 0.50572 * (96.07995 - z_deg) ** (-1.6364)
        if denominator <= 0.0:
            return float('inf')

        return 1.0 / denominator

    def compute_extinction_coefficient(self, wavelength_nm: Optional[float] = None) -> float:
        """
        Compute the extinction coefficient, optionally wavelength-dependent.

        Args:
            wavelength_nm: Wavelength in nanometers. If None, returns base coefficient.

        Returns:
            Extinction coefficient in mag/airmass.
        """
        if wavelength_nm is None or self.params.model_type == ExtinctionModelType.LINEAR:
            return self.params.extinction_coefficient

        # Wavelength-dependent: k(λ) = k_ref * (λ / λ_ref)^n
        ratio = wavelength_nm / self.params.reference_wavelength_nm
        if ratio <= 0.0:
            return self.params.extinction_coefficient
        return self.params.extinction_coefficient * (ratio ** self.params.wavelength_power)

    def compute_extinction(
        self,
        zenith_angle_deg: float,
        wavelength_nm: Optional[float] = None
    ) -> ExtinctionResult:
        """
        Compute atmospheric extinction for a given zenith angle and optional wavelength.

        Args:
            zenith_angle_deg: Zenith angle in degrees.
            wavelength_nm: Wavelength in nanometers (for wavelength-dependent models).

        Returns:
            ExtinctionResult with airmass, extinction in magnitudes, and transmission.
        """
        airmass = self.compute_airmass(zenith_angle_deg)
        k = self.compute_extinction_coefficient(wavelength_nm)
        extinction_mag = k * airmass

        # Transmission: T = 10^(-0.4 * A)
        if math.isinf(extinction_mag):
            transmission = 0.0
        else:
            transmission = 10.0 ** (-0.4 * extinction_mag)

        return ExtinctionResult(
            zenith_angle_deg=float(zenith_angle_deg),
            airmass=airmass,
            extinction_magnitudes=extinction_mag,
            transmission=transmission,
            wavelength_nm=wavelength_nm
        )

    def compute_extinction_array(
        self,
        zenith_angles_deg: np.ndarray,
        wavelength_nm: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized computation of extinction over an array of zenith angles.

        Args:
            zenith_angles_deg: Array of zenith angles in degrees.
            wavelength_nm: Wavelength in nanometers.

        Returns:
            Tuple of (airmasses, extinctions_mag, transmissions) as numpy arrays.
        """
        z = np.asarray(zenith_angles_deg, dtype=np.float64)
        z = np.clip(z, 0.0, 90.0)

        z_rad = np.radians(z)
        cos_z = np.cos(z_rad)

        # Kasten & Young airmass (vectorized)
        with np.errstate(divide='ignore', invalid='ignore'):
            denom = cos_z + 0.50572 * (96.07995 - z) ** (-1.6364)
            airmasses = np.where(denom > 0.0, 1.0 / denom, np.inf)

        k = self.compute_extinction_coefficient(wavelength_nm)
        extinctions = k * airmasses

        with np.errstate(over='ignore', invalid='ignore'):
            transmissions = np.where(
                np.isfinite(extinctions),
                10.0 ** (-0.4 * extinctions),
                0.0
            )

        return airmasses, extinctions, transmissions
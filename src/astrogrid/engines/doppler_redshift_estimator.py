"""
Doppler Redshift Estimator Engine.

Computes the dimensionless redshift parameter z from observed spectral line
shifts or broadband photometric flux ratios.
"""

import math
from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator
import numpy as np


class SpectralLine(BaseModel):
    """
    Represents a single observed spectral line.
    """
    rest_wavelength_nm: float = Field(..., gt=0, description="Rest wavelength in nanometers")
    observed_wavelength_nm: float = Field(..., gt=0, description="Observed wavelength in nanometers")
    line_strength: float = Field(1.0, ge=0, description="Relative strength/weight of the line")

    @field_validator("rest_wavelength_nm", "observed_wavelength_nm")
    @classmethod
    def validate_wavelengths(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Wavelength must be positive")
        return v


class PhotometricBand(BaseModel):
    """
    Represents a photometric band measurement.
    """
    band_name: str = Field(..., description="Name of the photometric band")
    flux_density: float = Field(..., gt=0, description="Flux density in Jy")
    effective_wavelength_nm: float = Field(..., gt=0, description="Effective wavelength of the band in nm")
    uncertainty: float = Field(0.0, ge=0, description="Uncertainty in flux density")


class RedshiftEstimate(BaseModel):
    """
    Result of a redshift estimation.
    """
    redshift: float = Field(..., description="Estimated redshift z")
    method: str = Field(..., description="Method used for estimation")
    confidence: float = Field(1.0, ge=0, le=1, description="Confidence score")
    n_lines_used: int = Field(0, ge=0, description="Number of spectral lines used")
    residual_std: float = Field(0.0, ge=0, description="Standard deviation of residuals")


class DopplerRedshiftEstimator:
    """
    Engine for estimating redshift from spectral lines or photometric data.
    """

    def __init__(self, method: Literal["spectral", "photometric", "auto"] = "auto"):
        """
        Initialize the estimator.

        Args:
            method: Estimation method - 'spectral', 'photometric', or 'auto'
        """
        self.method = method

    def estimate_from_spectral_lines(self, lines: List[SpectralLine]) -> RedshiftEstimate:
        """
        Estimate redshift from a list of spectral lines.

        Uses the relation: z = (lambda_obs - lambda_rest) / lambda_rest

        Args:
            lines: List of spectral lines with rest and observed wavelengths

        Returns:
            RedshiftEstimate with the computed redshift
        """
        if not lines:
            raise ValueError("At least one spectral line is required")

        # Compute individual redshift estimates
        z_values = []
        weights = []
        for line in lines:
            z = (line.observed_wavelength_nm - line.rest_wavelength_nm) / line.rest_wavelength_nm
            z_values.append(z)
            weights.append(line.line_strength)

        z_array = np.array(z_values)
        w_array = np.array(weights)

        # Weighted mean redshift
        if np.sum(w_array) > 0:
            z_mean = np.sum(z_array * w_array) / np.sum(w_array)
        else:
            z_mean = np.mean(z_array)

        # Compute residuals and standard deviation
        residuals = z_array - z_mean
        if len(residuals) > 1:
            residual_std = float(np.std(residuals))
        else:
            residual_std = 0.0

        # Confidence based on consistency of lines
        if residual_std > 0:
            confidence = max(0.0, 1.0 - residual_std / max(abs(z_mean), 1e-10))
        else:
            confidence = 1.0

        return RedshiftEstimate(
            redshift=float(z_mean),
            method="spectral",
            confidence=float(confidence),
            n_lines_used=len(lines),
            residual_std=residual_std
        )

    def estimate_from_photometry(self, bands: List[PhotometricBand]) -> RedshiftEstimate:
        """
        Estimate redshift from photometric flux ratios.

        Uses the approximation that for a flat spectrum, the flux ratio between
        two bands relates to the redshift via:
        F_obs / F_rest ~ (1 + z)^(-1) * (lambda_rest / lambda_obs)

        Args:
            bands: List of photometric band measurements

        Returns:
            RedshiftEstimate with the computed redshift
        """
        if len(bands) < 2:
            raise ValueError("At least two photometric bands are required")

        # Sort by effective wavelength
        sorted_bands = sorted(bands, key=lambda b: b.effective_wavelength_nm)

        # Use the ratio of the longest to shortest wavelength band
        f_short = sorted_bands[0].flux_density
        f_long = sorted_bands[-1].flux_density
        lam_short = sorted_bands[0].effective_wavelength_nm
        lam_long = sorted_bands[-1].effective_wavelength_nm

        if f_short <= 0 or f_long <= 0:
            raise ValueError("Flux densities must be positive")

        # Expected ratio for flat spectrum
        expected_ratio = lam_long / lam_short
        observed_ratio = f_long / f_short

        # For a power-law SED F ~ lambda^alpha, the relation is:
        # observed_ratio = expected_ratio * (1+z)^(alpha-1)
        alpha = 0.5
        if observed_ratio > 0 and expected_ratio > 0:
            ratio_deviation = observed_ratio / expected_ratio
            if ratio_deviation > 0:
                z = (ratio_deviation ** (1.0 / (alpha - 1.0))) - 1.0
            else:
                z = 0.0
        else:
            z = 0.0

        # Clamp to reasonable range
        z = max(0.0, min(z, 10.0))

        # Confidence based on number of bands and uncertainty
        n_bands = len(bands)
        avg_uncertainty = np.mean([b.uncertainty / b.flux_density for b in bands if b.flux_density > 0])
        confidence = max(0.0, min(1.0, 1.0 - avg_uncertainty * 10.0))
        confidence *= min(1.0, n_bands / 4.0)

        return RedshiftEstimate(
            redshift=float(z),
            method="photometric",
            confidence=float(confidence),
            n_lines_used=n_bands,
            residual_std=float(avg_uncertainty)
        )

    def estimate(self, lines: Optional[List[SpectralLine]] = None,
                 bands: Optional[List[PhotometricBand]] = None) -> RedshiftEstimate:
        """
        Estimate redshift using the configured method.

        Args:
            lines: Optional list of spectral lines
            bands: Optional list of photometric bands

        Returns:
            RedshiftEstimate with the computed redshift
        """
        if self.method == "spectral" or (self.method == "auto" and lines and not bands):
            if not lines:
                raise ValueError("Spectral lines required for spectral method")
            return self.estimate_from_spectral_lines(lines)
        elif self.method == "photometric" or (self.method == "auto" and bands and not lines):
            if not bands:
                raise ValueError("Photometric bands required for photometric method")
            return self.estimate_from_photometry(bands)
        elif self.method == "auto" and lines and bands:
            # Use spectral method as it is more precise
            return self.estimate_from_spectral_lines(lines)
        else:
            raise ValueError("Insufficient data: provide spectral lines or photometric bands")
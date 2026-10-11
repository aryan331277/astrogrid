"""Radial velocity solver for AstroGrid platform.

Computes line-of-sight velocity of celestial objects with barycentric
corrections and relativistic effects.
"""

import numpy as np
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Union
from datetime import datetime
from enum import Enum


class CoordinateSystem(str, Enum):
    ICRS = "ICRS"
    GCRS = "GCRS"
    BCRS = "BCRS"


class RadialVelocityInput(BaseModel):
    """Input parameters for radial velocity computation."""
    ra: float = Field(..., ge=0.0, le=360.0, description="Right ascension in degrees")
    dec: float = Field(..., ge=-90.0, le=90.0, description="Declination in degrees")
    distance_pc: Optional[float] = Field(None, gt=0.0, description="Distance in parsecs")
    observer_ra: float = Field(0.0, ge=0.0, le=360.0, description="Observer RA in degrees")
    observer_dec: float = Field(0.0, ge=-90.0, le=90.0, description="Observer Dec in degrees")
    observer_velocity: Optional[List[float]] = Field(
        None, description="Observer velocity vector [vx, vy, vz] in m/s"
    )
    barycentric_velocity: Optional[List[float]] = Field(
        None, description="Barycentric velocity vector [vx, vy, vz] in m/s"
    )
    coordinate_system: CoordinateSystem = Field(CoordinateSystem.ICRS)
    timestamp: Optional[datetime] = Field(None, description="Observation timestamp")

    @field_validator("observer_velocity", "barycentric_velocity", mode="before")
    @classmethod
    def validate_velocity_vector(cls, v):
        if v is None:
            return None
        if not isinstance(v, (list, tuple)) or len(v) != 3:
            raise ValueError("Velocity vector must be a list of 3 floats")
        return [float(x) for x in v]


class RadialVelocityResult(BaseModel):
    """Result of radial velocity computation."""
    radial_velocity_ms: float = Field(..., description="Radial velocity in m/s")
    radial_velocity_kms: float = Field(..., description="Radial velocity in km/s")
    barycentric_correction_ms: float = Field(0.0, description="Barycentric correction in m/s")
    observer_correction_ms: float = Field(0.0, description="Observer motion correction in m/s")
    relativistic_correction_ms: float = Field(0.0, description="Relativistic correction in m/s")
    coordinate_system: CoordinateSystem = Field(CoordinateSystem.ICRS)
    timestamp: Optional[datetime] = None


class RadialVelocitySolver:
    """Computes radial velocity with barycentric and relativistic corrections."""

    SPEED_OF_LIGHT = 299792458.0  # m/s

    def __init__(self):
        pass

    def _deg_to_rad(self, deg: float) -> float:
        return np.deg2rad(deg)

    def _unit_vector_from_ra_dec(self, ra_deg: float, dec_deg: float) -> np.ndarray:
        """Convert RA/Dec to unit vector in ICRS Cartesian coordinates."""
        ra = self._deg_to_rad(ra_deg)
        dec = self._deg_to_rad(dec_deg)
        x = np.cos(dec) * np.cos(ra)
        y = np.cos(dec) * np.sin(ra)
        z = np.sin(dec)
        return np.array([x, y, z], dtype=np.float64)

    def _compute_barycentric_correction(
        self,
        target_unit_vec: np.ndarray,
        barycentric_velocity: Optional[List[float]],
    ) -> float:
        """Compute barycentric velocity correction along line of sight."""
        if barycentric_velocity is None:
            return 0.0
        v_bary = np.array(barycentric_velocity, dtype=np.float64)
        return float(np.dot(target_unit_vec, v_bary))

    def _compute_observer_correction(
        self,
        target_unit_vec: np.ndarray,
        observer_velocity: Optional[List[float]],
    ) -> float:
        """Compute observer motion correction along line of sight."""
        if observer_velocity is None:
            return 0.0
        v_obs = np.array(observer_velocity, dtype=np.float64)
        return float(np.dot(target_unit_vec, v_obs))

    def _compute_relativistic_correction(self, v_total: float) -> float:
        """Compute relativistic correction for radial velocity.

        Uses the special relativistic Doppler formula:
        v_rel = v * (1 - v/c) / (1 + v/c) approximation for small v/c.
        The correction term is the difference between Newtonian and relativistic.
        """
        c = self.SPEED_OF_LIGHT
        if abs(v_total) < 1e-10:
            return 0.0
        beta = v_total / c
        # Relativistic Doppler shift factor
        # For small beta: v_rel ≈ v * (1 - beta^2/2)
        # Correction = v - v_rel ≈ v * beta^2 / 2
        correction = v_total * (beta ** 2) / 2.0
        return float(correction)

    def solve(self, input_data: RadialVelocityInput) -> RadialVelocityResult:
        """Solve for radial velocity given input parameters.

        Args:
            input_data: RadialVelocityInput with target and observer parameters.

        Returns:
            RadialVelocityResult with computed velocities and corrections.
        """
        # Compute target unit vector
        target_unit_vec = self._unit_vector_from_ra_dec(input_data.ra, input_data.dec)

        # Compute corrections
        barycentric_corr = self._compute_barycentric_correction(
            target_unit_vec, input_data.barycentric_velocity
        )
        observer_corr = self._compute_observer_correction(
            target_unit_vec, input_data.observer_velocity
        )

        # Total radial velocity (Newtonian)
        v_total = barycentric_corr - observer_corr

        # Relativistic correction
        relativistic_corr = self._compute_relativistic_correction(v_total)

        # Final radial velocity
        v_final = v_total + relativistic_corr

        return RadialVelocityResult(
            radial_velocity_ms=float(v_final),
            radial_velocity_kms=float(v_final / 1000.0),
            barycentric_correction_ms=float(barycentric_corr),
            observer_correction_ms=float(observer_corr),
            relativistic_correction_ms=float(relativistic_corr),
            coordinate_system=input_data.coordinate_system,
            timestamp=input_data.timestamp,
        )

    def solve_batch(
        self, inputs: List[RadialVelocityInput]
    ) -> List[RadialVelocityResult]:
        """Solve radial velocity for a batch of inputs.

        Args:
            inputs: List of RadialVelocityInput objects.

        Returns:
            List of RadialVelocityResult objects.
        """
        return [self.solve(inp) for inp in inputs]
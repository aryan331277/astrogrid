"""
Pydantic data schemas, telemetry records, and coordinate models for the WCS Tangent Point Solver.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Union
import math


class WcsTangentSolverInput(BaseModel):
    """
    Input schema for the WCS Tangent Point Solver.
    
    Defines the celestial sphere parameters and the observed pixel coordinates
    used to solve for the tangent point (alpha_0, delta_0) and rotation angle theta.
    """
    alpha_obs: List[float] = Field(..., description="List of observed Right Ascension values in radians")
    delta_obs: List[float] = Field(..., description="List of observed Declination values in radians")
    pixel_x: List[float] = Field(..., description="List of pixel x-coordinates")
    pixel_y: List[float] = Field(..., description="List of pixel y-coordinates")
    cdelt: float = Field(..., gt=0.0, description="Pixel scale in radians per pixel")
    initial_alpha0: Optional[float] = Field(None, description="Initial guess for tangent point RA in radians")
    initial_delta0: Optional[float] = Field(None, description="Initial guess for tangent point Dec in radians")
    initial_theta: Optional[float] = Field(None, description="Initial guess for rotation angle in radians")
    max_iterations: int = Field(100, gt=0, description="Maximum number of iterations for solver")
    tolerance: float = Field(1e-10, gt=0.0, description="Convergence tolerance")

    @field_validator("alpha_obs", "delta_obs", "pixel_x", "pixel_y")
    @classmethod
    def validate_lists(cls, v: List[float]) -> List[float]:
        if len(v) < 3:
            raise ValueError("At least 3 observations are required for a unique solution")
        return v

    @field_validator("alpha_obs", "delta_obs")
    @classmethod
    def validate_coordinates(cls, v: List[float]) -> List[float]:
        for val in v:
            if not (-math.pi <= val <= math.pi):
                raise ValueError(f"Coordinate {val} out of range [-pi, pi]")
        return v


class WcsTangentSolverOutput(BaseModel):
    """
    Output schema for the WCS Tangent Point Solver.
    
    Contains the solved tangent point coordinates, rotation angle, and solver metadata.
    """
    alpha0: float = Field(..., description="Solved tangent point Right Ascension in radians")
    delta0: float = Field(..., description="Solved tangent point Declination in radians")
    theta: float = Field(..., description="Solved rotation angle in radians")
    converged: bool = Field(..., description="Whether the solver converged")
    iterations: int = Field(..., description="Number of iterations performed")
    residual_rms: float = Field(..., ge=0.0, description="Root mean square of residuals in pixels")
    cdelt: float = Field(..., gt=0.0, description="Pixel scale in radians per pixel")


class WcsTangentSolverTelemetry(BaseModel):
    """
    Telemetry record for monitoring the WCS Tangent Point Solver.
    """
    timestamp: float = Field(..., description="Unix timestamp of the solve operation")
    input_count: int = Field(..., ge=0, description="Number of input observations")
    converged: bool = Field(..., description="Whether the solver converged")
    iterations: int = Field(..., ge=0, description="Number of iterations performed")
    residual_rms: float = Field(..., ge=0.0, description="RMS residual in pixels")
    solve_time_ms: float = Field(..., ge=0.0, description="Time taken to solve in milliseconds")
    error_message: Optional[str] = Field(None, description="Error message if solver failed")


class TangentPointCoordinate(BaseModel):
    """
    Represents a tangent point on the celestial sphere.
    """
    alpha: float = Field(..., description="Right Ascension in radians")
    delta: float = Field(..., description="Declination in radians")
    theta: float = Field(0.0, description="Rotation angle in radians")

    @field_validator("alpha")
    @classmethod
    def validate_alpha(cls, v: float) -> float:
        if not (-math.pi <= v <= math.pi):
            raise ValueError(f"Alpha {v} out of range [-pi, pi]")
        return v

    @field_validator("delta")
    @classmethod
    def validate_delta(cls, v: float) -> float:
        if not (-math.pi / 2 <= v <= math.pi / 2):
            raise ValueError(f"Delta {v} out of range [-pi/2, pi/2]")
        return v

    def to_degrees(self) -> "TangentPointCoordinate":
        """Convert coordinates to degrees."""
        return TangentPointCoordinate(
            alpha=math.degrees(self.alpha),
            delta=math.degrees(self.delta),
            theta=math.degrees(self.theta)
        )

    def to_radians(self) -> "TangentPointCoordinate":
        """Convert coordinates to radians (identity if already in radians)."""
        return TangentPointCoordinate(
            alpha=math.radians(self.alpha) if abs(self.alpha) > math.pi else self.alpha,
            delta=math.radians(self.delta) if abs(self.delta) > math.pi / 2 else self.delta,
            theta=math.radians(self.theta) if abs(self.theta) > math.pi else self.theta
        )


class PixelCoordinate(BaseModel):
    """
    Represents a pixel coordinate in the detector frame.
    """
    x: float = Field(..., description="Pixel x-coordinate")
    y: float = Field(..., description="Pixel y-coordinate")

    def distance(self, other: "PixelCoordinate") -> float:
        """Calculate Euclidean distance to another pixel coordinate."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


class SolverStatus(BaseModel):
    """
    Enum-like model for solver status.
    """
    status: str = Field(..., description="Status: 'converged', 'diverged', 'max_iterations', 'error'")
    message: Optional[str] = Field(None, description="Additional status message")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        valid_statuses = {"converged", "diverged", "max_iterations", "error"}
        if v not in valid_statuses:
            raise ValueError(f"Invalid status: {v}. Must be one of {valid_statuses}")
        return v
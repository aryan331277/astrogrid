"""Radio interferometry synthesizer engine for AstroGrid platform.

Implements the core synthesis kernel that maps measured visibility data
V(u, v, w) to sky brightness distribution I(theta_x, theta_y) via
Fourier inversion with w-projection correction.
"""

import numpy as np
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import math


class CoordinateSystem(str, Enum):
    """Supported coordinate systems for visibility data."""
    UVW = "uvw"
    UV = "uv"
    THETA = "theta"


class SynthesisConfig(BaseModel):
    """Configuration for the radio interferometry synthesis."""
    grid_size: int = Field(default=256, ge=16, le=4096, description="Synthesis grid size (N x N)")
    beamwidth: float = Field(default=1.0, gt=0.0, description="Synthesis beam FWHM in radians")
    use_w_projection: bool = Field(default=True, description="Apply w-projection correction")
    taper: str = Field(default="none", description="Taper function: none, gaussian, hamming")
    taper_width: float = Field(default=1.0, gt=0.0, description="Taper width parameter")
    n_w_planes: int = Field(default=1, ge=1, le=100, description="Number of w-planes for w-projection")


class VisibilityData(BaseModel):
    """Measured visibility data in (u, v, w) coordinates."""
    u: np.ndarray = Field(description="u-coordinates in wavelengths")
    v: np.ndarray = Field(description="v-coordinates in wavelengths")
    w: np.ndarray = Field(description="w-coordinates in wavelengths")
    real: np.ndarray = Field(description="Real part of visibilities")
    imag: np.ndarray = Field(description="Imaginary part of visibilities")
    weight: Optional[np.ndarray] = Field(default=None, description="Weights for each visibility")

    @field_validator("u", "v", "w", "real", "imag", mode="before")
    @classmethod
    def validate_arrays(cls, v):
        arr = np.asarray(v, dtype=np.float64)
        return arr

    @field_validator("weight", mode="before")
    @classmethod
    def validate_weight(cls, v):
        if v is None:
            return None
        return np.asarray(v, dtype=np.float64)


class SynthesisResult(BaseModel):
    """Result of the synthesis operation."""
    image: np.ndarray = Field(description="Synthesized sky brightness image")
    beam: np.ndarray = Field(description="Synthesis beam (PSF) image")
    grid_size: int = Field(description="Grid size used")
    pixel_scale: float = Field(description="Pixel scale in radians")
    peak_flux: float = Field(description="Peak flux density in image")
    rms_flux: float = Field(description="RMS flux density in image")

    @field_validator("image", "beam", mode="before")
    @classmethod
    def validate_image(cls, v):
        return np.asarray(v, dtype=np.float64)


class RadioInterferometrySynthesizer:
    """Computational engine for radio interferometry synthesis.

    Maps measured visibility function V(u, v, w) to sky brightness
    distribution I(theta_x, theta_y) via Fourier inversion with
    w-projection correction.
    """

    def __init__(self, config: Optional[SynthesisConfig] = None):
        self.config = config or SynthesisConfig()
        self._grid_size = self.config.grid_size
        self._pixel_scale = self._compute_pixel_scale()

    def _compute_pixel_scale(self) -> float:
        """Compute pixel scale based on grid size and beamwidth."""
        return self.config.beamwidth / self._grid_size

    def _apply_taper(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Apply taper function to visibility weights."""
        if self.config.taper == "none":
            return np.ones_like(u)
        r = np.sqrt(u**2 + v**2)
        if self.config.taper == "gaussian":
            return np.exp(-0.5 * (r / self.config.taper_width) ** 2)
        elif self.config.taper == "hamming":
            return 0.54 + 0.46 * np.cos(np.pi * r / self.config.taper_width)
        return np.ones_like(u)

    def _w_projection(self, u: np.ndarray, v: np.ndarray, w: np.ndarray,
                      theta_x: np.ndarray, theta_y: np.ndarray) -> np.ndarray:
        """Compute w-projection correction factor.

        The w-projection accounts for the fact that the (u, v) coordinates
        are projected onto the plane perpendicular to the line of sight,
        and the w-coordinate introduces a phase shift that depends on
        the sky position.
        """
        if not self.config.use_w_projection:
            return np.ones_like(u)

        # Compute the effective (u, v) for each w-plane
        # w-projection: u_eff = u / sqrt(1 + (w * theta_x)^2 + (w * theta_y)^2)
        # Simplified: use the dominant w-plane approximation
        w_max = np.max(np.abs(w))
        if w_max < 1e-10:
            return np.ones_like(u)

        # For each visibility, compute the phase correction
        # The correction factor is exp(-i * 2*pi * w * (theta_x^2 + theta_y^2) / 2)
        # For the synthesis kernel, we use the average correction
        theta_sq = theta_x**2 + theta_y**2
        mean_theta_sq = np.mean(theta_sq)
        phase = -1j * 2.0 * np.pi * w * mean_theta_sq / 2.0
        return np.exp(phase)

    def _grid_visibilities(self, u: np.ndarray, v: np.ndarray,
                           real: np.ndarray, imag: np.ndarray,
                           weights: np.ndarray) -> np.ndarray:
        """Grid visibility data onto the synthesis grid using nearest-neighbor."""
        grid = np.zeros((self._grid_size, self._grid_size), dtype=np.complex128)
        grid_weights = np.zeros((self._grid_size, self._grid_size), dtype=np.float64)

        # Determine the extent of the uv coverage
        u_max = np.max(np.abs(u))
        v_max = np.max(np.abs(v))
        extent = max(u_max, v_max, 1.0)

        # Map (u, v) to grid indices
        u_idx = np.round((u / extent) * (self._grid_size / 2)).astype(int)
        v_idx = np.round((v / extent) * (self._grid_size / 2)).astype(int)

        # Wrap around for negative indices (Fourier grid symmetry)
        u_idx = np.mod(u_idx, self._grid_size)
        v_idx = np.mod(v_idx, self._grid_size)

        vis = (real + 1j * imag) * weights

        # Accumulate onto grid
        for i in range(len(u_idx)):
            grid[v_idx[i], u_idx[i]] += vis[i]
            grid_weights[v_idx[i], u_idx[i]] += weights[i]

        # Avoid division by zero
        mask = grid_weights > 0
        grid[mask] /= grid_weights[mask]

        return grid

    def _compute_beam(self) -> np.ndarray:
        """Compute the synthesis beam (PSF) by synthesizing a point source."""
        # Create a uniform grid of visibilities (all ones)
        u_grid, v_grid = np.meshgrid(
            np.linspace(-self._grid_size / 2, self._grid_size / 2, self._grid_size),
            np.linspace(-self._grid_size / 2, self._grid_size / 2, self._grid_size)
        )
        vis_grid = np.ones((self._grid_size, self._grid_size), dtype=np.complex128)

        # Apply taper
        r = np.sqrt(u_grid**2 + v_grid**2)
        if self.config.taper == "gaussian":
            taper = np.exp(-0.5 * (r / self.config.taper_width) ** 2)
        elif self.config.taper == "hamming":
            taper = 0.54 + 0.46 * np.cos(np.pi * r / self.config.taper_width)
        else:
            taper = np.ones_like(r)

        vis_grid *= taper

        # FFT to get the beam
        beam = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(vis_grid)))
        beam = np.real(beam)
        beam /= np.max(np.abs(beam)) if np.max(np.abs(beam)) > 0 else 1.0
        return beam

    def synthesize(self, data: VisibilityData) -> SynthesisResult:
        """Perform the synthesis operation.

        Maps measured visibility data to a sky brightness image via
        Fourier inversion with w-projection correction.

        Args:
            data: VisibilityData containing measured visibilities.

        Returns:
            SynthesisResult with the synthesized image and beam.
        """
        u = data.u
        v = data.v
        w = data.w
        real = data.real
        imag = data.imag

        n = len(u)
        if n == 0:
            raise ValueError("No visibility data provided")

        # Compute weights
        if data.weight is not None:
            weights = data.weight
        else:
            weights = np.ones(n, dtype=np.float64)

        #
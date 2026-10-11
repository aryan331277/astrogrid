"""Aperture photometry engine for AstroGrid.

Computes source flux by summing pixel values within a defined aperture,
correcting for background, and applying aperture corrections.
"""

import numpy as np
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union, Dict, Any
from enum import Enum


class ApertureType(str, Enum):
    CIRCULAR = "circular"
    ELLIPTICAL = "elliptical"
    RECTANGULAR = "rectangular"


class ApertureConfig(BaseModel):
    """Configuration for an aperture photometry measurement."""
    type: ApertureType = ApertureType.CIRCULAR
    center_x: float = Field(..., description="X coordinate of aperture center in pixels")
    center_y: float = Field(..., description="Y coordinate of aperture center in pixels")
    radius: float = Field(..., gt=0, description="Radius (or semi-major axis) in pixels")
    radius_minor: Optional[float] = Field(None, gt=0, description="Semi-minor axis for elliptical apertures")
    position_angle: float = Field(0.0, description="Position angle in degrees for elliptical apertures")
    n_background: int = Field(3, ge=1, description="Number of background annuli")
    background_inner_radius: Optional[float] = Field(None, description="Inner radius for background annulus")
    background_outer_radius: Optional[float] = Field(None, description="Outer radius for background annulus")
    aperture_correction: float = Field(1.0, description="Aperture correction factor")

    @field_validator("radius_minor")
    @classmethod
    def validate_radius_minor(cls, v, values):
        if values.get("type") == ApertureType.ELLIPTICAL and v is None:
            raise ValueError("radius_minor is required for elliptical apertures")
        return v


class ApertureResult(BaseModel):
    """Result of an aperture photometry measurement."""
    raw_flux: float
    background_flux: float
    net_flux: float
    aperture_flux: float
    background_level: float
    background_std: float
    n_aperture_pixels: int
    n_background_pixels: int
    error: Optional[float] = None


def _create_circular_mask(
    image_shape: tuple,
    center_x: float,
    center_y: float,
    radius: float,
) -> np.ndarray:
    """Create a boolean mask for a circular aperture."""
    h, w = image_shape[:2]
    y, x = np.ogrid[:h, :w]
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    return mask


def _create_elliptical_mask(
    image_shape: tuple,
    center_x: float,
    center_y: float,
    radius_major: float,
    radius_minor: float,
    position_angle: float,
) -> np.ndarray:
    """Create a boolean mask for an elliptical aperture."""
    h, w = image_shape[:2]
    y, x = np.ogrid[:h, :w]
    dx = x - center_x
    dy = y - center_y
    pa_rad = np.deg2rad(position_angle)
    cos_pa = np.cos(pa_rad)
    sin_pa = np.sin(pa_rad)
    # Rotate coordinates
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    mask = (x_rot / radius_major) ** 2 + (y_rot / radius_minor) ** 2 <= 1.0
    return mask


def _create_rectangular_mask(
    image_shape: tuple,
    center_x: float,
    center_y: float,
    radius: float,
) -> np.ndarray:
    """Create a boolean mask for a rectangular aperture (square with side 2*radius)."""
    h, w = image_shape[:2]
    y, x = np.ogrid[:h, :w]
    mask = (np.abs(x - center_x) <= radius) & (np.abs(y - center_y) <= radius)
    return mask


def _create_annulus_mask(
    image_shape: tuple,
    center_x: float,
    center_y: float,
    inner_radius: float,
    outer_radius: float,
    is_elliptical: bool = False,
    radius_minor: Optional[float] = None,
    position_angle: float = 0.0,
) -> np.ndarray:
    """Create a boolean mask for an annulus (ring) region."""
    if is_elliptical:
        outer_mask = _create_elliptical_mask(
            image_shape, center_x, center_y, outer_radius,
            radius_minor if radius_minor else outer_radius, position_angle
        )
        inner_mask = _create_elliptical_mask(
            image_shape, center_x, center_y, inner_radius,
            (radius_minor / outer_radius) * inner_radius if radius_minor else inner_radius,
            position_angle
        )
    else:
        outer_mask = _create_circular_mask(image_shape, center_x, center_y, outer_radius)
        inner_mask = _create_circular_mask(image_shape, center_x, center_y, inner_radius)
    return outer_mask & ~inner_mask


def compute_aperture_photometry(
    image: np.ndarray,
    config: ApertureConfig,
    error_image: Optional[np.ndarray] = None,
) -> ApertureResult:
    """Compute aperture photometry for a source in an image.

    Args:
        image: 2D numpy array of pixel values.
        config: Aperture configuration.
        error_image: Optional 2D numpy array of per-pixel errors.

    Returns:
        ApertureResult with flux measurements.
    """
    if image.ndim != 2:
        raise ValueError("Image must be 2D")

    # Create aperture mask
    if config.type == ApertureType.CIRCULAR:
        aperture_mask = _create_circular_mask(
            image.shape, config.center_x, config.center_y, config.radius
        )
    elif config.type == ApertureType.ELLIPTICAL:
        aperture_mask = _create_elliptical_mask(
            image.shape, config.center_x, config.center_y,
            config.radius, config.radius_minor, config.position_angle
        )
    elif config.type == ApertureType.RECTANGULAR:
        aperture_mask = _create_rectangular_mask(
            image.shape, config.center_x, config.center_y, config.radius
        )
    else:
        raise ValueError(f"Unsupported aperture type: {config.type}")

    n_aperture_pixels = int(np.sum(aperture_mask))
    if n_aperture_pixels == 0:
        raise ValueError("Aperture contains no pixels")

    # Raw flux: sum of pixel values in aperture
    raw_flux = float(np.sum(image[aperture_mask]))

    # Background estimation
    if config.background_inner_radius is not None and config.background_outer_radius is not None:
        inner_r = config.background_inner_radius
        outer_r = config.background_outer_radius
    else:
        inner_r = config.radius * 1.5
        outer_r = config.radius * 2.5

    is_elliptical = config.type == ApertureType.ELLIPTICAL
    bg_mask = _create_annulus_mask(
        image.shape, config.center_x, config.center_y,
        inner_r, outer_r,
        is_elliptical=is_elliptical,
        radius_minor=config.radius_minor,
        position_angle=config.position_angle,
    )

    n_background_pixels = int(np.sum(bg_mask))
    if n_background_pixels > 0:
        bg_values = image[bg_mask]
        background_level = float(np.mean(bg_values))
        background_std = float(np.std(bg_values))
        background_flux = background_level * n_aperture_pixels
    else:
        background_level = 0.0
        background_std = 0.0
        background_flux = 0.0

    # Net flux and aperture-corrected flux
    net_flux = raw_flux - background_flux
    aperture_flux = net_flux * config.aperture_correction

    # Error estimation
    error = None
    if error_image is not None:
        if error_image.shape != image.shape:
            raise ValueError("Error image shape must match image shape")
        aperture_errors = error_image[aperture_mask]
        bg_errors = error_image[bg_mask] if n_background_pixels > 0 else np.array([])
        # Propagate errors: sqrt(sum of squared errors in aperture)
        aperture_err = float(np.sqrt(np.sum(aperture_errors ** 2)))
        bg_err = float(np.sqrt(np.sum(bg_errors ** 2))) if len(bg_errors) > 0 else 0.0
        error = float(np.sqrt(aperture_err ** 2 + bg_err ** 2))

    return ApertureResult(
        raw_flux=raw_flux,
        background_flux=background_flux,
        net_flux=net_flux,
        aperture_flux=aperture_flux,
        background_level=background_level,
        background_std=background_std,
        n_aperture_pixels=n_aperture_pixels,
        n_background_pixels=n_background_pixels,
        error=error,
    )
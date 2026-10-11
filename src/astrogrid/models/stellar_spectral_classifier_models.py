"""
Pydantic data schemas, telemetry records, and coordinate models for stellar_spectral_classifier.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Literal
from enum import Enum
import numpy as np


class SpectralResolutionType(str, Enum):
    """
    Enum for spectral resolution types.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA_HIGH = "ultra_high"


class SpectralBand(str, Enum):
    """
    Enum for spectral bands.
    """
    UV = "uv"
    OPTICAL = "optical"
    NEAR_IR = "near_ir"
    MID_IR = "mid_ir"
    FAR_IR = "far_ir"


class StellarSpectrum(BaseModel):
    """
    Represents a stellar spectrum with wavelength/flux data.
    """
    wavelength: List[float] = Field(..., description="Wavelength values in Angstroms")
    flux: List[float] = Field(..., description="Flux density values in erg/s/cm^2/Angstrom")
    resolution_type: SpectralResolutionType = Field(..., description="Spectral resolution type")
    band: SpectralBand = Field(..., description="Spectral band")
    snr: Optional[float] = Field(None, description="Signal-to-noise ratio")
    exposure_time: Optional[float] = Field(None, description="Exposure time in seconds")

    @field_validator("wavelength", "flux")
    @classmethod
    def validate_length_match(cls, v: List[float], info: Any) -> List[float]:
        return v

    def validate(self) -> None:
        """Validate that wavelength and flux arrays have the same length."""
        if len(self.wavelength) != len(self.flux):
            raise ValueError("Wavelength and flux arrays must have the same length")
        if len(self.wavelength) == 0:
            raise ValueError("Wavelength array must not be empty")

    def to_numpy(self) -> Dict[str, np.ndarray]:
        """Convert to numpy arrays."""
        return {
            "wavelength": np.array(self.wavelength, dtype=np.float64),
            "flux": np.array(self.flux, dtype=np.float64),
        }


class SpectralClassificationResult(BaseModel):
    """
    Result of spectral classification of a stellar spectrum.
    """
    spectral_type: str = Field(..., description="Spectral type (e.g., O, B, A, F, G, K, M)")
    luminosity_class: Optional[str] = Field(None, description="Luminosity class (I-V)")
    temperature: Optional[float] = Field(None, description="Effective temperature in Kelvin")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence score")
    probabilities: Dict[str, float] = Field(..., description="Probability distribution over spectral types")
    features: Optional[Dict[str, float]] = Field(None, description="Extracted spectral features")

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities(cls, v: Dict[str, float]) -> Dict[str, float]:
        total = sum(v.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Probabilities must sum to 1.0, got {total}")
        for key, val in v.items():
            if val < 0.0 or val > 1.0:
                raise ValueError(f"Probability for {key} must be between 0 and 1")
        return v


class SpectralClassifierTelemetry(BaseModel):
    """
    Telemetry record for spectral classifier operations.
    """
    timestamp: float = Field(..., description="Unix timestamp")
    source_id: str = Field(..., description="Unique source identifier")
    input_spectrum_hash: str = Field(..., description="Hash of input spectrum")
    processing_time_ms: float = Field(..., ge=0.0, description="Processing time in milliseconds")
    classification_result: Optional[SpectralClassificationResult] = Field(None)
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    model_version: str = Field(..., description="Classifier model version")
    status: Literal["success", "failure", "timeout"] = Field(..., description="Processing status")


class CoordinateSystem(str, Enum):
    """
    Enum for coordinate systems.
    """
    ICRS = "icrs"
    GALACTIC = "galactic"
    ECLIPTIC = "ecliptic"


class SourceCoordinates(BaseModel):
    """
    Celestial coordinates for a source.
    """
    ra: float = Field(..., ge=0.0, le=360.0, description="Right ascension in degrees")
    dec: float = Field(..., ge=-90.0, le=90.0, description="Declination in degrees")
    coordinate_system: CoordinateSystem = Field(CoordinateSystem.ICRS, description="Coordinate system")
    distance_pc: Optional[float] = Field(None, ge=0.0, description="Distance in parsecs")
    proper_motion_ra: Optional[float] = Field(None, description="Proper motion in RA (mas/yr)")
    proper_motion_dec: Optional[float] = Field(None, description="Proper motion in Dec (mas/yr)")
    radial_velocity: Optional[float] = Field(None, description="Radial velocity in km/s")

    @field_validator("ra")
    @classmethod
    def validate_ra(cls, v: float) -> float:
        if v < 0.0 or v > 360.0:
            raise ValueError("RA must be between 0 and 360 degrees")
        return v

    @field_validator("dec")
    @classmethod
    def validate_dec(cls, v: float) -> float:
        if v < -90.0 or v > 90.0:
            raise ValueError("Dec must be between -90 and 90 degrees")
        return v


class SpectralClassifierRequest(BaseModel):
    """
    Request model for spectral classification.
    """
    source_id: str = Field(..., description="Unique source identifier")
    coordinates: SourceCoordinates = Field(..., description="Source coordinates")
    spectrum: StellarSpectrum = Field(..., description="Stellar spectrum data")
    options: Optional[Dict[str, Any]] = Field(None, description="Optional classifier parameters")

    @field_validator("spectrum")
    @classmethod
    def validate_spectrum(cls, v: StellarSpectrum) -> StellarSpectrum:
        v.validate()
        return v


class SpectralClassifierResponse(BaseModel):
    """
    Response model for spectral classification.
    """
    source_id: str = Field(..., description="Unique source identifier")
    result: SpectralClassificationResult = Field(..., description="Classification result")
    telemetry: SpectralClassifierTelemetry = Field(..., description="Telemetry record")
    processing_time_ms: float = Field(..., ge=0.0, description="Total processing time in milliseconds")
"""
Stellar Spectral Classifier Engine for AstroGrid.

Classifies stellar spectra into spectral types (O, B, A, F, G, K, M)
using feature extraction and template matching.
"""

import numpy as np
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from enum import Enum


class SpectralType(str, Enum):
    """Stellar spectral type classification."""
    O = "O"
    B = "B"
    A = "A"
    F = "F"
    G = "G"
    K = "K"
    M = "M"


class SpectralClassificationResult(BaseModel):
    """Result of spectral classification."""
    spectral_type: SpectralType
    confidence: float = Field(ge=0.0, le=1.0, description="Classification confidence score")
    temperature_estimate: float = Field(description="Estimated effective temperature in Kelvin")
    feature_scores: Dict[str, float] = Field(description="Scores for each spectral type")
    dominant_features: List[str] = Field(description="Key features driving classification")


class StellarSpectralClassifier:
    """
    Classifies stellar spectra into spectral types using feature-based analysis.

    Uses absorption line strengths and continuum shape features to determine
    the spectral type of a star.
    """

    # Approximate temperature ranges for each spectral type (K)
    TEMPERATURE_RANGES: Dict[SpectralType, tuple] = {
        SpectralType.O: (30000, 100000),
        SpectralType.B: (10000, 30000),
        SpectralType.A: (7500, 10000),
        SpectralType.F: (6000, 7500),
        SpectralType.G: (5200, 6000),
        SpectralType.K: (3700, 5200),
        SpectralType.M: (2400, 3700),
    }

    # Key absorption line wavelengths (Angstroms) for feature extraction
    KEY_LINES: Dict[str, float] = {
        "H_beta": 4861.3,
        "H_alpha": 6562.8,
        "Na_D": 5890.0,
        "Ca_H": 3968.5,
        "Ca_K": 3933.7,
        "Mg_b": 5175.0,
        "TiO_band": 7050.0,
        "He_II": 4686.0,
        "He_I": 5876.0,
    }

    def __init__(self, wavelength_range: tuple = (3000.0, 10000.0)):
        """
        Initialize the classifier.

        Args:
            wavelength_range: Tuple of (min_wavelength, max_wavelength) in Angstroms.
        """
        self.wavelength_min = wavelength_range[0]
        self.wavelength_max = wavelength_range[1]

    def _normalize_spectrum(self, flux: np.ndarray) -> np.ndarray:
        """Normalize spectrum to unit peak."""
        peak = np.max(flux)
        if peak <= 0:
            return flux
        return flux / peak

    def _extract_line_depth(self, flux: np.ndarray, wavelengths: np.ndarray,
                            line_wavelength: float, window: float = 10.0) -> float:
        """
        Extract absorption line depth at a given wavelength.

        Args:
            flux: Normalized flux array.
            wavelengths: Wavelength array in Angstroms.
            line_wavelength: Center wavelength of the line.
            window: Half-width of the window around the line (Angstroms).

        Returns:
            Line depth (0 = no absorption, 1 = complete absorption).
        """
        mask = (wavelengths >= line_wavelength - window) & (wavelengths <= line_wavelength + window)
        if not np.any(mask):
            return 0.0

        line_flux = np.min(flux[mask])

        # Estimate continuum from edges of the window
        left_mask = (wavelengths >= line_wavelength - window * 3) & (wavelengths < line_wavelength - window)
        right_mask = (wavelengths > line_wavelength + window) & (wavelengths <= line_wavelength + window * 3)

        continuum_vals = []
        if np.any(left_mask):
            continuum_vals.append(np.median(flux[left_mask]))
        if np.any(right_mask):
            continuum_vals.append(np.median(flux[right_mask]))

        if not continuum_vals:
            return 0.0

        continuum = np.mean(continuum_vals)
        if continuum <= 0:
            return 0.0

        depth = 1.0 - (line_flux / continuum)
        return np.clip(depth, 0.0, 1.0)

    def _compute_continuum_slope(self, flux: np.ndarray, wavelengths: np.ndarray) -> float:
        """
        Compute the slope of the continuum (red/blue ratio).

        Returns:
            Slope value; positive = redder, negative = bluer.
        """
        # Use log-linear fit on the continuum
        valid = flux > 0.01
        if np.sum(valid) < 10:
            return 0.0

        log_flux = np.log(flux[valid] + 1e-10)
        log_wl = np.log(wavelengths[valid])

        coeffs = np.polyfit(log_wl, log_flux, 1)
        return coeffs[0]

    def _compute_features(self, flux: np.ndarray, wavelengths: np.ndarray) -> Dict[str, float]:
        """
        Extract classification features from the spectrum.

        Args:
            flux: Normalized flux array.
            wavelengths: Wavelength array in Angstroms.

        Returns:
            Dictionary of feature values.
        """
        features = {}

        # Absorption line depths
        for name, wl in self.KEY_LINES.items():
            features[f"depth_{name}"] = self._extract_line_depth(flux, wavelengths, wl)

        # Continuum slope
        features["continuum_slope"] = self._compute_continuum_slope(flux, wavelengths)

        # Blue-to-red flux ratio
        blue_mask = wavelengths < 5000.0
        red_mask = wavelengths >= 5000.0
        if np.any(blue_mask) and np.any(red_mask):
            features["blue_red_ratio"] = np.mean(flux[blue_mask]) / (np.mean(flux[red_mask]) + 1e-10)
        else:
            features["blue_red_ratio"] = 1.0

        # Overall flux level
        features["mean_flux"] = float(np.mean(flux))

        return features

    def _score_spectral_type(self, features: Dict[str, float], stype: SpectralType) -> float:
        """
        Score how well the features match a given spectral type.

        Uses weighted feature matching based on known spectral characteristics.
        """
        scores = []

        # He II and He I lines are strong in O and B types
        if stype in (SpectralType.O, SpectralType.B):
            he_weight = 0.3 if stype == SpectralType.O else 0.2
            scores.append(he_weight * features.get("depth_He_II", 0.0))
            scores.append(he_weight * features.get("depth_He_I", 0.0))
            # H lines moderate in B, strong in A
            scores.append(0.2 * features.get("depth_H_beta", 0.0))
            # Blue continuum
            scores.append(0.3 * min(features.get("blue_red_ratio", 1.0), 2.0) / 2.0)

        # H lines peak in A type
        elif stype == SpectralType.A:
            scores.append(0.4 * features.get("depth_H_beta", 0.0))
            scores.append(0.3 * features.get("depth_H_alpha", 0.0))
            scores.append(0.3 * min(features.get("blue_red_ratio", 1.0), 1.5) / 1.5)

        # Metal lines increase from F through M
        elif stype in (SpectralType.F, SpectralType.G, SpectralType.K, SpectralType.M):
            # Ca and Mg lines
            scores.append(0.25 * features.get("depth_Ca_H", 0.0))
            scores.append(0.25 * features.get("depth_Ca_K", 0.0))
            scores.append(0.2 * features.get("depth_Mg_b", 0.0))
            # Na D line
            scores.append(0.15 * features.get("depth_Na_D", 0.0))
            # TiO band is strong in M types
            if stype == SpectralType.M:
                scores.append(0.3 * features.get("depth_TiO_band", 0.0))
            # Redder continuum for later types
            redness = 1.0 / (features.get("blue_red_ratio", 1.0) + 1e-10)
            scores.append(0.15 * min(redness, 3.0) / 3.0)

        # Normalize score to [0, 1]
        total = sum(scores)
        max_possible = len(scores) if scores else 1.0
        return np.clip
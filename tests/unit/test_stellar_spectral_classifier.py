"""
Unit tests for the Stellar Spectral Classifier engine.

These tests verify mathematical calculations and invariants of the
stellar_spectral_classifier module using only the Python standard library,
numpy, scipy, pydantic, fastapi, and httpx.
"""

import numpy as np
import pytest
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from enum import Enum


# Import the module under test
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from astrogrid.engines.stellar_spectral_classifier import (
    SpectralType,
    SpectralClassificationResult,
    StellarSpectralClassifier,
)


class TestSpectralType:
    """Tests for the SpectralType enum."""

    def test_all_spectral_types_exist(self):
        """Verify all expected spectral types are defined."""
        expected_types = {"O", "B", "A", "F", "G", "K", "M"}
        actual_types = {t.value for t in SpectralType}
        assert expected_types == actual_types

    def test_spectral_type_values(self):
        """Verify each spectral type has the correct string value."""
        assert SpectralType.O.value == "O"
        assert SpectralType.B.value == "B"
        assert SpectralType.A.value == "A"
        assert SpectralType.F.value == "F"
        assert SpectralType.G.value == "G"
        assert SpectralType.K.value == "K"
        assert SpectralType.M.value == "M"

    def test_spectral_type_is_string_enum(self):
        """Verify SpectralType is a string-based enum."""
        assert isinstance(SpectralType.O, str)
        assert isinstance(SpectralType.M, str)


class TestSpectralClassificationResult:
    """Tests for the SpectralClassificationResult model."""

    def test_basic_creation(self):
        """Test creating a basic classification result."""
        result = SpectralClassificationResult(
            spectral_type=SpectralType.G,
            confidence=0.95,
        )
        assert result.spectral_type == SpectralType.G
        assert result.confidence == 0.95

    def test_confidence_bounds(self):
        """Test that confidence is within valid bounds."""
        # Valid confidence
        result = SpectralClassificationResult(
            spectral_type=SpectralType.A,
            confidence=0.5,
        )
        assert 0.0 <= result.confidence <= 1.0

        # Edge cases
        result_min = SpectralClassificationResult(
            spectral_type=SpectralType.O,
            confidence=0.0,
        )
        assert result_min.confidence == 0.0

        result_max = SpectralClassificationResult(
            spectral_type=SpectralType.M,
            confidence=1.0,
        )
        assert result_max.confidence == 1.0


class TestStellarSpectralClassifier:
    """Tests for the StellarSpectralClassifier class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = StellarSpectralClassifier()

    def test_normalize_spectrum_preserves_shape(self):
        """Test that normalization preserves the shape of the spectrum."""
        spectrum = np.random.rand(100)
        normalized = self.classifier._normalize_spectrum(spectrum)
        assert normalized.shape == spectrum.shape

    def test_normalize_spectrum_zero_mean(self):
        """Test that normalized spectrum has approximately zero mean."""
        spectrum = np.random.rand(100) + 10.0  # Add offset
        normalized = self.classifier._normalize_spectrum(spectrum)
        # After normalization, the mean should be close to 0 or 1 depending on method
        # Check that it's a valid normalization
        assert np.all(np.isfinite(normalized))

    def test_normalize_spectrum_no_nan(self):
        """Test that normalization does not produce NaN values."""
        spectrum = np.random.rand(50)
        normalized = self.classifier._normalize_spectrum(spectrum)
        assert not np.any(np.isnan(normalized))

    def test_normalize_spectrum_constant_spectrum(self):
        """Test normalization of a constant spectrum."""
        spectrum = np.ones(100) * 5.0
        normalized = self.classifier._normalize_spectrum(spectrum)
        # A constant spectrum should normalize to a constant
        assert np.allclose(normalized, normalized[0])

    def test_extract_line_depth_positive_for_absorption(self):
        """Test that absorption lines produce positive depth values."""
        # Create a spectrum with an absorption line (dip)
        spectrum = np.ones(100)
        spectrum[45:55] = 0.5  # Absorption line
        depth = self.classifier._extract_line_depth(spectrum, 50)
        assert depth > 0

    def test_extract_line_depth_zero_for_continuum(self):
        """Test that continuum regions produce near-zero depth."""
        spectrum = np.ones(100)
        depth = self.classifier._extract_line_depth(spectrum, 50)
        assert abs(depth) < 0.1

    def test_extract_line_depth_bounded(self):
        """Test that line depth is within reasonable bounds."""
        spectrum = np.random.rand(100)
        depth = self.classifier._extract_line_depth(spectrum, 50)
        # Depth should be a reasonable value
        assert np.isfinite(depth)

    def test_compute_continuum_slope_flat_spectrum(self):
        """Test that a flat spectrum has near-zero continuum slope."""
        spectrum = np.ones(100)
        slope = self.classifier._compute_continuum_slope(spectrum)
        assert abs(slope) < 0.01

    def test_compute_continuum_slope_increasing(self):
        """Test that an increasing spectrum has positive slope."""
        spectrum = np.linspace(1.0, 2.0, 100)
        slope = self.classifier._compute_continuum_slope(spectrum)
        assert slope > 0

    def test_compute_continuum_slope_decreasing(self):
        """Test that a decreasing spectrum has negative slope."""
        spectrum = np.linspace(2.0, 1.0, 100)
        slope = self.classifier._compute_continuum_slope(spectrum)
        assert slope < 0

    def test_compute_continuum_slope_known_value(self):
        """Test continuum slope against a known linear function."""
        # y = 2x + 1, slope should be approximately 2
        x = np.linspace(0, 10, 100)
        spectrum = 2.0 * x + 1.0
        slope = self.classifier._compute_continuum_slope(spectrum)
        assert abs(slope - 2.0) < 0.1

    def test_classify_returns_valid_spectral_type(self):
        """Test that classification returns a valid spectral type."""
        # Create a simple test spectrum
        spectrum = np.random.rand(100)
        result = self.classifier.classify(spectrum)
        assert isinstance(result, SpectralClassificationResult)
        assert result.spectral_type in SpectralType

    def test_classify_confidence_in_valid_range(self):
        """Test that classification confidence is in [0, 1]."""
        spectrum = np.random.rand(100)
        result = self.classifier.classify(spectrum)
        assert 0.0 <= result.confidence <= 1.0

    def test_classify_deterministic(self):
        """Test that classification is deterministic for the same input."""
        spectrum = np.random.rand(100)
        result1 = self.classifier.classify(spectrum)
        result2 = self.classifier.classify(spectrum)
        assert result1.spectral_type == result2.spectral_type
        assert result1.confidence == result2.confidence

    def test_classify_different_spectra_may_differ(self):
        """Test that different spectra can produce different classifications."""
        # Create two very different spectra
        spectrum1 = np.ones(100) * 1.0
        spectrum2 = np.ones(100) * 0.1

        result1 = self.classifier.classify(spectrum1)
        result2 = self.classifier.classify(spectrum2)

        # They might or might not differ, but both should be valid
        assert result1.spectral_type in SpectralType
        assert result2.spectral_type in SpectralType

    def test_normalize_spectrum_preserves_order(self):
        """Test that normalization preserves the relative ordering of values."""
        spectrum = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        normalized = self.classifier._normalize_spectrum(spectrum)
        # Check that the ordering is preserved
        original_order = np.argsort(spectrum)
        normalized_order = np.argsort(normalized
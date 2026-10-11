"""
Unit tests for stellar_isochrone_interpolator module.

These tests verify the mathematical calculations and invariants of the
StellarIsochroneInterpolator without using any mocks.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from astrogrid.engines.stellar_isochrone_interpolator import (
    IsochroneParameterSpace,
    StellarObservables,
    StellarIsochroneInterpolator,
)


class TestIsochroneParameterSpace:
    """Tests for the IsochroneParameterSpace model."""

    def test_basic_creation(self):
        """Test creating a basic parameter space."""
        space = IsochroneParameterSpace(
            age_range=(1e6, 1e10),
            metallicity_range=(-1.0, 0.5),
            mass_range=(0.1, 10.0),
        )
        assert space.age_range == (1e6, 1e10)
        assert space.metallicity_range == (-1.0, 0.5)
        assert space.mass_range == (0.1, 10.0)

    def test_invalid_age_range(self):
        """Test that invalid age range raises an error."""
        with pytest.raises(Exception):
            IsochroneParameterSpace(
                age_range=(1e10, 1e6),
                metallicity_range=(-1.0, 0.5),
                mass_range=(0.1, 10.0),
            )

    def test_invalid_metallicity_range(self):
        """Test that invalid metallicity range raises an error."""
        with pytest.raises(Exception):
            IsochroneParameterSpace(
                age_range=(1e6, 1e10),
                metallicity_range=(0.5, -1.0),
                mass_range=(0.1, 10.0),
            )

    def test_invalid_mass_range(self):
        """Test that invalid mass range raises an error."""
        with pytest.raises(Exception):
            IsochroneParameterSpace(
                age_range=(1e6, 1e10),
                metallicity_range=(-1.0, 0.5),
                mass_range=(10.0, 0.1),
            )


class TestStellarObservables:
    """Tests for the StellarObservables model."""

    def test_basic_creation(self):
        """Test creating basic stellar observables."""
        obs = StellarObservables(
            mass=1.0,
            luminosity=1.0,
            temperature=5778.0,
        )
        assert obs.mass == 1.0
        assert obs.luminosity == 1.0
        assert obs.temperature == 5778.0

    def test_non_negative_mass(self):
        """Test that negative mass raises an error."""
        with pytest.raises(Exception):
            StellarObservables(
                mass=-1.0,
                luminosity=1.0,
                temperature=5778.0,
            )

    def test_non_negative_luminosity(self):
        """Test that negative luminosity raises an error."""
        with pytest.raises(Exception):
            StellarObservables(
                mass=1.0,
                luminosity=-1.0,
                temperature=5778.0,
            )

    def test_positive_temperature(self):
        """Test that non-positive temperature raises an error."""
        with pytest.raises(Exception):
            StellarObservables(
                mass=1.0,
                luminosity=1.0,
                temperature=0.0,
            )


class TestStellarIsochroneInterpolator:
    """Tests for the StellarIsochroneInterpolator class."""

    def _create_test_interpolator(self):
        """Create a test interpolator with known isochrone data."""
        space = IsochroneParameterSpace(
            age_range=(1e6, 1e10),
            metallicity_range=(-1.0, 0.5),
            mass_range=(0.1, 10.0),
        )

        ages = np.array([1e6, 1e8, 1e10])
        metallicities = np.array([-1.0, 0.0, 0.5])
        masses = np.array([0.5, 1.0, 5.0])

        age_grid, metal_grid, mass_grid = np.meshgrid(ages, metallicities, masses, indexing='ij')
        ages_flat = age_grid.flatten()
        metals_flat = metal_grid.flatten()
        masses_flat = mass_grid.flatten()

        age_factor = np.log10(ages_flat) / 10.0
        metal_factor = 1.0 + 0.1 * metals_flat
        luminosities = masses_flat * age_factor * metal_factor
        temperatures = 5000.0 + 1000.0 * masses_flat

        observables = []
        for i in range(len(ages_flat)):
            obs = StellarObservables(
                mass=float(masses_flat[i]),
                luminosity=float(luminosities[i]),
                temperature=float(temperatures[i]),
            )
            observables.append(obs)

        interpolator = StellarIsochroneInterpolator(
            parameter_space=space,
            isochrone_data=observables,
            ages=ages_flat,
            metallicities=metals_flat,
            masses=masses_flat,
        )
        return interpolator, space

    def test_initialization(self):
        """Test that the interpolator initializes correctly."""
        interpolator, space = self._create_test_interpolator()
        assert interpolator.parameter_space == space
        assert interpolator.is_fitted is False

    def test_fit_creates_interpolator(self):
        """Test that fit() creates the internal interpolator."""
        interpolator, space = self._create_test_interpolator()
        interpolator.fit()
        assert interpolator.is_fitted is True

    def test_interpolate_returns_valid_observables(self):
        """Test that interpolate returns valid stellar observables."""
        interpolator, space = self._create_test_interpolator()
        interpolator.fit()

        result = interpolator.interpolate(
            age=1e8,
            metallicity=0.0,
            mass=1.0,
        )
        assert result is not None
        assert result.mass > 0
        assert result.luminosity > 0
        assert result.temperature > 0

    def test_interpolate_within_bounds(self):
        """Test that interpolation stays within expected bounds."""
        interpolator, space = self._create_test_interpolator()
        interpolator.fit()

        test_points = [
            (1e6, -1.0, 0.5),
            (1e8, 0.0, 1.0),
            (1e10, 0.5, 5.0),
        ]

        for age, metal, mass in test_points:
            result = interpolator.interpolate(
                age=age,
                metallicity=metal,
                mass=mass,
            )
            assert result.mass > 0
            assert result.luminosity > 0
            assert result.temperature > 0

    def test_interpolate_batch(self):
        """Test batch interpolation."""
        interpolator, space = self._create_test_interpolator()
        interpolator.fit()

        ages = np.array([1e6, 1e8, 1e10])
        metals = np.array([-1.0, 0.0, 0.5])
        masses = np.array([0.5, 1.0, 5.0])

        results = interpolator.interpolate_batch(
            ages=ages,
            metallicities=metals,
            masses=masses,
        )

        assert len(results) == 3
        for result in results:
            assert result.mass > 0
            assert result.luminosity > 0
            assert result.temperature > 0

    def test_interpolation_continuity(self):
        """Test that interpolation is continuous by checking nearby points."""
        interpolator, space = self._create_test_interpolator()
        interpolator.fit()

        base_age = 1e8
        base_metal = 0.0
        base_mass = 1.0

        result_base = interpolator.interpolate(
            age=base_age,
            metallicity=base_metal,
            mass=base_mass,
        )

        delta = 1e-3
        result_perturbed = interpolator.interpolate(
            age=base_age * (1 + delta),
            metallicity=base_metal + delta,
            mass=base_mass * (1 + delta),
        )

        assert abs(result_base.luminosity - result_perturbed.luminosity) < 1.
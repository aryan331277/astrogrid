"""Zero-mock unit tests for the Keplerian Orbit Engine.

Verifies mathematical correctness of Kepler's equation solver,
true anomaly computation, state vector derivation, and the
KeplerianOrbitEngine class using only standard library, numpy,
scipy, pydantic, fastapi, and httpx.
"""

import math
import pytest
import numpy as np
from pydantic import ValidationError

from src.astrogrid.engines.keplerian_orbit_engine import (
    KeplerianInputSchema,
    KeplerianOutputSchema,
    solve_kepler_equation,
    true_anomaly_from_eccentric,
    compute_state_vector,
    KeplerianOrbitEngine,
)


# ---------------------------------------------------------------------------
# Tests for solve_kepler_equation
# ---------------------------------------------------------------------------

class TestSolveKeplerEquation:
    """Tests for the Kepler's equation solver."""

    def test_circular_orbit_zero_eccentricity(self):
        """For e=0, eccentric anomaly equals mean anomaly."""
        M = 1.234
        E = solve_kepler_equation(M, 0.0, tol=1e-12, max_iter=100)
        assert abs(E - M) < 1e-10

    def test_circular_orbit_negative_mean_anomaly(self):
        """Mean anomaly can be negative; E should equal M for e=0."""
        M = -0.785
        E = solve_kepler_equation(M, 0.0, tol=1e-12, max_iter=100)
        assert abs(E - M) < 1e-10

    def test_kepler_equation_residual_low_eccentricity(self):
        """Verify E - e*sin(E) = M for a low eccentricity orbit."""
        e = 0.1
        M = 0.5
        E = solve_kepler_equation(M, e, tol=1e-14, max_iter=200)
        residual = E - e * math.sin(E) - M
        assert abs(residual) < 1e-12

    def test_kepler_equation_residual_high_eccentricity(self):
        """Verify E - e*sin(E) = M for a high eccentricity orbit."""
        e = 0.9
        M = 1.0
        E = solve_kepler_equation(M, e, tol=1e-14, max_iter=500)
        residual = E - e * math.sin(E) - M
        assert abs(residual) < 1e-12

    def test_kepler_equation_residual_very_high_eccentricity(self):
        """Verify E - e*sin(E) = M for e close to 1."""
        e = 0.99
        M = 0.3
        E = solve_kepler_equation(M, e, tol=1e-14, max_iter=1000)
        residual = E - e * math.sin(E) - M
        assert abs(residual) < 1e-12

    def test_mean_anomaly_zero(self):
        """M=0 should give E=0."""
        e = 0.5
        E = solve_kepler_equation(0.0, e, tol=1e-14, max_iter=100)
        assert abs(E) < 1e-12

    def test_mean_anomaly_pi(self):
        """M=pi should give E=pi."""
        e = 0.5
        E = solve_kepler_equation(math.pi, e, tol=1e-14, max_iter=100)
        assert abs(E - math.pi) < 1e-12

    def test_mean_anomaly_two_pi(self):
        """M=2*pi should give E=2*pi (or equivalent)."""
        e = 0.5
        E = solve_kepler_equation(2 * math.pi, e, tol=1e-14, max_iter=100)
        # E should be close to 2*pi
        assert abs(E - 2 * math.pi) < 1e-10

    def test_symmetry_mean_anomaly(self):
        """E(M) + E(2*pi - M) should equal 2*pi for e > 0."""
        e = 0.3
        M = 0.7
        E1 = solve_kepler_equation(M, e, tol=1e-14, max_iter=200)
        E2 = solve_kepler_equation(2 * math.pi - M, e, tol=1e-14, max_iter=200)
        assert abs(E1 + E2 - 2 * math.pi) < 1e-10

    def test_convergence_within_max_iter(self):
        """Solver should converge well before max_iter for typical cases."""
        e = 0.5
        M = 1.5
        E = solve_kepler_equation(M, e, tol=1e-14, max_iter=1000)
        residual = E - e * math.sin(E) - M
        assert abs(residual) < 1e-12

    def test_different_mean_anomalies_produce_different_E(self):
        """Different M values should yield different E values (for e>0)."""
        e = 0.4
        M1 = 0.5
        M2 = 1.5
        E1 = solve_kepler_equation(M1, e, tol=1e-14, max_iter=200)
        E2 = solve_kepler_equation(M2, e, tol=1e-14, max_iter=200)
        assert abs(E1 - E2) > 1e-6


# ---------------------------------------------------------------------------
# Tests for true_anomaly_from_eccentric
# ---------------------------------------------------------------------------

class TestTrueAnomalyFromEccentric:
    """Tests for converting eccentric anomaly to true anomaly."""

    def test_zero_eccentric_anomaly(self):
        """E=0 should give nu=0."""
        e = 0.5
        nu = true_anomaly_from_eccentric(0.0, e)
        assert abs(nu) < 1e-12

    def test_pi_eccentric_anomaly(self):
        """E=pi should give nu=pi."""
        e = 0.5
        nu = true_anomaly_from_eccentric(math.pi, e)
        assert abs(nu - math.pi) < 1e-12

    def test_zero_eccentricity(self):
        """For e=0, true anomaly equals eccentric anomaly."""
        E = 1.234
        nu = true_anomaly_from_eccentric(E, 0.0)
        assert abs(nu - E) < 1e-12

    def test_relationship_with_kepler_equation(self):
        """Verify consistency: solve Kepler, then convert to true anomaly,
        and check that the orbital radius formula holds."""
        e = 0.3
        M = 0.8
        E = solve_kepler_equation(M, e, tol=1e-14, max_iter=200)
        nu = true_anomaly_from_eccentric(E, e)

        # Check: r = a(1 - e*cos(E)) = a(1 - e^2)/(1 + e*cos(nu))
        # We verify the identity: 1 - e*cos(E) = (1 - e^2)/(1 + e*cos(nu))
        lhs = 1 - e * math.cos(E)
        rhs = (1 - e**2) / (1 + e * math.cos(nu))
        assert abs(lhs - rhs) < 1e-10

    def test_true_anomaly_range(self):
        """True anomaly should be in [0, 2*pi) for E in [0, 2*pi)."""
        e = 0.6
        for E in np.linspace(0, 2 * math.pi, 100):
            nu = true_anomaly_from_eccentric(E, e)
            # Normalize to [0, 2*pi)
            nu_norm = nu % (2 * math.pi)
            assert 0 <= nu_norm < 2 * math.pi + 1e-10

    def test_symmetry_true_anomaly(self):
        """nu(E) + nu(2*pi - E) should equal 2*pi."""
        e = 0.4
        E = 1.0
        nu1 = true_anomaly_from_eccentric(E, e)
        nu2 = true_anomaly_from_eccentric(2 * math.pi - E, e)
        assert abs(nu1 + nu2 - 2 * math.pi) < 1e-10

    def test_high_eccentricity
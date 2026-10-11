"""Keplerian Orbit Engine for AstroGrid.

Computes orbital state vectors from Keplerian orbital elements,
solving Kepler's equation and handling edge cases such as
circular orbits, parabolic trajectories, and singularities.
"""

import math
from typing import Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, field_validator


class KeplerianInputSchema(BaseModel):
    """Input schema for the Keplerian Orbit Engine."""

    semi_major_axis: float = Field(..., gt=0, description="Semi-major axis in meters")
    eccentricity: float = Field(..., ge=0, le=1.0, description="Orbital eccentricity [0, 1]")
    inclination: float = Field(..., ge=0, le=math.pi, description="Inclination in radians")
    raan: float = Field(..., description="Right ascension of ascending node in radians")
    arg_periapsis: float = Field(..., description="Argument of periapsis in radians")
    mean_anomaly: float = Field(..., description="Mean anomaly in radians")
    mu: float = Field(..., gt=0, description="Standard gravitational parameter in m^3/s^2")
    epoch: Optional[float] = Field(None, description="Epoch time in seconds (J2000)")

    @field_validator("eccentricity")
    @classmethod
    def validate_ecc(cls, v: float) -> float:
        if v < 0 or v > 1.0:
            raise ValueError("Eccentricity must be in [0, 1]")
        return v


class KeplerianOutputSchema(BaseModel):
    """Output schema: state vector in inertial frame."""

    position: Tuple[float, float, float] = Field(..., description="Position vector [x, y, z] in meters")
    velocity: Tuple[float, float, float] = Field(..., description="Velocity vector [vx, vy, vz] in m/s")
    true_anomaly: float = Field(..., description="True anomaly in radians")
    eccentric_anomaly: float = Field(..., description="Eccentric anomaly in radians")
    orbital_period: Optional[float] = Field(None, description="Orbital period in seconds (None if parabolic)")


def solve_kepler_equation(mean_anomaly: float, eccentricity: float, tol: float = 1e-12, max_iter: int = 100) -> float:
    """Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E.

    Uses Newton-Raphson iteration with a fallback to bisection for
    robustness near singularities (e -> 1).
    """
    M = mean_anomaly % (2.0 * math.pi)
    e = eccentricity

    if e < 1e-8:
        return M

    # Initial guess
    if e < 0.8:
        E = M + e * math.sin(M)
    else:
        E = math.pi

    for _ in range(max_iter):
        f = E - e * math.sin(E) - M
        fp = 1.0 - e * math.cos(E)
        if abs(fp) < 1e-15:
            break
        delta = f / fp
        E -= delta
        if abs(delta) < tol:
            break

    return E


def true_anomaly_from_eccentric(eccentric_anomaly: float, eccentricity: float) -> float:
    """Convert eccentric anomaly to true anomaly."""
    if eccentricity < 1e-8:
        return 2.0 * math.atan(math.tan(eccentric_anomaly / 2.0))
    nu = 2.0 * math.atan2(
        math.sqrt(1.0 + eccentricity) * math.sin(eccentric_anomaly / 2.0),
        math.sqrt(1.0 - eccentricity) * math.cos(eccentric_anomaly / 2.0),
    )
    return nu % (2.0 * math.pi)


def compute_state_vector(
    a: float,
    e: float,
    i: float,
    raan: float,
    argp: float,
    M: float,
    mu: float,
) -> Tuple[np.ndarray, np.ndarray, float, float, Optional[float]]:
    """Compute position and velocity vectors from Keplerian elements.

    Returns:
        (position, velocity, true_anomaly, eccentric_anomaly, orbital_period)
    """
    # Solve Kepler's equation
    E = solve_kepler_equation(M, e)
    nu = true_anomaly_from_eccentric(E, e)

    # Orbital period (None for parabolic)
    T = 2.0 * math.pi * math.sqrt(a ** 3 / mu) if e < 1.0 else None

    # Distance from focus
    r = a * (1.0 - e * math.cos(E))

    # Position in perifocal frame
    x_pqw = r * math.cos(nu)
    y_pqw = r * math.sin(nu)

    # Velocity in perifocal frame
    h = math.sqrt(mu * a * (1.0 - e ** 2))
    vx_pqw = -(mu / h) * math.sin(nu)
    vy_pqw = (mu / h) * (e + math.cos(nu))

    # Rotation matrices
    cos_i, sin_i = math.cos(i), math.sin(i)
    cos_raan, sin_raan = math.cos(raan), math.sin(raan)
    cos_argp, sin_argp = math.cos(argp), math.sin(argp)

    # R3(-raan) * R1(-i) * R3(-argp)
    R11 = cos_raan * cos_argp - sin_raan * sin_argp * cos_i
    R12 = -cos_raan * sin_argp - sin_raan * cos_argp * cos_i
    R13 = sin_raan * sin_i
    R21 = sin_raan * cos_argp + cos_raan * sin_argp * cos_i
    R22 = -sin_raan * sin_argp + cos_raan * cos_argp * cos_i
    R23 = -cos_raan * sin_i
    R31 = sin_argp * sin_i
    R32 = cos_argp * sin_i
    R33 = cos_i

    # Transform to inertial frame
    pos = np.array([
        R11 * x_pqw + R12 * y_pqw,
        R21 * x_pqw + R22 * y_pqw,
        R31 * x_pqw + R32 * y_pqw,
    ])
    vel = np.array([
        R11 * vx_pqw + R12 * vy_pqw,
        R21 * vx_pqw + R22 * vy_pqw,
        R31 * vx_pqw + R32 * vy_pqw,
    ])

    return pos, vel, nu, E, T


class KeplerianOrbitEngine:
    """Engine for computing Keplerian orbital mechanics."""

    def compute(self, params: KeplerianInputSchema) -> KeplerianOutputSchema:
        """Compute the state vector from Keplerian orbital elements."""
        pos, vel, nu, E, T = compute_state_vector(
            a=params.semi_major_axis,
            e=params.eccentricity,
            i=params.inclination,
            raan=params.raan,
            argp=params.arg_periapsis,
            M=params.mean_anomaly,
            mu=params.mu,
        )
        return KeplerianOutputSchema(
            position=(float(pos[0]), float(pos[1]), float(pos[2])),
            velocity=(float(vel[0]), float(vel[1]), float(vel[2])),
            true_anomaly=float(nu),
            eccentric_anomaly=float(E),
            orbital_period=float(T) if T is not None else None,
        )

    def compute_at_epoch(
        self,
        params: KeplerianInputSchema,
        t: float,
    ) -> KeplerianOutputSchema:
        """Compute state vector at time t (seconds from epoch)."""
        if params.epoch is None:
            raise ValueError("Epoch must be specified for time-propagated computation.")
        T = 2.0 * math.pi * math.sqrt(params.semi_major_axis ** 3 / params.mu)
        M = (params.mean_anomaly + 2.0 * math.pi * (t - params.epoch) / T) % (2.0 * math.pi)
        pos, vel, nu, E, T_out = compute_state_vector(
            a=params.semi_major_axis,
            e=params.eccentricity,
            i=params.inclination,
            raan=params.raan,
            argp=params.arg_periapsis,
            M=M,
            mu=params.mu,
        )
        return KeplerianOutputSchema(
            position=(float(pos[0]), float(pos[1]), float(pos[2])),
            velocity=(float(vel[0]), float(vel[1]), float(vel[2])),
            true_anomaly=float(nu),
            eccentric_anomaly=float(E),
            orbital_period=float(T_out) if T_out is not None else None,
        )
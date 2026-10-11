"""Celestial coordinate transformation engine for AstroGrid.

Implements transformations between ICRF, ITRF, and GCRS coordinate systems
using standard astronomical algorithms (precession, nutation, Earth rotation).
"""

import numpy as np
from pydantic import BaseModel, Field
from typing import Optional, Tuple
from datetime import datetime, timezone
from enum import Enum


class CoordinateSystem(str, Enum):
    ICRF = "ICRF"
    ITRF = "ITRF"
    GCRS = "GCRS"


class CelestialCoordinate(BaseModel):
    """A point in a celestial coordinate system."""
    system: CoordinateSystem
    ra: float = Field(..., description="Right ascension in radians")
    dec: float = Field(..., description="Declination in radians")
    timestamp: datetime = Field(..., description="Observation time (UTC)")
    distance: Optional[float] = Field(None, description="Distance in AU (for 3D transforms)")


class TransformResult(BaseModel):
    """Result of a coordinate transformation."""
    source_system: CoordinateSystem
    target_system: CoordinateSystem
    ra: float
    dec: float
    distance: Optional[float] = None
    timestamp: datetime


def _julian_day(dt: datetime) -> float:
    """Convert a datetime to Julian Day."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_utc = dt.astimezone(timezone.utc)
    y = dt_utc.year
    m = dt_utc.month
    d = dt_utc.day
    h = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0 + dt_utc.microsecond / 3600000.0
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5 + h / 24.0
    return jd


def _julian_century(jd: float) -> float:
    """Julian centuries since J2000.0."""
    return (jd - 2451545.0) / 36525.0


def _precession_matrix(t: float) -> np.ndarray:
    """Compute the precession rotation matrix from J2000.0 to epoch t (Julian centuries)."""
    zeta_a = (-0.0529496 * t) * 3600.0
    z = (0.0529496 * t) * 3600.0
    theta = (-0.0130125 * t) * 3600.0

    zeta_a_rad = np.radians(zeta_a / 3600.0)
    z_rad = np.radians(z / 3600.0)
    theta_rad = np.radians(theta / 3600.0)

    Rz1 = np.array([
        [np.cos(zeta_a_rad), np.sin(zeta_a_rad), 0],
        [-np.sin(zeta_a_rad), np.cos(zeta_a_rad), 0],
        [0, 0, 1]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_rad), np.sin(theta_rad)],
        [0, -np.sin(theta_rad), np.cos(theta_rad)]
    ])
    Rz2 = np.array([
        [np.cos(z_rad), np.sin(z_rad), 0],
        [-np.sin(z_rad), np.cos(z_rad), 0],
        [0, 0, 1]
    ])

    return Rz2 @ Rx @ Rz1


def _nutation_angles(t: float) -> Tuple[float, float]:
    """Compute nutation in longitude and obliquity (simplified, leading terms only)."""
    T = t
    L = 4.8950630 + 6283.0758500 * T
    Lp = 4.8950630 + 6283.0758500 * T
    F = 3.0586968 + 84398.6923266 * T
    D = 2.5534270 + 529.6909650 * T
    M = 6.2400408 - 1256.6043920 * T
    Mp = 2.9577938 + 477.1988676 * T

    d_lambda = (
        -17.20 * np.sin(F)
        + 1.32 * np.sin(2 * D)
        - 0.23 * np.sin(2 * F)
    )
    d_epsilon = (
        9.20 * np.cos(F)
        - 0.57 * np.cos(2 * D)
        + 0.10 * np.cos(2 * F)
    )

    return np.radians(d_lambda / 3600.0), np.radians(d_epsilon / 3600.0)


def _earth_rotation_angle(jd: float) -> float:
    """Compute the Greenwich Mean Sidereal Time (GMST) in radians."""
    T = _julian_century(jd)
    gmst_deg = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * T**2 - T**3 / 38710000.0
    gmst_deg = gmst_deg % 360.0
    return np.radians(gmst_deg)


def _icrf_to_gcrs(ra: float, dec: float, jd: float) -> Tuple[float, float]:
    """Transform ICRF coordinates to GCRS (geocentric celestial reference system)."""
    t = _julian_century(jd)

    prec_mat = _precession_matrix(t)
    d_lambda, d_epsilon = _nutation_angles(t)

    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    vec = np.array([x, y, z])

    vec_prec = prec_mat @ vec

    Rz_nut = np.array([
        [np.cos(d_lambda), np.sin(d_lambda), 0],
        [-np.sin(d_lambda), np.cos(d_lambda), 0],
        [0, 0, 1]
    ])
    Rx_nut = np.array([
        [1, 0, 0],
        [0, np.cos(d_epsilon), np.sin(d_epsilon)],
        [0, -np.sin(d_epsilon), np.cos(d_epsilon)]
    ])
    vec_nut = Rx_nut @ Rz_nut @ vec_prec

    x_n, y_n, z_n = vec_nut
    ra_out = np.arctan2(y_n, x_n)
    dec_out = np.arcsin(np.clip(z_n, -1.0, 1.0))

    return ra_out, dec_out


def _gcrs_to_itrf(ra: float, dec: float, jd: float) -> Tuple[float, float, float]:
    """Transform GCRS coordinates to ITRF (geocentric terrestrial reference frame).

    Returns (longitude, latitude, altitude) in radians.
    """
    gmst = _earth_rotation_angle(jd)

    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)

    # Rotate by GMST to get ITRF Cartesian
    x_itrf = x * np.cos(gmst) + y * np.sin(gmst)
    y_itrf = -x * np.sin(gmst) + y * np.cos(gmst)
    z_itrf = z

    # Convert to spherical (longitude, latitude, altitude)
    lon = np.arctan2(y_itrf, x_itrf)
    lat = np.arcsin(np.clip(z_itrf, -1.0, 1.0))
    alt = np.sqrt(x_itrf**2 + y_itrf**2 + z_itrf**2)

    return lon, lat, alt


def transform_coordinates(source: CelestialCoordinate, target_system: CoordinateSystem) -> TransformResult:
    """Transform a celestial coordinate from one system to another.

    Supported transformations:
    - ICRF -> GCRS (precession + nutation)
    - GCRS -> ITRF (Earth rotation)
    - ICRF -> ITRF (combined)
    - Reverse transformations
    """
    jd = _julian_day(source.timestamp)
    t = _julian_century(jd)

    src = source.system
    tgt = target_system

    if src == tgt:
        return TransformResult(
            source_system=src,
            target_system=tgt,
            ra=source.ra,
            dec=source.dec,
            distance=source.distance,
            timestamp=source.timestamp
        )

    # Normalize
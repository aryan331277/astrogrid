"""Solar System Ephemeris Engine for AstroGrid.

Computes positions and velocities of solar system bodies using
Keplerian orbital elements and numerical propagation.
"""

import numpy as np
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class CelestialBodyType(str, Enum):
    PLANET = "planet"
    MOON = "moon"
    STAR = "star"
    ASTEROID = "asteroid"
    COMET = "comet"


class CelestialBody(BaseModel):
    name: str
    body_type: CelestialBodyType
    mass_kg: float = 0.0
    semi_major_axis_au: float = 0.0
    eccentricity: float = 0.0
    inclination_deg: float = 0.0
    longitude_ascending_node_deg: float = 0.0
    argument_of_perihelion_deg: float = 0.0
    mean_anomaly_deg: float = 0.0
    epoch_jd: float = 2451545.0  # J2000.0


class EphemerisState(BaseModel):
    body_name: str
    jd: float
    x_au: float
    y_au: float
    z_au: float
    vx_au_per_day: float
    vy_au_per_day: float
    vz_au_per_day: float


class SolarSystemEphemerisEngine:
    """Engine for computing solar system body positions and velocities."""

    G_AU3_KG_DAY2 = 2.959122082855911e-4  # G in AU^3 / (kg * day^2)
    AU_PER_M = 1.0 / 149597870700.0

    def __init__(self, bodies: Optional[List[CelestialBody]] = None):
        self.bodies: Dict[str, CelestialBody] = {}
        if bodies:
            for body in bodies:
                self.bodies[body.name] = body

    def add_body(self, body: CelestialBody) -> None:
        self.bodies[body.name] = body

    def solve_kepler(self, mean_anomaly: float, eccentricity: float,
                     tol: float = 1e-12, max_iter: int = 100) -> float:
        """Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E."""
        e = min(eccentricity, 0.999999)
        if e < 1e-8:
            return mean_anomaly
        # Initial guess
        if e < 0.8:
            E = mean_anomaly + e * np.sin(mean_anomaly)
        else:
            E = np.pi
        for _ in range(max_iter):
            f = E - e * np.sin(E) - mean_anomaly
            fp = 1.0 - e * np.cos(E)
            delta = f / fp
            E -= delta
            if abs(delta) < tol:
                break
        return E

    def orbital_elements_to_state(self, body: CelestialBody, jd: float) -> EphemerisState:
        """Convert orbital elements to Cartesian state vector at given JD."""
        # Time since epoch in days
        dt = jd - body.epoch_jd

        # Mean motion (rad/day)
        a = body.semi_major_axis_au
        if a <= 0:
            return EphemerisState(
                body_name=body.name, jd=jd,
                x_au=0.0, y_au=0.0, z_au=0.0,
                vx_au_per_day=0.0, vy_au_per_day=0.0, vz_au_per_day=0.0
            )

        # Mean motion: n = 2*pi / T, T = 2*pi*sqrt(a^3/GM)
        # For heliocentric: n = sqrt(GM_sun / a^3)
        GM_sun = 1.32712440018e20  # m^3/s^2
        GM_sun_au3_day2 = GM_sun * (self.AU_PER_M ** 3) * (86400.0 ** 2)
        n = np.sqrt(GM_sun_au3_day2 / (a ** 3))  # rad/day

        # Mean anomaly at time jd
        M0 = np.radians(body.mean_anomaly_deg)
        M = (M0 + n * dt) % (2.0 * np.pi)

        # Solve Kepler's equation
        E = self.solve_kepler(M, body.eccentricity)

        # True anomaly
        e = body.eccentricity
        nu = 2.0 * np.arctan2(
            np.sqrt(1.0 + e) * np.sin(E / 2.0),
            np.sqrt(1.0 - e) * np.cos(E / 2.0)
        )

        # Radius
        r = a * (1.0 - e * np.cos(E))

        # Position in orbital plane
        x_orb = r * np.cos(nu)
        y_orb = r * np.sin(nu)

        # Velocity in orbital plane
        h = np.sqrt(GM_sun_au3_day2 * a * (1.0 - e ** 2))
        vx_orb = (-h / r) * np.sin(nu)
        vy_orb = (h / r) * (e * np.cos(nu) + 1.0)

        # Rotation angles
        omega = np.radians(body.argument_of_perihelion_deg)
        Omega = np.radians(body.longitude_ascending_node_deg)
        inc = np.radians(body.inclination_deg)

        # Rotation matrix: orbital plane -> ecliptic
        cos_w, sin_w = np.cos(omega), np.sin(omega)
        cos_O, sin_O = np.cos(Omega), np.sin(Omega)
        cos_i, sin_i = np.cos(inc), np.sin(inc)

        # Position
        x = (cos_O * cos_w - sin_O * sin_w * cos_i) * x_orb + \
            (-cos_O * sin_w - sin_O * cos_w * cos_i) * y_orb
        y = (sin_O * cos_w + cos_O * sin_w * cos_i) * x_orb + \
            (-sin_O * sin_w + cos_O * cos_w * cos_i) * y_orb
        z = (sin_w * sin_i) * x_orb + (cos_w * sin_i) * y_orb

        # Velocity
        vx = (cos_O * cos_w - sin_O * sin_w * cos_i) * vx_orb + \
             (-cos_O * sin_w - sin_O * cos_w * cos_i) * vy_orb
        vy = (sin_O * cos_w + cos_O * sin_w * cos_i) * vx_orb + \
             (-sin_O * sin_w + cos_O * cos_w * cos_i) * vy_orb
        vz = (sin_w * sin_i) * vx_orb + (cos_w * sin_i) * vy_orb

        return EphemerisState(
            body_name=body.name, jd=jd,
            x_au=x, y_au=y, z_au=z,
            vx_au_per_day=vx, vy_au_per_day=vy, vz_au_per_day=vz
        )

    def get_state(self, body_name: str, jd: float) -> EphemerisState:
        """Get the state vector for a named body at a given Julian Date."""
        if body_name not in self.bodies:
            raise ValueError(f"Body '{body_name}' not found in ephemeris.")
        return self.orbital_elements_to_state(self.bodies[body_name], jd)

    def get_all_states(self, jd: float) -> List[EphemerisState]:
        """Get state vectors for all bodies at a given Julian Date."""
        return [self.orbital_elements_to_state(body, jd) for body in self.bodies.values()]

    def distance_au(self, body1: str, body2: str, jd: float) -> float:
        """Compute distance between two bodies in AU at a given JD."""
        s1 = self.get_state(body1, jd)
        s2 = self.get_state(body2, jd)
        dx = s1.x_au - s2.x_au
        dy = s1.y_au - s2.y_au
        dz = s1.z_au - s2.z_au
        return float(np.sqrt(dx * dx + dy * dy + dz * dz))

    def relative_velocity_au_per_day(self, body1: str, body2: str, jd: float) -> float:
        """Compute relative speed between two bodies in AU/day at a given JD."""
        s1 = self.get_state(body1, jd)
        s2 = self.get_state(body2, jd)
        dvx = s1.vx_au_per_day - s2.vx_au_per_day
        dvy = s1.vy_au_per_day - s2.vy_au_per_day
        dvz = s1.vz_au_per_day - s2.vz_au_per_day
        return float(np.sqrt(dvx * dvx + dvy * dvy + dvz * dvz))

    @staticmethod
    def default_solar_system() -> List[CelestialBody]:
        """Return default orbital elements for major solar system bodies (J2000.0
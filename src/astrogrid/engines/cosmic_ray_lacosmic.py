"""
Cosmic Ray Propagation and Interaction Engine.

Models the transport of high-energy cosmic rays through the interstellar medium
and their interactions with background fields (CMB, EBL).
"""

import numpy as np
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Literal
from enum import Enum
import datetime


class CoordinateSystem(str, Enum):
    ICRS = "icrs"
    GALACTIC = "galactic"
    ECLIPTIC = "ecliptic"
    LOCAL = "local"


class CosmicRaySpecies(str, Enum):
    PROTON = "proton"
    ELECTRON = "electron"
    PHOTON = "photon"
    NEUTRINO = "neutrino"
    NUCLEUS = "nucleus"


class BackgroundField(str, Enum):
    CMB = "cmb"
    EBL = "ebl"
    ISM = "ism"
    MAGNETIC = "magnetic"


class CosmicRayParticle(BaseModel):
    """Represents a single cosmic ray particle."""
    species: CosmicRaySpecies
    energy: float = Field(..., gt=0, description="Energy in GeV")
    momentum: Optional[float] = Field(None, ge=0, description="Momentum in GeV/c")
    mass: float = Field(..., ge=0, description="Rest mass in GeV/c^2")
    charge: int = Field(0, description="Electric charge in units of e")
    position: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0], description="Position in kpc")
    velocity: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0], description="Velocity in c")
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)


class BackgroundFieldState(BaseModel):
    """State of a background field."""
    field_type: BackgroundField
    temperature: Optional[float] = Field(None, gt=0, description="Temperature in K")
    intensity: Optional[float] = Field(None, ge=0, description="Intensity in appropriate units")
    spectrum: Optional[List[float]] = Field(None, description="Spectral distribution")
    spatial_extent: float = Field(1.0, gt=0, description="Spatial extent in kpc")


class PropagationResult(BaseModel):
    """Result of cosmic ray propagation calculation."""
    initial_particle: CosmicRayParticle
    final_particle: CosmicRayParticle
    path_length: float = Field(..., ge=0, description="Path length in kpc")
    energy_loss: float = Field(..., ge=0, description="Total energy loss in GeV")
    interaction_events: List[dict] = Field(default_factory=list, description="List of interaction events")
    survival_probability: float = Field(..., ge=0, le=1, description="Probability of survival")
    computation_time: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)


class CosmicRayLACosmicEngine:
    """
    Engine for cosmic ray propagation and interaction calculations.
    
    Models transport through ISM and interactions with CMB, EBL, and magnetic fields.
    """

    # Physical constants
    CMB_TEMPERATURE = 2.725  # K
    CMB_PEAK_ENERGY = 6.34e-4  # eV
    EBL_PEAK_ENERGY = 1.0  # eV
    ISM_DENSITY = 0.1  # particles/cm^3
    MAGNETIC_FIELD_STRENGTH = 5e-6  # Gauss
    PROTON_MASS = 0.938  # GeV/c^2
    ELECTRON_MASS = 0.000511  # GeV/c^2
    PHOTON_MASS = 0.0
    NEUTRINO_MASS = 0.001  # GeV/c^2 (approximate)

    def __init__(self, coordinate_system: CoordinateSystem = CoordinateSystem.ICRS):
        self.coordinate_system = coordinate_system
        self._background_fields: List[BackgroundFieldState] = []
        self._initialization_time = datetime.datetime.utcnow()

    def add_background_field(self, field: BackgroundFieldState) -> None:
        """Add a background field to the simulation environment."""
        self._background_fields.append(field)

    def remove_background_field(self, field_type: BackgroundField) -> bool:
        """Remove a background field from the simulation environment."""
        initial_len = len(self._background_fields)
        self._background_fields = [f for f in self._background_fields if f.field_type != field_type]
        return len(self._background_fields) < initial_len

    def get_background_fields(self) -> List[BackgroundFieldState]:
        """Get all background fields in the simulation environment."""
        return self._background_fields.copy()

    def _get_particle_mass(self, species: CosmicRaySpecies) -> float:
        """Get the rest mass for a given particle species."""
        mass_map = {
            CosmicRaySpecies.PROTON: self.PROTON_MASS,
            CosmicRaySpecies.ELECTRON: self.ELECTRON_MASS,
            CosmicRaySpecies.PHOTON: self.PHOTON_MASS,
            CosmicRaySpecies.NEUTRINO: self.NEUTRINO_MASS,
            CosmicRaySpecies.NUCLEUS: self.PROTON_MASS,  # Default to proton mass
        }
        return mass_map.get(species, self.PROTON_MASS)

    def _calculate_gamma(self, energy: float, mass: float) -> float:
        """Calculate Lorentz factor from total energy and rest mass."""
        if mass <= 0:
            return np.inf
        return energy / mass

    def _calculate_momentum(self, energy: float, mass: float) -> float:
        """Calculate momentum from total energy and rest mass."""
        if mass <= 0:
            return energy
        gamma = self._calculate_gamma(energy, mass)
        beta = np.sqrt(1 - 1 / (gamma ** 2)) if gamma > 1 else 0
        return gamma * beta * mass

    def _calculate_velocity(self, energy: float, mass: float) -> float:
        """Calculate velocity (in units of c) from total energy and rest mass."""
        if mass <= 0:
            return 1.0
        gamma = self._calculate_gamma(energy, mass)
        if gamma <= 1:
            return 0.0
        return np.sqrt(1 - 1 / (gamma ** 2))

    def _calculate_energy_loss_rate(self, particle: CosmicRayParticle, field: BackgroundFieldState) -> float:
        """
        Calculate energy loss rate for a particle interacting with a background field.
        
        Returns energy loss in GeV/kpc.
        """
        energy = particle.energy
        mass = particle.mass

        if field.field_type == BackgroundField.CMB:
            # Inverse Compton scattering for electrons, pair production for protons
            if particle.species == CosmicRaySpecies.ELECTRON:
                # Approximate IC energy loss
                return 1e-12 * energy ** 2
            elif particle.species == CosmicRaySpecies.PROTON:
                # Pair production on CMB photons
                threshold = 2 * mass * self.CMB_PEAK_ENERGY * 1e9  # Convert eV to GeV
                if energy > threshold:
                    return 1e-15 * (energy - threshold)
                return 0.0
            else:
                return 0.0

        elif field.field_type == BackgroundField.EBL:
            # Interactions with Extragalactic Background Light
            if particle.species == CosmicRaySpecies.ELECTRON:
                return 1e-11 * energy ** 2
            elif particle.species == CosmicRaySpecies.PROTON:
                # Pion production on EBL photons
                threshold = 1.45e10  # GeV (approximate)
                if energy > threshold:
                    return 1e-14 * (energy - threshold)
                return 0.0
            else:
                return 0.0

        elif field.field_type == BackgroundField.ISM:
            # Interactions with interstellar medium
            if particle.species == CosmicRaySpecies.PROTON:
                # Ionization and excitation losses
                return 1e-9 * np.log(1 + energy / 1e3)
            elif particle.species == CosmicRaySpecies.ELECTRON:
                # Synchrotron and bremsstrahlung
                return 1e-10 * energy
            else:
                return 0.0

        elif field.field_type == BackgroundField.MAGNETIC:
            # Synchrotron radiation for charged particles
            if particle.charge != 0 and mass > 0:
                gamma = self._calculate_gamma(energy, mass)
                return 1e-13 * gamma ** 2 * self.MAGNETIC_FIELD_STRENGTH ** 2
            return 0.0

        return 0.0

    def _calculate_interaction_probability(self, particle: CosmicRayParticle, field: BackgroundFieldState, path_length: float) -> float:
        """
        Calculate probability of interaction over a given path length.
        
        Returns probability in [0, 1].
        """
        energy = particle.energy
        mass = particle.mass

        if field.field_type == BackgroundField.CMB:
            if particle.species == CosmicRaySpecies.PROTON:
                threshold = 2 * mass * self.CMB_PEAK_ENERGY * 1e9
                if energy > threshold:
                    cross_section = 1e-28 * (energy / threshold) ** 0.5  # cm^2
                    density =
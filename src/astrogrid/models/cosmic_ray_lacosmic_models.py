"""Pydantic data schemas, telemetry records, and coordinate models for cosmic_ray_lacosmic."""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union, Literal
from enum import Enum
import datetime
import math


class CoordinateSystem(str, Enum):
    ICRS = "icrs"
    GALACTIC = "galactic"
    ECLIPTIC = "ecliptic"
    LOCAL = "local"


class CosmicRaySpecies(str, Enum):
    PROTON = "proton"
    ELECTRON = "electron"
    POSITRON = "positron"
    NEUTRON = "neutron"
    ALPHA = "alpha"
    HEAVY = "heavy"


class InteractionType(str, Enum):
    CMB_PHOTON = "cmb_photon"
    EBL_PHOTON = "ebl_photon"
    ISM_GAS = "ism_gas"
    MAGNETIC_FIELD = "magnetic_field"
    SELF_INTERACTION = "self_interaction"


class CosmicRayParticle(BaseModel):
    """Represents a single cosmic ray particle state."""
    species: CosmicRaySpecies
    energy_gev: float = Field(gt=0.0, description="Particle energy in GeV")
    momentum_gev: float = Field(gt=0.0, description="Particle momentum in GeV/c")
    charge: int = Field(description="Electric charge in units of e")
    mass_gev: float = Field(gt=0.0, description="Rest mass in GeV/c^2")
    direction: List[float] = Field(
        min_length=3, max_length=3,
        description="Unit vector direction [lx, ly, lz]"
    )
    coordinate_system: CoordinateSystem = CoordinateSystem.ICRS
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

    @field_validator("direction")
    @classmethod
    def validate_unit_vector(cls, v: List[float]) -> List[float]:
        norm = math.sqrt(sum(x * x for x in v))
        if abs(norm - 1.0) > 1e-6:
            raise ValueError(f"Direction vector must be unit length, got norm={norm}")
        return v


class CosmicRayInteraction(BaseModel):
    """Records a single interaction event for a cosmic ray particle."""
    interaction_type: InteractionType
    particle: CosmicRayParticle
    target_energy_gev: float = Field(gt=0.0, description="Target field/particle energy in GeV")
    interaction_probability: float = Field(ge=0.0, le=1.0)
    energy_loss_gev: float = Field(ge=0.0, description="Energy lost in this interaction")
    secondary_particles: List[CosmicRayParticle] = Field(default_factory=list)
    position: List[float] = Field(
        min_length=3, max_length=3,
        description="Position in kpc [x, y, z]"
    )
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)


class CosmicRayPropagationStep(BaseModel):
    """A single step in the cosmic ray propagation simulation."""
    step_index: int = Field(ge=0)
    particle: CosmicRayParticle
    path_length_kpc: float = Field(ge=0.0, description="Distance traveled in this step")
    magnetic_field_strength_gauss: float = Field(ge=0.0)
    diffusion_coefficient: float = Field(
        ge=0.0,
        description="Diffusion coefficient in kpc^2/Myr"
    )
    interactions: List[CosmicRayInteraction] = Field(default_factory=list)
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)


class CosmicRaySpectrum(BaseModel):
    """Differential energy spectrum of cosmic rays."""
    species: CosmicRaySpecies
    energies_gev: List[float] = Field(
        min_length=2,
        description="Energy bins in GeV (log-spaced)"
    )
    flux_per_gev: List[float] = Field(
        min_length=2,
        description="Differential flux dN/dE in particles/(cm^2*s*sr*GeV)"
    )
    spectral_index: float = Field(
        description="Power-law spectral index (negative for falling spectrum)"
    )
    normalization: float = Field(
        gt=0.0,
        description="Normalization constant"
    )
    coordinate_system: CoordinateSystem = CoordinateSystem.ICRS
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

    @field_validator("flux_per_gev")
    @classmethod
    def validate_flux_length(cls, v: List[float], info) -> List[float]:
        energies = info.data.get("energies_gev")
        if energies is not None and len(v) != len(energies):
            raise ValueError("flux_per_gev must have same length as energies_gev")
        return v


class CosmicRayTelemetryRecord(BaseModel):
    """Telemetry record for monitoring cosmic ray propagation simulations."""
    record_id: str
    simulation_id: str
    step_index: int = Field(ge=0)
    particle_count: int = Field(ge=0)
    total_energy_gev: float = Field(ge=0.0)
    mean_energy_gev: float = Field(ge=0.0)
    max_energy_gev: float = Field(ge=0.0)
    interaction_count: int = Field(ge=0)
    total_energy_loss_gev: float = Field(ge=0.0)
    diffusion_coefficient_mean: float = Field(ge=0.0)
    magnetic_field_strength_mean_gauss: float = Field(ge=0.0)
    status: Literal["running", "completed", "failed", "paused"] = "running"
    error_message: Optional[str] = None
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)


class CosmicRayFieldConfiguration(BaseModel):
    """Configuration for background fields affecting cosmic ray propagation."""
    cmb_temperature_k: float = Field(default=2.725, gt=0.0, description="CMB temperature in K")
    ebl_photon_density: float = Field(
        default=1.0, gt=0.0,
        description="EBL photon density in photons/cm^3"
    )
    ism_gas_density_cm3: float = Field(
        default=0.1, gt=0.0,
        description="ISM gas density in particles/cm^3"
    )
    magnetic_field_strength_gauss: float = Field(
        default=5.0, gt=0.0,
        description="Mean magnetic field strength in Gauss"
    )
    magnetic_field_coherence_length_kpc: float = Field(
        default=10.0, gt=0.0,
        description="Magnetic field coherence length in kpc"
    )
    diffusion_coefficient_base: float = Field(
        default=1.0, gt=0.0,
        description="Base diffusion coefficient in kpc^2/Myr"
    )
    diffusion_energy_dependence: float = Field(
        default=0.5,
        description="Energy dependence exponent for diffusion (D ~ E^delta)"
    )
    galactic_radius_kpc: float = Field(default=15.0, gt=0.0)
    halo_height_kpc: float = Field(default=15.0, gt=0.0)


class CosmicRaySimulationResult(BaseModel):
    """Complete result from a cosmic ray propagation simulation."""
    simulation_id: str
    configuration: CosmicRayFieldConfiguration
    initial_spectrum: CosmicRaySpectrum
    final_spectrum: Optional[CosmicRaySpectrum] = None
    propagation_steps: List[CosmicRayPropagationStep] = Field(default_factory=list)
    total_interactions: int = Field(ge=0)
    total_energy_loss_gev: float = Field(ge=0.0)
    survival_fraction: float = Field(ge=0.0, le=1.0)
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime] = None
    status: Literal["running", "completed", "failed"] = "running"
    error_message: Optional[str] = None
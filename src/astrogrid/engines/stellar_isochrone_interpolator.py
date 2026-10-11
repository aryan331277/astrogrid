"""
Stellar Isochrone Interpolator Engine.

Maps discrete stellar evolutionary tracks (isochrones) onto a continuous
parameter space to interpolate observable properties (e.g., mass, luminosity,
temperature) for arbitrary input parameters.
"""

import numpy as np
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
from scipy.interpolate import RegularGridInterpolator


class IsochroneParameterSpace(BaseModel):
    """
    Defines the continuous parameter space for isochrone interpolation.
    """
    age: Tuple[float, float] = Field(..., description="Age range in years")
    metallicity: Tuple[float, float] = Field(..., description="Metallicity range [Fe/H]")
    mass: Tuple[float, float] = Field(..., description="Mass range in solar masses")


class StellarObservables(BaseModel):
    """
    Observable properties of a star.
    """
    luminosity: float = Field(..., description="Luminosity in solar units")
    temperature: float = Field(..., description="Effective temperature in Kelvin")
    radius: float = Field(..., description="Radius in solar units")
    mass: float = Field(..., description="Mass in solar units")


class StellarIsochroneInterpolator:
    """
    Interpolates stellar isochrone data onto a continuous parameter space.
    
    Uses RegularGridInterpolator from scipy to perform trilinear interpolation
    over a 3D grid of (age, metallicity, mass) -> observables.
    """
    
    def __init__(self, parameter_space: IsochroneParameterSpace):
        self.parameter_space = parameter_space
        self._interpolators: Dict[str, RegularGridInterpolator] = {}
        self._is_fitted = False
        
    def fit(self, ages: np.ndarray, metallicities: np.ndarray, masses: np.ndarray,
            luminosities: np.ndarray, temperatures: np.ndarray, radii: np.ndarray) -> None:
        """
        Fit the interpolator with discrete isochrone data.
        
        Args:
            ages: 1D array of age values (years)
            metallicities: 1D array of metallicity values [Fe/H]
            masses: 1D array of mass values (solar masses)
            luminosities: 3D array of luminosity values [age_idx, metal_idx, mass_idx]
            temperatures: 3D array of temperature values [age_idx, metal_idx, mass_idx]
            radii: 3D array of radius values [age_idx, metal_idx, mass_idx]
        """
        # Validate input shapes
        n_ages = len(ages)
        n_metals = len(metallicities)
        n_masses = len(masses)
        
        expected_shape = (n_ages, n_metals, n_masses)
        
        if luminosities.shape != expected_shape:
            raise ValueError(f"Luminosities shape {luminosities.shape} != expected {expected_shape}")
        if temperatures.shape != expected_shape:
            raise ValueError(f"Temperatures shape {temperatures.shape} != expected {expected_shape}")
        if radii.shape != expected_shape:
            raise ValueError(f"Radii shape {radii.shape} != expected {expected_shape}")
        
        # Create grid points
        grid_points = (ages, metallicities, masses)
        
        # Create interpolators for each observable
        self._interpolators['luminosity'] = RegularGridInterpolator(
            grid_points, luminosities, method='linear', bounds_error=False, fill_value=None
        )
        self._interpolators['temperature'] = RegularGridInterpolator(
            grid_points, temperatures, method='linear', bounds_error=False, fill_value=None
        )
        self._interpolators['radius'] = RegularGridInterpolator(
            grid_points, radii, method='linear', bounds_error=False, fill_value=None
        )
        
        self._is_fitted = True
    
    def interpolate(self, age: float, metallicity: float, mass: float) -> StellarObservables:
        """
        Interpolate stellar observables for given parameters.
        
        Args:
            age: Age in years
            metallicity: Metallicity [Fe/H]
            mass: Mass in solar units
            
        Returns:
            StellarObservables with interpolated values
        """
        if not self._is_fitted:
            raise RuntimeError("Interpolator has not been fitted. Call fit() first.")
        
        # Check bounds
        if not (self.parameter_space.age[0] <= age <= self.parameter_space.age[1]):
            raise ValueError(f"Age {age} out of bounds {self.parameter_space.age}")
        if not (self.parameter_space.metallicity[0] <= metallicity <= self.parameter_space.metallicity[1]):
            raise ValueError(f"Metallicity {metallicity} out of bounds {self.parameter_space.metallicity}")
        if not (self.parameter_space.mass[0] <= mass <= self.parameter_space.mass[1]):
            raise ValueError(f"Mass {mass} out of bounds {self.parameter_space.mass}")
        
        # Create query point
        query_point = np.array([[age, metallicity, mass]])
        
        # Interpolate each observable
        lum = self._interpolators['luminosity'](query_point)[0]
        temp = self._interpolators['temperature'](query_point)[0]
        rad = self._interpolators['radius'](query_point)[0]
        
        return StellarObservables(
            luminosity=float(lum),
            temperature=float(temp),
            radius=float(rad),
            mass=float(mass)
        )
    
    def interpolate_batch(self, ages: np.ndarray, metallicities: np.ndarray, masses: np.ndarray) -> List[StellarObservables]:
        """
        Interpolate stellar observables for multiple parameter sets.
        
        Args:
            ages: 1D array of age values
            metallicities: 1D array of metallicity values
            masses: 1D array of mass values
            
        Returns:
            List of StellarObservables
        """
        if not self._is_fitted:
            raise RuntimeError("Interpolator has not been fitted. Call fit() first.")
        
        n = len(ages)
        if len(metallicities) != n or len(masses) != n:
            raise ValueError("All input arrays must have the same length")
        
        # Create query points
        query_points = np.column_stack([ages, metallicities, masses])
        
        # Interpolate each observable
        lums = self._interpolators['luminosity'](query_points)
        temps = self._interpolators['temperature'](query_points)
        rads = self._interpolators['radius'](query_points)
        
        results = []
        for i in range(n):
            results.append(StellarObservables(
                luminosity=float(lums[i]),
                temperature=float(temps[i]),
                radius=float(rads[i]),
                mass=float(masses[i])
            ))
        
        return results
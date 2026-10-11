"""
Box-Less Least Squares (BLS) Transit Detector Engine.

Implements the BLS algorithm for detecting periodic transit signals in
light curve data using only numpy and scipy.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


class BLSInputData(BaseModel):
    """Input schema for the BLS transit detector."""
    time: List[float] = Field(..., description="Time stamps of observations")
    flux: List[float] = Field(..., description="Flux measurements")
    flux_err: Optional[List[float]] = Field(None, description="Flux uncertainties")
    period_range: Tuple[float, float] = Field((0.5, 10.0), description="Period search range in days")
    duration_range: Tuple[float, float] = Field((0.01, 0.5), description="Transit duration range in days")
    n_periods: int = Field(1000, description="Number of periods to search")
    n_durations: int = Field(50, description="Number of durations to search")
    transit_depth: Optional[float] = Field(None, description="Expected transit depth (fractional)")


class BLSResult(BaseModel):
    """Output schema for the BLS transit detector."""
    best_period: float = Field(..., description="Best-fit period in days")
    best_duration: float = Field(..., description="Best-fit transit duration in days")
    best_depth: float = Field(..., description="Best-fit transit depth (fractional)")
    best_phase: float = Field(..., description="Best-fit transit phase (0-1)")
    likelihood: float = Field(..., description="Log-likelihood at best fit")
    periodogram: List[float] = Field(..., description="Log-likelihood values for each period")
    periods: List[float] = Field(..., description="Periods searched")


def _compute_bls_likelihood(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    duration: float,
    phase: float,
    depth: float,
) -> float:
    """
    Compute the log-likelihood for a given set of BLS parameters.
    
    Uses a box model for the transit shape.
    """
    # Phase-fold the time series
    t_folded = (time / period + phase) % 1.0
    
    # Define transit window
    half_dur = duration / (2.0 * period)
    in_transit = (t_folded < half_dur) | (t_folded > (1.0 - half_dur))
    
    # Model flux
    model = np.ones_like(flux)
    model[in_transit] = 1.0 - depth
    
    # Compute chi-squared
    residuals = flux - model
    chi2 = np.sum((residuals / flux_err) ** 2)
    
    # Log-likelihood (up to a constant)
    log_likelihood = -0.5 * chi2
    
    return log_likelihood


def _optimize_phase(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    duration: float,
    depth: float,
) -> Tuple[float, float]:
    """
    Optimize the transit phase for given period, duration, and depth.
    Returns (best_phase, best_log_likelihood).
    """
    def neg_log_likelihood(phase: float) -> float:
        ll = _compute_bls_likelihood(time, flux, flux_err, period, duration, phase, depth)
        return -ll
    
    # Coarse search over phase
    n_phase_steps = 100
    phases = np.linspace(0.0, 1.0, n_phase_steps, endpoint=False)
    ll_values = np.array([_compute_bls_likelihood(time, flux, flux_err, period, duration, p, depth) for p in phases])
    
    # Find best coarse phase
    best_idx = np.argmax(ll_values)
    best_phase_coarse = phases[best_idx]
    
    # Fine optimization around best coarse phase
    phase_width = 1.0 / n_phase_steps
    result = minimize_scalar(
        neg_log_likelihood,
        bounds=(max(0.0, best_phase_coarse - phase_width), min(1.0, best_phase_coarse + phase_width)),
        method='bounded',
        options={'xatol': 1e-6}
    )
    
    best_phase = result.x
    best_ll = -result.fun
    
    return best_phase, best_ll


def _optimize_depth(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    duration: float,
    phase: float,
) -> Tuple[float, float]:
    """
    Optimize the transit depth for given period, duration, and phase.
    Returns (best_depth, best_log_likelihood).
    """
    def neg_log_likelihood(depth: float) -> float:
        ll = _compute_bls_likelihood(time, flux, flux_err, period, duration, phase, depth)
        return -ll
    
    # Depth should be between 0 and some reasonable maximum
    result = minimize_scalar(
        neg_log_likelihood,
        bounds=(0.0, 0.5),
        method='bounded',
        options={'xatol': 1e-6}
    )
    
    best_depth = result.x
    best_ll = -result.fun
    
    return best_depth, best_ll


def run_bls_detection(input_data: BLSInputData) -> BLSResult:
    """
    Run the BLS transit detection algorithm.
    
    Args:
        input_data: BLSInputData containing time series and search parameters.
        
    Returns:
        BLSResult with best-fit parameters and periodogram.
    """
    time = np.array(input_data.time, dtype=np.float64)
    flux = np.array(input_data.flux, dtype=np.float64)
    
    if input_data.flux_err is not None:
        flux_err = np.array(input_data.flux_err, dtype=np.float64)
    else:
        # Estimate errors from scatter
        flux_err = np.full_like(flux, np.std(flux))
    
    # Normalize flux
    flux_mean = np.mean(flux)
    flux_norm = flux / flux_mean
    flux_err_norm = flux_err / flux_mean
    
    # Generate period grid
    periods = np.linspace(input_data.period_range[0], input_data.period_range[1], input_data.n_periods)
    
    # Generate duration grid
    durations = np.linspace(input_data.duration_range[0], input_data.duration_range[1], input_data.n_durations)
    
    # If transit depth is provided, use it; otherwise optimize
    fixed_depth = input_data.transit_depth if input_data.transit_depth is not None else None
    
    best_ll = -np.inf
    best_period = periods[0]
    best_duration = durations[0]
    best_phase = 0.0
    best_depth = 0.0
    
    periodogram = []
    
    for period in periods:
        best_period_ll = -np.inf
        best_period_duration = durations[0]
        best_period_phase = 0.0
        best_period_depth = 0.0
        
        for duration in durations:
            if fixed_depth is not None:
                depth = fixed_depth
                # Optimize phase only
                phase, ll = _optimize_phase(time, flux_norm, flux_err_norm, period, duration, depth)
            else:
                # Optimize phase with a default depth, then optimize depth
                default_depth = 0.01
                phase, ll_phase = _optimize_phase(time, flux_norm, flux_err_norm, period, duration, default_depth)
                depth, ll = _optimize_depth(time, flux_norm, flux_err_norm, period, duration, phase)
            
            if ll > best_period_ll:
                best_period_ll = ll
                best_period_duration = duration
                best_period_phase = phase
                best_period_depth = depth
        
        periodogram.append(best_period_ll)
        
        if best_period_ll > best_ll:
            best_ll = best_period_ll
            best_period = period
            best_duration = best_period_duration
            best_phase = best_period_phase
            best_depth = best_period_depth
    
    return BLSResult(
        best_period=float(best_period),
        best_duration=float(best_duration),
        best_depth=float(best_depth),
        best_phase=float(best_phase),
        likelihood=float(best_ll),
        periodogram=[float(x) for x in periodogram],
        periods=[float(p) for p in periods],
    )
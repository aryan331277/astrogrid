"""Light curve detrending engine for AstroGrid.

Provides robust detrending of astronomical light curves using LOWESS
(LOcally WEighted Scatterplot and Smoothing) and Savitzky-Golay filtering.
"""

import numpy as np
from scipy import stats
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
from enum import Enum


class DetrendingMethod(str, Enum):
    """Enum for supported detrending algorithms."""
    LOWESS = "lowess"
    SAVITZKY_GOLAY = "savitzky_golay"
    POLYNOMIAL = "polynomial"


class DetrendResult(BaseModel):
    """Result of a detrending operation."""
    original: List[float] = Field(..., description="Original light curve values")
    detrended: List[float] = Field(..., description="Detrended light curve values")
    trend: List[float] = Field(..., description="Estimated trend component")
    method: str = Field(..., description="Method used for detrending")
    n_points: int = Field(..., description="Number of data points")


def _lowess(
    x: np.ndarray,
    y: np.ndarray,
    frac: float = 0.3,
    iters: int = 3,
) -> np.ndarray:
    """Compute LOWESS smoothed values.

    Parameters
    ----------
    x : np.ndarray
        Independent variable (times).
    y : np.ndarray
        Dependent variable (flux values).
    frac : float
        Fraction of data used in each local regression.
    iters : int
        Number of robust iteration passes.

    Returns
    -------
    np.ndarray
        Smoothed values.
    """
    n = len(x)
    if n < 3:
        return y.copy()

    # Sort by x
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    # Window size
    window = max(3, int(np.ceil(frac * n)))

    # Initial weights
    weights = np.ones(n)

    for _ in range(iters):
        smoothed = np.zeros(n)
        for i in range(n):
            # Find nearest neighbors
            dists = np.abs(x_sorted - x_sorted[i])
            nearest = np.argsort(dists)[:window]

            # Local weights based on distance
            max_dist = dists[nearest[-1]]
            if max_dist == 0:
                local_w = np.ones(window)
            else:
                u = dists[nearest] / max_dist
                local_w = (1 - u ** 3) ** 3

            # Combine with robustness weights
            combined_w = local_w * weights[nearest]

            # Weighted linear regression
            x_local = x_sorted[nearest]
            y_local = y_sorted[nearest]

            # Center x for numerical stability
            x_centered = x_local - x_local.mean()
            x_mean = x_local.mean()

            # Solve weighted least squares for y = a + b*(x - x_mean)
            Sxx = np.sum(combined_w * x_centered ** 2)
            Sxy = np.sum(combined_w * x_centered * y_local)
            Sy = np.sum(combined_w * y_local)
            Sw = np.sum(combined_w)

            if Sxx == 0:
                smoothed[i] = Sy / Sw if Sw > 0 else y_sorted[i]
            else:
                b = Sxy / Sxx
                a = (Sy - b * np.sum(combined_w * x_centered)) / Sw
                smoothed[i] = a + b * (x_sorted[i] - x_mean)

        # Update robustness weights
        residuals = y_sorted - smoothed
        sigma = 1.4826 * np.median(np.abs(residuals))
        if sigma > 0:
            u = np.abs(residuals) / sigma
            weights = np.where(u < 1.0, (1 - u ** 2) ** 2, 0.0)
        else:
            weights = np.ones(n)

    # Unsort
    result = np.zeros(n)
    result[order] = smoothed
    return result


def _savitzky_golay(
    y: np.ndarray,
    window_length: int = 11,
    poly_order: int = 3,
) -> np.ndarray:
    """Apply Savitzky-Golay filter.

    Parameters
    ----------
    y : np.ndarray
        Input signal.
    window_length : int
        Length of the filter window (must be odd).
    poly_order : int
        Order of the polynomial used to fit samples.

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    n = len(y)
    if n < window_length:
        window_length = n if n % 2 == 1 else n - 1
    if window_length < 3:
        return y.copy()
    if poly_order >= window_length:
        poly_order = window_length - 1

    # Build the convolution matrix
    half = window_length // 2
    m = np.arange(-half, half + 1)
    # Vandermonde-like matrix for polynomial coefficients
    A = np.vander(m, poly_order + 1, increasing=True)
    # We want the coefficient for the constant term (index 0)
    # Solve (A^T A)^{-1} A^T e_0 where e_0 is the first basis vector
    try:
        coeffs = np.linalg.lstsq(A, np.eye(A.shape[1])[:, 0], rcond=None)[0]
    except np.linalg.LinAlgError:
        return y.copy()

    # Apply convolution
    padded = np.pad(y, half, mode="edge")
    result = np.convolve(padded, coeffs[::-1], mode="valid")
    return result[:n]


def _polynomial_trend(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 3,
) -> np.ndarray:
    """Fit a polynomial trend and return the trend values.

    Parameters
    ----------
    x : np.ndarray
        Independent variable.
    y : np.ndarray
        Dependent variable.
    degree : int
        Polynomial degree.

    Returns
    -------
    np.ndarray
        Polynomial trend evaluated at x.
    """
    n = len(x)
    if n <= degree:
        degree = max(1, n - 1)
    try:
        coeffs = np.polyfit(x, y, degree)
        trend = np.polyval(coeffs, x)
    except (np.linalg.LinAlgError, ValueError):
        trend = np.full(n, np.mean(y))
    return trend


def detrend_light_curve(
    times: List[float],
    fluxes: List[float],
    method: DetrendingMethod = DetrendingMethod.LOWESS,
    **kwargs,
) -> DetrendResult:
    """Detrend a light curve.

    Parameters
    ----------
    times : List[float]
        Time stamps of observations.
    fluxes : List[float]
        Flux measurements.
    method : DetrendingMethod
        Detrending algorithm to use.
    **kwargs :
        Additional method-specific parameters:
        - LOWESS: frac (float, default 0.3), iters (int, default 3)
        - SAVITZKY_GOLAY: window_length (int, default 11), poly_order (int, default 3)
        - POLYNOMIAL: degree (int, default 3)

    Returns
    -------
    DetrendResult
        Object containing original, detrended, and trend arrays.
    """
    t = np.asarray(times, dtype=np.float64)
    f = np.asarray(fluxes, dtype=np.float64)

    if len(t) != len(f):
        raise ValueError("times and fluxes must have the same length")
    if len(t) == 0:
        return DetrendResult(
            original=[], detrended=[], trend=[], method=method.value, n_points=0
        )

    # Remove NaN/inf
    mask = np.isfinite(t) & np.isfinite(f)
    if not np.all(mask):
        t = t[mask]
        f = f[mask]

    if len(t) < 3:
        trend = np.full(len(t), np.mean(f) if len(f) > 0 else 0.0)
        detrended = f - trend
    else:
        if method == DetrendingMethod.LOWESS:
            frac = kwargs.get("frac", 0.3)
            iters = kwargs.get("iters", 3)
            trend = _lowess(t, f, frac=frac, iters=iters)
        elif method == DetrendingMethod.SAVITZKY_GOLAY:
            window_length = kwargs.get("window_length", 11)
            poly_order = kwargs.get("poly_order", 3)
            trend = _savitzky_golay(f, window_length=window_length, poly_order=poly_order)
        elif method == DetrendingMethod.POLYNOMIAL:
            degree = kwargs.get("degree", 3)
            trend = _polynomial_trend(t, f, degree=degree)
        else:
            raise ValueError(f"Unsupported method: {method}")

        detrended = f - trend

    return DetrendResult(
        original=[float(v) for v in f],
        detrended=[float(v) for v in detrended],
        trend=[float(v) for v in trend],
        method=method.value,
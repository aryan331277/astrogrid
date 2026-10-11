"""PSF Fitting Engine for AstroGrid.

Models the Point Spread Function (PSF) as a convolution of the intrinsic
source profile with the instrumental and atmospheric response. Supports
Gaussian, Mixture of Gaussians (MoG), and Kron profiles.
"""

import numpy as np
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from scipy.optimize import least_squares
from scipy.ndimage import convolve


class PSFModelType(str, Enum):
    GAUSSIAN = "gaussian"
    MOG = "mog"
    KRON = "kron"


class PSFFitResult(BaseModel):
    """Result of a PSF fitting operation."""
    model_type: PSFModelType
    parameters: Dict[str, float]
    chi_squared: float
    dof: int
    reduced_chi_squared: float
    converged: bool
    n_iterations: int
    residuals: Optional[List[float]] = None


class PSFFittingEngine:
    """Engine for fitting PSF models to observed image data.

    The observed image intensity I_obs(x, y) is modeled as the convolution
    of the intrinsic source profile with the instrumental/atmospheric PSF.
    """

    def __init__(self, model_type: PSFModelType = PSFModelType.GAUSSIAN):
        self.model_type = model_type
        self._last_result: Optional[PSFFitResult] = None

    def _gaussian_psf(self, x: np.ndarray, y: np.ndarray,
                      amplitude: float, x0: float, y0: float,
                      sigma: float, theta: float) -> np.ndarray:
        """Evaluate a 2D Gaussian PSF."""
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        dx = x - x0
        dy = y - y0
        x_rot = cos_t * dx + sin_t * dy
        y_rot = -sin_t * dx + cos_t * dy
        exponent = -(x_rot ** 2 + y_rot ** 2) / (2.0 * sigma ** 2)
        return amplitude * np.exp(exponent)

    def _mog_psf(self, x: np.ndarray, y: np.ndarray,
                 params: List[float]) -> np.ndarray:
        """Evaluate a Mixture of Gaussians PSF."""
        n_components = len(params) // 5
        result = np.zeros_like(x, dtype=np.float64)
        for i in range(n_components):
            base = i * 5
            amp = params[base]
            x0 = params[base + 1]
            y0 = params[base + 2]
            sig = max(params[base + 3], 1e-6)
            theta = params[base + 4]
            result += self._gaussian_psf(x, y, amp, x0, y0, sig, theta)
        return result

    def _kron_psf(self, x: np.ndarray, y: np.ndarray,
                  amplitude: float, x0: float, y0: float,
                  sigma: float, beta: float) -> np.ndarray:
        """Evaluate a Kron PSF profile: I(r) = A * (1 + beta * r^2)^(-4)."""
        dx = x - x0
        dy = y - y0
        r2 = dx ** 2 + dy ** 2
        denom = (1.0 + beta * r2) ** 4
        return amplitude / denom

    def _build_model(self, x: np.ndarray, y: np.ndarray,
                     params: np.ndarray) -> np.ndarray:
        """Build the PSF model given parameters."""
        if self.model_type == PSFModelType.GAUSSIAN:
            amp, x0, y0, sigma, theta = params
            sigma = max(abs(sigma), 1e-6)
            return self._gaussian_psf(x, y, amp, x0, y0, sigma, theta)
        elif self.model_type == PSFModelType.MOG:
            return self._mog_psf(x, y, params.tolist())
        elif self.model_type == PSFModelType.KRON:
            amp, x0, y0, sigma, beta = params
            beta = max(abs(beta), 1e-6)
            return self._kron_psf(x, y, amp, x0, y0, sigma, beta)
        else:
            raise ValueError(f"Unsupported PSF model type: {self.model_type}")

    def _residuals(self, params: np.ndarray, x: np.ndarray, y: np.ndarray,
                   observed: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute weighted residuals for least-squares fitting."""
        model = self._build_model(x, y, params)
        diff = observed - model
        if weights is not None:
            diff = diff * weights
        return diff.ravel()

    def fit(self, observed: np.ndarray, initial_params: Optional[np.ndarray] = None,
            weights: Optional[np.ndarray] = None, max_iter: int = 200) -> PSFFitResult:
        """Fit the PSF model to observed image data."""
        ny, nx = observed.shape
        y_coords, x_coords = np.mgrid[0:ny, 0:nx]

        if weights is None:
            weights = np.ones_like(observed)

        if initial_params is None:
            initial_params = self._default_initial_params(observed, x_coords, y_coords)

        method = 'lm' if len(initial_params) <= 10 else 'trf'
        result = least_squares(
            self._residuals,
            initial_params,
            args=(x_coords, y_coords, observed, weights),
            max_nfev=max_iter,
            method=method
        )

        fitted_params = result.x
        model = self._build_model(x_coords, y_coords, fitted_params)
        residuals = (observed - model) * weights
        chi_squared = float(np.sum(residuals ** 2))
        n_params = len(fitted_params)
        dof = max(observed.size - n_params, 1)
        reduced_chi = chi_squared / dof

        param_names = self._get_param_names(n_params)
        param_dict = {param_names[i]: float(fitted_params[i]) for i in range(n_params)}

        fit_result = PSFFitResult(
            model_type=self.model_type,
            parameters=param_dict,
            chi_squared=chi_squared,
            dof=dof,
            reduced_chi_squared=reduced_chi,
            converged=bool(result.success),
            n_iterations=int(result.nfev),
            residuals=residuals.ravel().tolist()
        )
        self._last_result = fit_result
        return fit_result

    def _default_initial_params(self, observed: np.ndarray,
                                x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        """Generate default initial parameter estimates from observed data."""
        peak_idx = np.unravel_index(np.argmax(observed), observed.shape)
        x0 = float(peak_idx[1])
        y0 = float(peak_idx[0])
        amp = float(observed[peak_idx])

        total = np.sum(observed)
        if total > 0:
            weighted_x = np.sum(observed * x_coords) / total
            weighted_y = np.sum(observed * y_coords) / total
            var_x = np.sum(observed * (x_coords - weighted_x) ** 2) / total
            var_y = np.sum(observed * (y_coords - weighted_y) ** 2) / total
            sigma = max(np.sqrt((var_x + var_y) / 2.0), 1.0)
        else:
            sigma = 2.0

        if self.model_type == PSFModelType.GAUSSIAN:
            return np.array([amp, x0, y0, sigma, 0.0])
        elif self.model_type == PSFModelType.MOG:
            return np.array([amp * 0.8, x0, y0, sigma, 0.0,
                             amp * 0.2, x0, y0, sigma * 1.5, 0.0])
        elif self.model_type == PSFModelType.KRON:
            beta = 1.0 / (2.0 * sigma ** 2)
            return np.array([amp, x0, y0, sigma, beta])
        else:
            raise ValueError(f"Unsupported PSF model type: {self.model_type}")

    def _get_param_names(self, n_params: int) -> List[str]:
        """Get parameter names for the current model type."""
        if self.model_type == PSFModelType.GAUSSIAN:
            return ["amplitude", "x0", "y0", "sigma", "theta"]
        elif self.model_type == PSFModelType.MOG:
            names = []
            n_comp = n_params // 5
            for i in range(n_comp):
                names.extend([f"amp_{i+1}", f"x0_{i+1}", f"y0_{i+1}",
                              f"sigma_{i+1}", f"theta_{i+1}"])
            return names
        elif self.model_type == PSFModelType.KRON:
            return ["amplitude", "x0", "y0", "sigma", "beta"]
        else:
            return [f"param_{i}" for i in range(n_params)]
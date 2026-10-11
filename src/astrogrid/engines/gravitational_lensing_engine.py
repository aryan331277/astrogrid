"""
Gravitational Lensing Engine for AstroGrid.

Computes gravitational lensing effects including deflection angles,
magnification, and image positions for point-mass and extended mass
distributions.
"""

import numpy as np
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum


class LensModel(str, Enum):
    """Supported gravitational lens models."""
    POINT_MASS = "point_mass"
    SIS = "sis"
    NFW = "nfw"
    ELLIPTICAL = "elliptical"


class LensParameters(BaseModel):
    """Parameters defining a gravitational lens."""
    model: LensModel = Field(default=LensModel.POINT_MASS)
    mass: float = Field(default=1.0)
    velocity_dispersion: float = Field(default=0.0)
    einstein_radius: float = Field(default=1.0)
    ellipticity: float = Field(default=0.0)
    position_angle: float = Field(default=0.0)
    center: Tuple[float, float] = Field(default=(0.0, 0.0))


class LensingResult(BaseModel):
    """Result of a gravitational lensing calculation."""
    image_positions: List[Tuple[float, float]] = Field(default_factory=list)
    magnifications: List[float] = Field(default_factory=list)
    deflection_angles: List[Tuple[float, float]] = Field(default_factory=list)
    total_magnification: float = Field(default=0.0)
    critical_curves: Optional[List[Tuple[float, float]]] = None
    caustics: Optional[List[Tuple[float, float]]] = None


class GravitationalLensingEngine:
    """
    Computational engine for gravitational lensing calculations.
    """

    def __init__(self, lens: Optional[LensParameters] = None):
        self.lens = lens or LensParameters()
        self._einstein_radius = self.lens.einstein_radius

    def set_lens(self, lens: LensParameters) -> None:
        """Set the lens model parameters."""
        self.lens = lens
        self._einstein_radius = lens.einstein_radius

    def deflection_angle(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute the deflection angle at a given position in the image plane.

        Args:
            theta: Array of shape (N, 2) with positions in arcseconds.

        Returns:
            Array of shape (N, 2) with deflection angles in arcseconds.
        """
        theta = np.atleast_2d(theta)
        cx, cy = self.lens.center
        dx = theta[:, 0] - cx
        dy = theta[:, 1] - cy
        r = np.sqrt(dx**2 + dy**2)
        r_safe = np.where(r > 1e-10, r, 1e-10)

        if self.lens.model == LensModel.POINT_MASS:
            alpha = self._einstein_radius**2 / r_safe
            ax = alpha * dx / r_safe
            ay = alpha * dy / r_safe

        elif self.lens.model == LensModel.SIS:
            alpha = self._einstein_radius / r_safe
            ax = alpha * dx / r_safe
            ay = alpha * dy / r_safe

        elif self.lens.model == LensModel.NFW:
            x = r_safe / self._einstein_radius
            psi = np.log(1 + x) - x / (1 + x)
            alpha = self._einstein_radius / r_safe * psi
            ax = alpha * dx / r_safe
            ay = alpha * dy / r_safe

        elif self.lens.model == LensModel.ELLIPTICAL:
            pa = np.radians(self.lens.position_angle)
            e = self.lens.ellipticity
            cos_pa = np.cos(pa)
            sin_pa = np.sin(pa)
            x_rot = dx * cos_pa + dy * sin_pa
            y_rot = -dx * sin_pa + dy * cos_pa
            r_rot = np.sqrt(x_rot**2 + y_rot**2)
            r_rot_safe = np.where(r_rot > 1e-10, r_rot, 1e-10)
            alpha_rot = self._einstein_radius**2 / r_rot_safe
            ax_rot = alpha_rot * x_rot / r_rot_safe
            ay_rot = alpha_rot * y_rot / r_rot_safe
            ax_rot *= (1 + e)
            ay_rot *= (1 - e)
            ax = ax_rot * cos_pa - ay_rot * sin_pa
            ay = ax_rot * sin_pa + ay_rot * cos_pa

        else:
            raise ValueError(f"Unsupported lens model: {self.lens.model}")

        return np.column_stack([ax, ay])

    def lens_equation(self, beta: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute the residual of the lens equation: beta - theta + alpha(theta).

        Args:
            beta: Source position, shape (2,) or (N, 2).
            theta: Image position, shape (2,) or (N, 2).

        Returns:
            Residual array, same shape as theta.
        """
        beta = np.atleast_2d(beta)
        theta = np.atleast_2d(theta)
        alpha = self.deflection_angle(theta)
        return beta - theta + alpha

    def find_images(self, beta: np.ndarray, n_grid: int = 100,
                    search_radius: float = 5.0) -> List[Tuple[float, float]]:
        """
        Find image positions for a given source position using grid search + Newton iteration.

        Args:
            beta: Source position (beta_1, beta_2) in arcseconds.
            n_grid: Number of grid points per dimension for initial search.
            search_radius: Radius in arcseconds to search for images.

        Returns:
            List of image positions (theta_1, theta_2).
        """
        beta = np.asarray(beta, dtype=float)
        images: List[Tuple[float, float]] = []

        theta_range = np.linspace(-search_radius, search_radius, n_grid)
        theta_grid = np.array([[x, y] for x in theta_range for y in theta_range])

        residuals = self.lens_equation(beta, theta_grid)
        residual_magnitudes = np.linalg.norm(residuals, axis=1)

        threshold = np.percentile(residual_magnitudes, 5)
        candidate_indices = np.where(residual_magnitudes < threshold)[0]

        used: set = set()
        for idx in candidate_indices:
            pos = theta_grid[idx]
            is_new = True
            for j in used:
                if np.linalg.norm(pos - theta_grid[j]) < 0.5:
                    is_new = False
                    break
            if is_new:
                used.add(idx)
                refined = self._newton_refine(beta, pos, max_iter=50, tol=1e-10)
                if refined is not None:
                    res = self.lens_equation(beta, refined)
                    if np.linalg.norm(res) < 1e-6:
                        is_dup = False
                        for img in images:
                            if np.linalg.norm(np.array(img) - refined) < 0.01:
                                is_dup = True
                                break
                        if not is_dup:
                            images.append((float(refined[0]), float(refined[1])))

        return images

    def _newton_refine(self, beta: np.ndarray, theta0: np.ndarray,
                       max_iter: int = 50, tol: float = 1e-10) -> Optional[np.ndarray]:
        """
        Newton-Raphson refinement of an image position.

        Args:
            beta: Source position.
            theta0: Initial guess for image position.
            max_iter: Maximum iterations.
            tol: Convergence tolerance.

        Returns:
            Refined image position or None if not converged.
        """
        theta = theta0.copy()
        for _ in range(max_iter):
            f = self.lens_equation(beta, theta)
            h = 1e-7
            jacobian = np.zeros((2, 2))
            for j in range(2):
                theta_plus = theta.copy()
                theta_plus[j] += h
                f_plus = self.lens_equation(beta, theta_plus)
                jacobian[:, j] = (f_plus - f) / h
            det = jacobian[0, 0] * jacobian[1, 1] - jacobian[0, 1] * jacobian[1, 0]
            if abs(det) < 1e-15:
                return None
            inv_det = 1.0 / det
            dx = (jacobian[1, 1] * f[0] - jacobian[0, 1] * f[1]) * inv_det
            dy = (-jacobian[1, 0] * f[0] + jacobian[0, 0] * f[1]) * inv_det
            theta[0] -= dx
            theta[1] -= dy
            if abs(dx) + abs(dy) < tol:
                return theta
        return None

    def magnification(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute the magnification at given image positions.

        Args:
            theta: Array of shape (N, 2) with image positions.

        Returns:
            Array of shape (N,) with magnification values.
        """
        theta = np.atleast_2d(theta)
        h = 1e-7
"""Spectral condensation of SurfaceRZFourier surfaces.

Derives a new set of Fourier coefficients that represent the same surface geometry
but with reduced spectral width, by minimizing the spectral width subject to a
constraint that the surface points do not move beyond a given tolerance.
"""

import dataclasses
import logging
from collections.abc import Callable

import jaxtyping as jt
import numpy as np
from scipy import optimize

from constellaration.geometry import surface_rz_fourier, surface_utils

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class SpectralCondensationSettings:
    """Settings for spectral condensation of a SurfaceRZFourier surface."""

    p: int = 4
    """Poloidal mode exponent in the spectral width denominator."""

    q: int = 1
    """Poloidal mode exponent in the spectral width numerator."""

    normalize: bool = True
    """Whether to normalize the spectral width."""

    maximum_normal_displacement: float = float(np.finfo(float).eps)
    """Maximum allowed normal displacement of the condensed surface from the
    original."""

    energy_scale: float = 3.5
    """Energy-based spectrum scaling factor for preconditioning."""

    bounds: float = 1.0
    """Symmetric bounds on the normalized optimization variables."""

    n_restarts: int = 0
    """Number of optimization restarts from the previous best solution."""


def spectrally_condense_surface(
    surface: surface_rz_fourier.SurfaceRZFourier,
    settings: SpectralCondensationSettings = SpectralCondensationSettings(),
) -> surface_rz_fourier.SurfaceRZFourier:
    r"""Spectrally condense a SurfaceRZFourier surface.

    Minimizes the spectral width of the Fourier representation while constraining
    the discretized surface points to remain within a given normal displacement
    tolerance of the original surface.

    The spectral width is defined as:

    .. math::
        W = \sum_{m,n} m^{p+q} (r_{m,n}^2 + z_{m,n}^2)

    optionally normalized by :math:`\sum_{m,n} m^p (r_{m,n}^2 + z_{m,n}^2)`.

    Args:
        surface: The surface to condense.
        settings: Optimization settings.

    Returns:
        A new SurfaceRZFourier with reduced spectral width.

    Raises:
        NotImplementedError: If the surface is not stellarator symmetric.
    """
    if not surface.is_stellarator_symmetric:
        raise NotImplementedError(
            "Non stellarator symmetric surfaces are not supported yet."
        )

    # Determine free variables --------------------------------------------------
    ntor = surface.max_toroidal_mode

    free_r = np.ones(surface.r_cos.shape, dtype=bool)
    free_r[0, :ntor] = False  # r_cos(m=0, n<0) fixed at 0
    free_r[0, ntor] = False  # r_cos(0,0) - major radius

    free_z = np.ones(surface.z_sin.shape, dtype=bool)
    free_z[0, : ntor + 1] = False  # z_sin(m=0, n<=0) fixed at 0

    n_free_r = int(free_r.sum())

    # Energy spectrum scaling for preconditioning -------------------------------
    # Uses the same exp-based formula as the reference implementation to ensure
    # the energy_scale default (3.5) works well.
    energy = surface.poloidal_modes**2 + surface.toroidal_modes**2
    scale = np.exp(-np.sqrt(energy.astype(np.float64)) / settings.energy_scale)

    r_cos_0 = np.array(surface.r_cos, dtype=np.float64)
    z_sin_0 = np.array(surface.z_sin, dtype=np.float64)

    # Pre-compute mode weights for the analytical gradient
    m = surface.poloidal_modes.astype(np.float64)
    w_num = m ** (settings.p + settings.q)  # numerator weight
    w_den = m**settings.p  # denominator weight

    def _x_to_coeffs(
        x: jt.Float[np.ndarray, " n"],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Optimization vector -> (r_cos, z_sin) arrays."""
        r_cos = r_cos_0.copy()
        z_sin = z_sin_0.copy()
        r_cos[free_r] += x[:n_free_r] * scale[free_r]
        z_sin[free_z] += x[n_free_r:] * scale[free_z]
        return r_cos, z_sin

    def _x_to_surface(
        x: jt.Float[np.ndarray, " n"],
    ) -> surface_rz_fourier.SurfaceRZFourier:
        r_cos, z_sin = _x_to_coeffs(x)
        return surface.model_copy(update=dict(r_cos=r_cos, z_sin=z_sin))

    def _objective_and_grad(
        x: jt.Float[np.ndarray, " n"],
    ) -> tuple[float, np.ndarray]:
        """Spectral width value and gradient w.r.t. x."""
        r_cos, z_sin = _x_to_coeffs(x)
        coeff_sq = r_cos**2 + z_sin**2

        numerator = float(np.sum(w_num * coeff_sq))
        denominator = float(np.sum(w_den * coeff_sq))

        if settings.normalize and denominator != 0.0:
            value = numerator / denominator
            # dW/dc = 2*c*(w_num*D - w_den*N) / D^2 = 2*c*(w_num - w_den*W) / D
            common = 2.0 * (w_num - w_den * value) / denominator
        else:
            value = numerator
            common = 2.0 * w_num

        grad_r_full = common * r_cos
        grad_z_full = common * z_sin
        # Chain rule: dx -> coefficients includes the scale factor
        grad = np.concatenate(
            [grad_r_full[free_r] * scale[free_r], grad_z_full[free_z] * scale[free_z]]
        )
        return value, grad

    # Hessian-based preconditioning ---------------------------------------------
    x0 = np.zeros(n_free_r + int(free_z.sum()))
    hessian = _approx_hessian(lambda x: _objective_and_grad(x)[0], x0)
    eigvals, _ = np.linalg.eigh(hessian)
    with np.errstate(divide="ignore"):
        preconditioner = np.where(
            np.abs(eigvals) > 1e-30, np.sqrt(1.0 / np.abs(eigvals)), 1.0
        )
    logger.debug("Preconditioner: %s", preconditioner)

    def preconditioned_value_and_grad(
        y: jt.Float[np.ndarray, " n"],
    ) -> tuple[float, np.ndarray]:
        """Preconditioned objective returning (value, gradient)."""
        val, grad_x = _objective_and_grad(preconditioner * y)
        return val, grad_x * preconditioner

    # Constraint: normal displacement -------------------------------------------
    (
        n_pol,
        n_tor,
    ) = surface_utils.n_poloidal_toroidal_points_to_satisfy_nyquist_criterion(
        surface.n_poloidal_modes, surface.max_toroidal_mode
    )
    if surface.max_toroidal_mode == 0:
        n_tor = 1

    _normal_distance_fn = _create_normal_distance_constraint(
        target_surface=surface,
        n_poloidal_points=n_pol,
        n_toroidal_points=n_tor,
    )

    def preconditioned_constraint(y: jt.Float[np.ndarray, " n"]) -> np.ndarray:
        return _normal_distance_fn(_x_to_surface(preconditioner * y))

    constraint = optimize.NonlinearConstraint(
        fun=preconditioned_constraint,
        lb=-settings.maximum_normal_displacement,
        ub=settings.maximum_normal_displacement,
    )

    # Optimization --------------------------------------------------------------
    def _run_minimize(x0_arg: np.ndarray) -> optimize.OptimizeResult:
        res = optimize.minimize(
            fun=preconditioned_value_and_grad,
            x0=x0_arg,
            method="trust-constr",
            jac=True,
            constraints=[constraint],
            options={"maxiter": 1000},
            bounds=optimize.Bounds(
                lb=float(-settings.bounds),
                ub=float(settings.bounds),
            ),
        )
        logger.debug("Optimizer result: %s", res)
        if res.status not in (0, 1):
            logger.warning("Optimization did not converge: %s", res.message)
        return res

    logger.info("Spectral condensation, initial optimization.")
    result = _run_minimize(np.zeros_like(x0))
    for restart in range(settings.n_restarts):
        logger.info("Spectral condensation, restart %d.", restart + 1)
        result = _run_minimize(result.x.copy())

    violation = np.max(np.abs(preconditioned_constraint(result.x)))
    logger.debug("Constraint violation inf-norm: %s", violation)

    return _x_to_surface(preconditioner * result.x)


def _create_normal_distance_constraint(
    target_surface: surface_rz_fourier.SurfaceRZFourier,
    n_poloidal_points: int,
    n_toroidal_points: int,
) -> Callable[[surface_rz_fourier.SurfaceRZFourier], np.ndarray]:
    """Build a constraint function that measures normal distance to *target_surface*.

    The reference surface data (xyz, unit normals) is pre-computed once so that
    only the comparison surface needs to be evaluated at each optimization step.
    """
    phi_upper_bound = (
        2.0
        * np.pi
        / target_surface.n_field_periods
        / (1.0 + int(target_surface.is_stellarator_symmetric))
    )
    theta_phi = surface_utils.make_theta_phi_grid(
        n_theta=n_poloidal_points,
        n_phi=n_toroidal_points,
        phi_upper_bound=phi_upper_bound,
    )

    ref_xyz = np.asarray(
        surface_rz_fourier.evaluate_points_xyz(target_surface, theta_phi)
    )
    ref_normal = np.asarray(
        surface_rz_fourier.evaluate_unit_normal(target_surface, theta_phi)
    )

    def constraint_fn(
        comparison_surface: surface_rz_fourier.SurfaceRZFourier,
    ) -> np.ndarray:
        new_xyz = np.asarray(
            surface_rz_fourier.evaluate_points_xyz(comparison_surface, theta_phi)
        )
        n_points = n_poloidal_points * n_toroidal_points
        distances = np.empty(n_points)
        idx = 0
        for j in range(n_toroidal_points):
            ref_at_phi = ref_xyz[:, j, :]
            normals_at_phi = ref_normal[:, j, :]
            new_at_phi = new_xyz[:, j, :]
            diff = new_at_phi[:, np.newaxis, :] - ref_at_phi[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=-1)
            closest = np.argmin(dist_sq, axis=1)
            delta = new_at_phi - ref_at_phi[closest]
            normal_closest = normals_at_phi[closest]
            distances[idx : idx + n_poloidal_points] = np.sum(
                delta * normal_closest, axis=-1
            )
            idx += n_poloidal_points
        return distances

    return constraint_fn


def _approx_hessian(
    fun: Callable[[np.ndarray], float],
    x0: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Approximate the Hessian of *fun* at *x0* via finite differences."""
    n = len(x0)
    hessian = np.zeros((n, n))
    f0 = fun(x0)
    fi = np.empty(n)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = eps
        fi[i] = fun(x0 + ei)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = eps
        for j in range(i, n):
            ej = np.zeros(n)
            ej[j] = eps
            fij = fun(x0 + ei + ej)
            hessian[i, j] = (fij - fi[i] - fi[j] + f0) / (eps * eps)
            hessian[j, i] = hessian[i, j]
    return hessian

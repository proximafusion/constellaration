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
    """Energy-based spectrum scaling factor (unused, kept for API compatibility)."""

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

    # Free variables: exclude ALL m=0 modes.
    # m=0 modes have zero weight in the spectral width objective (m^p = 0 for m=0)
    # and create degenerate Hessian directions.
    free_mask = np.asarray(surface.poloidal_modes > 0)
    n_free_r = int(free_mask.sum())

    r_cos_0 = np.array(surface.r_cos, dtype=np.float64)
    z_sin_0 = np.array(surface.z_sin, dtype=np.float64)

    # x0: initial free coefficients [r_cos_free, z_sin_free]
    x0 = np.concatenate([r_cos_0[free_mask], z_sin_0[free_mask]])

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
        r_cos[free_mask] = x[:n_free_r]
        z_sin[free_mask] = x[n_free_r:]
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
        grad = np.concatenate([grad_r_full[free_mask], grad_z_full[free_mask]])
        return value, grad

    # Jacobi preconditioner from Hessian diagonal ----------------------------------
    # Using the diagonal (not eigenvalues) is critical: eigenvalues are sorted and
    # not aligned with coordinate axes, so applying them as per-variable scaling
    # mixes up directions and can worsen conditioning.
    hessian_diag = _approx_hessian_diagonal(lambda x: _objective_and_grad(x)[0], x0)
    preconditioner = np.where(
        np.abs(hessian_diag) >= 1e-6,
        np.sqrt(1.0 / np.abs(hessian_diag)),
        1e3,
    )

    abs_diag = np.abs(hessian_diag)
    positive_diag = abs_diag[abs_diag > 1e-16]
    if len(positive_diag) > 0:
        kappa = positive_diag.max() / positive_diag.min()
        logger.info(
            "Hessian diagonal range: [%.3e, %.3e], kappa = %.3e",
            positive_diag.min(),
            positive_diag.max(),
            kappa,
        )
    logger.debug("Preconditioner: %s", preconditioner)

    def preconditioned_value_and_grad(
        u: jt.Float[np.ndarray, " n"],
    ) -> tuple[float, np.ndarray]:
        """Preconditioned objective returning (value, gradient).

        Uses the affine map x = P*u + x0, so u=0 corresponds to the original surface.
        """
        x = preconditioner * u + x0
        val, grad_x = _objective_and_grad(x)
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

    def preconditioned_constraint(u: jt.Float[np.ndarray, " n"]) -> np.ndarray:
        return _normal_distance_fn(_x_to_surface(preconditioner * u + x0))

    constraint = optimize.NonlinearConstraint(
        fun=preconditioned_constraint,
        lb=-settings.maximum_normal_displacement,
        ub=settings.maximum_normal_displacement,
    )

    # Optimization --------------------------------------------------------------
    u0 = np.zeros_like(x0)

    def _run_minimize(u0_arg: np.ndarray) -> optimize.OptimizeResult:
        res = optimize.minimize(
            fun=preconditioned_value_and_grad,
            x0=u0_arg,
            method="slsqp",
            jac=True,
            constraints=[constraint],
            options={"maxiter": 1000},
            bounds=optimize.Bounds(
                lb=float(-settings.bounds),
                ub=float(settings.bounds),
            ),
        )
        logger.debug("Optimizer result: %s", res)
        if not res.success:
            logger.warning("Optimization did not converge: %s", res.message)
        return res

    logger.info("Spectral condensation, initial optimization.")
    result = _run_minimize(u0)
    for restart in range(settings.n_restarts):
        logger.info("Spectral condensation, restart %d.", restart + 1)
        result = _run_minimize(result.x.copy())

    violation = np.max(np.abs(preconditioned_constraint(result.x)))
    logger.debug("Constraint violation inf-norm: %s", violation)

    return _x_to_surface(preconditioner * result.x + x0)


def _create_normal_distance_constraint(
    target_surface: surface_rz_fourier.SurfaceRZFourier,
    n_poloidal_points: int,
    n_toroidal_points: int,
) -> Callable[[surface_rz_fourier.SurfaceRZFourier], np.ndarray]:
    """Build a constraint function that measures normal distance to *target_surface*.

    Uses line-segment projection for smooth closest-point correspondence at fixed
    phi, rather than discrete nearest-neighbor (which creates a non-smooth constraint
    landscape with kinks at segment boundaries).

    The reference surface data (xyz, unit normals, line segments) is pre-computed once
    so that only the comparison surface needs to be evaluated at each optimization step.
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

    # Close the theta loop by appending the first row (theta wraps at 2*pi).
    ref_xyz_closed = np.concatenate([ref_xyz, ref_xyz[:1]], axis=0)
    ref_normal_closed = np.concatenate([ref_normal, ref_normal[:1]], axis=0)

    # Pre-compute line segments per phi slice.
    seg_starts = ref_xyz_closed[:-1]  # (n_theta, n_phi, 3)
    seg_ends = ref_xyz_closed[1:]  # (n_theta, n_phi, 3)
    seg_d = seg_ends - seg_starts  # (n_theta, n_phi, 3)
    seg_d_sq = np.sum(seg_d * seg_d, axis=-1)  # (n_theta, n_phi)

    normal_starts = ref_normal_closed[:-1]  # (n_theta, n_phi, 3)
    normal_ends = ref_normal_closed[1:]  # (n_theta, n_phi, 3)

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
            new_at_phi = new_xyz[:, j, :]  # (n_new, 3)

            # Line-segment projection: for each new point, project onto all segments
            # at this phi slice and find the closest.
            s = seg_starts[:, j, :]  # (n_seg, 3)
            d = seg_d[:, j, :]  # (n_seg, 3)
            d_sq = seg_d_sq[:, j]  # (n_seg,)

            # v[i, k, :] = new_at_phi[i] - s[k]
            v = new_at_phi[:, np.newaxis, :] - s[np.newaxis, :, :]  # (n_new, n_seg, 3)

            # Projection parameter t = (v . d) / (d . d), clamped to [0, 1]
            t_num = np.sum(v * d[np.newaxis, :, :], axis=-1)  # (n_new, n_seg)
            safe_d_sq = np.where(d_sq > 1e-30, d_sq, 1.0)
            t = np.clip(t_num / safe_d_sq[np.newaxis, :], 0.0, 1.0)  # (n_new, n_seg)

            # Closest point on each segment
            closest = (
                s[np.newaxis, :, :] + t[:, :, np.newaxis] * d[np.newaxis, :, :]
            )  # (n_new, n_seg, 3)

            # Squared distance to each segment's closest point
            dist_sq = np.sum(
                (new_at_phi[:, np.newaxis, :] - closest) ** 2, axis=-1
            )  # (n_new, n_seg)

            # Best segment per new point
            best_seg = np.argmin(dist_sq, axis=1)  # (n_new,)
            arange = np.arange(n_poloidal_points)
            t_best = t[arange, best_seg]  # (n_new,)

            best_closest = closest[arange, best_seg]  # (n_new, 3)

            # Interpolate unit normal at the closest point
            ns = normal_starts[:, j, :]  # (n_seg, 3)
            ne = normal_ends[:, j, :]  # (n_seg, 3)
            interp_normal = (1.0 - t_best)[:, np.newaxis] * ns[best_seg] + t_best[
                :, np.newaxis
            ] * ne[best_seg]
            norms = np.linalg.norm(interp_normal, axis=-1, keepdims=True)
            interp_normal = interp_normal / np.where(norms > 1e-30, norms, 1.0)

            # Signed normal distance
            delta = new_at_phi - best_closest
            distances[idx : idx + n_poloidal_points] = np.sum(
                delta * interp_normal, axis=-1
            )
            idx += n_poloidal_points
        return distances

    return constraint_fn


def _approx_hessian_diagonal(
    fun: Callable[[np.ndarray], float],
    x0: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Approximate the diagonal of the Hessian of *fun* at *x0* via central finite
    differences.

    Uses 2*n+1 function evaluations (much cheaper than the full O(n^2) Hessian).
    """
    n = len(x0)
    f0 = fun(x0)
    diag = np.empty(n)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = eps
        fp = fun(x0 + ei)
        fm = fun(x0 - ei)
        diag[i] = (fp - 2.0 * f0 + fm) / (eps * eps)
    return diag

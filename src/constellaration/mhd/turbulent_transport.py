from typing import Any, cast

import jaxtyping as jt
import numpy as np
import pydantic
from constellaration.mhd import vmec_utils
from simsopt import mhd


class IdealMHDTurbulentTransportMetricsSettings(
    pydantic.BaseModel, arbitrary_types_allowed=True
):
    """Settings for the turbulent transport metrics of an ideal-MHD equilibrium."""

    normalized_toroidal_flux: jt.Float[np.ndarray, " n_surfaces"]
    """The normalized toroidal flux of the flux surfaces on which the turbulent
    transport metrics are evaluated."""

    n_field_lines: int
    """The number of field lines to use."""

    n_toroidal_points: int
    """The number of points along field lines to use."""


def compute_flux_compression_in_regions_of_bad_curvature(
    equilibrium: vmec_utils.VmecppWOut,
    settings: IdealMHDTurbulentTransportMetricsSettings,
) -> float:
    vmec = vmec_utils.as_simsopt_vmec(equilibrium)

    alpha, theta = _set_up_surface_grid(
        settings=settings,
        n_field_periods=equilibrium.nfp,
        is_stellarator_symmetric=not equilibrium.lasym,
    )

    data = mhd.vmec_fieldlines(
        vs=vmec,
        s=settings.normalized_toroidal_flux,
        alpha=alpha,
        theta1d=theta,
    )
    data = cast(Any, data)  # silence pylance warnings about data

    minor_radius = equilibrium.Aminor_p

    flux_surface_average_weights = (
        np.abs(data.sqrt_g_vmec)
        / np.abs(data.sqrt_g_vmec).mean(axis=(1, 2))[:, np.newaxis, np.newaxis]
    )

    grad_r_dot_grad_r = (
        minor_radius**2
        / (4 * settings.normalized_toroidal_flux[:, None, None])
        * data.grad_s_dot_grad_s
    )

    edge_toroidal_flux = equilibrium.phipf[-1]
    toroidal_flux_sign = np.sign(edge_toroidal_flux)
    curvature_drift_tilde = data.B_cross_kappa_dot_grad_alpha * toroidal_flux_sign

    flux_compression_in_regions_of_bad_curvature = (
        np.heaviside(curvature_drift_tilde, 0) * grad_r_dot_grad_r
    )
    return np.mean(
        flux_compression_in_regions_of_bad_curvature * flux_surface_average_weights,
        axis=(1, 2),
    )


def _set_up_surface_grid(
    n_field_periods: int,
    is_stellarator_symmetric: bool,
    settings: IdealMHDTurbulentTransportMetricsSettings,  # noqa: E501
) -> tuple[
    jt.Float[np.ndarray, " n_field_lines"], jt.Float[np.ndarray, " n_toroidal_points"]
]:
    """Sets up the grid as it is done in STELLOPT.

    See: https://github.com/PrincetonUniversity/STELLOPT/blob/develop/STELLOPTV2/Sources/General/stellopt_txport.f90#L258
    """
    alpha_start = -(2.0 * np.pi) / n_field_periods / (1 + int(is_stellarator_symmetric))
    alpha_end = (2.0 * np.pi) / n_field_periods / (1 + int(is_stellarator_symmetric))
    alpha = np.linspace(alpha_start, alpha_end, settings.n_field_lines, endpoint=False)
    theta = np.linspace(-np.pi, np.pi, settings.n_toroidal_points, endpoint=False)
    return alpha, theta

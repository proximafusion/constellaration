import functools

import jaxtyping as jt
import numpy as np
import pydantic
from constellaration.boozer import boozer
from scipy import interpolate

_MACHINE_PRECISION = float(np.finfo(float).eps)


class QIMetrics(pydantic.BaseModel, arbitrary_types_allowed=True):
    residuals: jt.Float[
        np.ndarray, "n_flux_surfaces n_field_lines n_output_toroidal_points"
    ]


class QISettings(pydantic.BaseModel):
    n_field_lines: int = 75
    """Number of field lines."""

    n_toroidal_points: int = 601
    """Number of toroidal grid points."""

    n_field_contours: int = 401
    """Number of magnetic field levels to sample between the bottom and top of the
    magnetic well."""

    n_output_toroidal_points: int = 2000
    """Number of output toroidal grid points on which the residuals are evaluated."""


def quasi_isodynamicity_residual(
    boozer: boozer.BoozerOutput,
    settings: QISettings,
) -> QIMetrics:
    """Compute the quasi-isodynamicity residual for a given Boozer field following the
    procedure described in:

    Goodman, A. G., et al. "Constructing precisely quasi-isodynamic magnetic fields."
    Journal of Plasma Physics 89.5 (2023): 905890504.
    https://doi.org/10.1017/S002237782300065X

    Args:
        boozer: The Boozer equilibrium object containing the magnetic field data.

    Note: This function is a refactored version for improved readability of
    https://github.com/rogeriojorge/single_stage_optimization/blob/8673154/src/qi_functions.py#L10
    """

    n_flux_surfaces = boozer.n_boozer_flux_surfaces
    nalpha = settings.n_field_lines
    nphi = settings.n_toroidal_points
    nBj = settings.n_field_contours
    nphi_out = settings.n_output_toroidal_points

    out = np.zeros((n_flux_surfaces, nalpha, nphi_out))

    n_field_periods = boozer.n_field_periods

    # Determine toroidal domain based on the location of the magnetic field maxima
    if boozer.bmnc[1, 1] < 0:
        phi_start = np.pi / n_field_periods
    else:
        phi_start = 0.0
    phi_end = phi_start + 2 * np.pi / n_field_periods

    phis = np.linspace(phi_start, phi_end, nphi, dtype=np.float64)
    phis2D = np.tile(phis, (nalpha, 1)).T

    Bjs = np.linspace(0, 1, nBj)

    output_phi = np.linspace(phi_start, phi_end, nphi_out)

    xm_b = boozer.xm_b
    xn_b = boozer.xn_b

    iota = interpolate.UnivariateSpline(
        boozer.normalized_toroidal_flux, boozer.iota, k=1, s=0
    )(boozer.boozer_normalized_toroidal_flux)

    # Loop over flux surfaces
    for js in range(n_flux_surfaces):
        # Initialize the poloidal angle
        theta_start = -iota[js] * phi_start  # type: ignore
        theta_end = theta_start + 2 * np.pi
        thetas2D = (
            np.tile(np.linspace(theta_start, theta_end, nalpha), (nphi, 1))
            + iota[js] * phis2D
        )

        # Compute the magnetic field strength
        angle = (
            xm_b[:, None, None] * thetas2D[None, ...]
            - xn_b[:, None, None] * phis2D[None, ...]
        )
        modb = np.sum(np.cos(angle) * boozer.bmnc_b[:, js][:, None, None], axis=0)

        # Normalize the magnetic field strength
        modb = (modb - np.min(modb)) / (np.max(modb) - np.min(modb))

        # Initialize arrays for SQUASH, STRETCH, and SHUFFLE operations
        modb_squashed_stretched = np.zeros((nalpha, nphi))
        bounce_distances = np.zeros((nalpha, nBj))
        phi_bounce_points = np.zeros((nalpha, 2 * nBj - 1))
        phi_shuffled = np.zeros((nalpha, 2 * nBj - 1))
        fieldline_weights = np.zeros(nalpha)

        # Loop over field lines (alphas)
        for ialpha in range(nalpha):
            modb_along_field_line = modb[:, ialpha]

            # Find the index of the minimum of B along this fieldline
            modb_min_index = np.argmin(modb_along_field_line)
            if modb_min_index == 0 or modb_min_index == len(modb_along_field_line) - 1:
                raise ValueError("Minimum of B is at the boundary.")

            # SQUASH the magnetic well
            modb_left_side = _squash_left_side(
                modb_along_field_line[: modb_min_index + 1]
            )
            modb_right_side = _squash_right_side(modb_along_field_line[modb_min_index:])

            # STRETCH the squashed well
            modb_left_side = _stretch_left_side(modb_left_side)
            modb_right_side = _stretch_right_side(modb_right_side)

            # Combine the left and right sides of the magnetic well
            modb_squashed_stretched[ialpha, :] = np.concatenate(
                (modb_left_side[:-1], modb_right_side)
            )

            # Compute weights for each field line.
            # The weight is the inverse of the integral of the squared difference
            # between the original and squashed and stretched B along the field line.
            weight_function = interpolate.UnivariateSpline(
                phis,
                (modb_along_field_line - modb_squashed_stretched[ialpha, :]) ** 2,
            )
            fieldline_weights[ialpha] = (phi_end - phi_start) / float(
                weight_function.integral(phi_start, phi_end)  # type: ignore
            )

            # Find bounce distances and the toroidal locations of the bounce points
            for j in range(nBj):
                shuffled_phi_left, shuffled_phi_right = _find_bounce_points(
                    phi=phis,
                    modb=modb_squashed_stretched[ialpha, :],
                    modb_star=Bjs[j],
                    modb_max_on_flux_surface=1.0,
                    modb_min_on_flux_surface=0.0,
                )
                bounce_distances[ialpha, j] = shuffled_phi_right - shuffled_phi_left
                phi_bounce_points[ialpha, nBj - j - 1] = shuffled_phi_left
                phi_bounce_points[ialpha, nBj + j - 1] = shuffled_phi_right

        # Normalize weights and compute mean bounce distances
        fieldline_weights /= np.sum(fieldline_weights)

        # SHUFFLE the well
        mean_bounce_distances = np.sum(
            bounce_distances * fieldline_weights[:, np.newaxis], axis=0
        )

        # qi_modb has shape (2 * nBj - 1,)
        # It represents the magnetic field strength along the shuffled well.
        qi_modb = np.concatenate((np.flip(Bjs), Bjs[1:]))

        field_line_averaged_normalization = 0.0

        for ialpha in range(nalpha):
            delta_bounce = (bounce_distances[ialpha, :] - mean_bounce_distances) / 2.0
            shuffled_phi_left = phi_bounce_points[ialpha, :nBj] + np.flip(delta_bounce)
            shuffled_phi_right = phi_bounce_points[ialpha, nBj - 1 :] - delta_bounce
            phi_shuffled[ialpha, :nBj] = shuffled_phi_left
            phi_shuffled[ialpha, nBj - 1 :] = shuffled_phi_right

            # Interpolate B along original and shuffled phi
            modb_spline = interpolate.UnivariateSpline(phis, modb[:, ialpha], k=1, s=0)
            try:
                modB_shuffled_spline = interpolate.UnivariateSpline(
                    phi_shuffled[ialpha, :], qi_modb, k=1, s=0
                )
            except Exception:
                # Adjust phi_shuffled to ensure monotonicity
                shuffled_phi_left = phi_shuffled[ialpha, :nBj]
                shuffled_phi_right = phi_shuffled[ialpha, nBj - 1 :]
                for il in range(nBj - 1):
                    if shuffled_phi_left[il + 1] - shuffled_phi_left[il] < 0:
                        shuffled_phi_right[-il - 2] += (
                            shuffled_phi_left[il] - shuffled_phi_left[il + 1] + 1e-12
                        )
                        shuffled_phi_left[il + 1] = shuffled_phi_left[il] + 1e-12
                    if shuffled_phi_right[-il - 1] - shuffled_phi_right[-il - 2] < 0:
                        shuffled_phi_left[il + 1] += (
                            shuffled_phi_right[-il - 1]
                            - shuffled_phi_right[-il - 2]
                            - 1e-12
                        )
                        shuffled_phi_right[-il - 2] = (
                            shuffled_phi_right[-il - 1] - 1e-12
                        )
                phi_shuffled[ialpha, :nBj] = shuffled_phi_left
                phi_shuffled[ialpha, nBj - 1 :] = shuffled_phi_right
                modB_shuffled_spline = interpolate.UnivariateSpline(
                    phi_shuffled[ialpha, :], qi_modb, k=1, s=0
                )

            field_line_averaged_normalization += 1 / nalpha
            penalty = np.array(modB_shuffled_spline(output_phi)) - np.array(
                modb_spline(output_phi)
            )

            out[js, ialpha, :] = penalty / np.sqrt(nphi_out)

        out[js] *= field_line_averaged_normalization

    return QIMetrics(residuals=out / np.sqrt(nalpha))


def _squash_left_side(modb: np.ndarray) -> np.ndarray:
    """Squash the left side of the magnetic well."""
    modb = modb.copy()
    modb_max_index = np.argmax(modb)
    modb[:modb_max_index] = modb[modb_max_index]
    return np.minimum.accumulate(modb)


def _squash_right_side(modb: np.ndarray) -> np.ndarray:
    """Squash the right side of the magnetic well."""
    modb = modb.copy()
    index_of_modb_max = np.argmax(modb)
    modb[index_of_modb_max:] = modb[index_of_modb_max]
    return np.minimum.accumulate(modb[::-1])[::-1]


def _stretch_left_side(
    y: np.ndarray, a_min: float = 0.0, a_max: float = 1.0
) -> np.ndarray:
    """Stretch the squashed left side of the magnetic well."""
    return (y - y[-1]) / (y[0] - y[-1]) * (a_max - a_min) + a_min


def _stretch_right_side(
    y: np.ndarray, a_min: float = 0.0, a_max: float = 1.0
) -> np.ndarray:
    """Stretch the squashed right side of the magnetic well."""
    return (y - y[0]) / (y[-1] - y[0]) * (a_max - a_min) + a_min


def _find_bounce_points(
    phi: np.ndarray,
    modb: np.ndarray,
    modb_star: float,
    modb_max_on_flux_surface: float,
    modb_min_on_flux_surface: float,
) -> tuple[float, float]:
    """Find the toroidal location where the magnetic field strength crosses a specified
    value."""
    # Calculate differences between the magnetic field strength and the target value
    modb_diff = modb - modb_star
    # Compute the product of adjacent differences to find sign changes (zero crossings)
    sign_changes = modb_diff[:-1] * modb_diff[1:]
    # Find indices where sign change occurs (crossings)
    crossing_indices = np.where(sign_changes < 0)[0]

    # Handle special cases where modb_star is at or outside the range of modb
    if (
        modb_star - modb_min_on_flux_surface < _MACHINE_PRECISION
        or modb_star < modb_min_on_flux_surface
    ):
        min_index = np.argmin(modb)
        phi_min = phi[min_index]
        return phi_min, phi_min
    elif (
        modb_max_on_flux_surface - modb_star < _MACHINE_PRECISION
        or modb_star > modb_max_on_flux_surface
    ):
        return phi[0], phi[-1]

    # If more than two crossings, select the first and last
    if len(crossing_indices) >= 2:
        crossing_indices = [crossing_indices[0], crossing_indices[-1]]
    else:
        raise ValueError("Not enough crossings found.")

    interp = functools.partial(
        _interpolate_and_find_phi, modb=modb, phi=phi, modb_star=modb_star
    )

    return interp(crossing_indices[0]), interp(crossing_indices[1])


def _interpolate_and_find_phi(
    crossing_index: int, modb: np.ndarray, phi: np.ndarray, modb_star: float
) -> float:
    # Linearly interpolate to find precise crossing points
    dy = modb[crossing_index] - modb[crossing_index + 1]
    dx = phi[crossing_index] - phi[crossing_index + 1]
    m = dy / dx
    b = modb[crossing_index] - m * phi[crossing_index]
    return (modb_star - b) / m if m != 0 else phi[crossing_index]

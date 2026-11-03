"""Uncollided neutron wall load on a first wall that encloses a plasma boundary.

The neutron wall load is the power per unit area carried by uncollided fusion neutrons
through a wall surrounding the plasma. Its peak value, and the peak-to-mean ratio
(peaking factor), set the local damage and heating rate of the first wall and
blanket, so they are basic inputs to first wall and blanket engineering.

The model makes the usual assumptions of a first estimate:

- Neutrons are emitted isotropically from a volumetric source S inside the plasma
  boundary and travel in straight lines (no scattering, no attenuation).
- Each neutron is deposited where its straight path first leaves the wall surface.
  A source point therefore loads a wall point only if the segment between them stays
  inside the wall; paths that would cross the central hole of the torus, or a concave
  part of the wall, are occluded.

With these assumptions the load at a wall point x with outward unit normal n is

    q(x) = 1 / (4 pi) * integral over directions u with u . n < 0 of
           |u . n| * integral_0^L(u) S(x + s u) ds

where L(u) is the distance from x along u to the first point where the ray leaves
the wall. Unlike the equivalent volume integral of
S(r) (x - r) . n / (4 pi |x - r|^3), this form has no singularity, so it does not
need a source grid that is finer than the gap between the plasma and the wall.
Because every neutron leaves the wall exactly once, the wall integral of q equals the
total power. The returned `power_balance_error` measures how well the
discretisation satisfies this identity. It is a necessary check, not a sufficient one:
errors of the direction quadrature partly cancel in the wall integral, so local values
should be checked by increasing `n_polar_directions` and `n_azimuthal_directions`.
"""

import math
from typing import Callable

import jaxtyping as jt
import numpy as np
import pydantic
from vmecpp import _pydantic_numpy as pydantic_numpy

from constellaration.geometry import surface_rz_fourier, surface_utils

SourceProfile = Callable[[np.ndarray], np.ndarray]


class NeutronWallLoad(pydantic_numpy.BaseModelWithNumpy):
    """The neutron wall load on a wall surface, over one period of the load."""

    theta_phi: jt.Float[np.ndarray, "n_theta n_phi 2"]
    """The (theta, phi) coordinates of the wall at which the load is evaluated. Phi
    covers one period of the load, [0, 2 pi / n_periods)."""

    wall_load: jt.Float[np.ndarray, "n_theta n_phi"]
    """The wall load at `theta_phi`, in units of `total_power` per square meter (W/m^2
    if the power is given in W and the surfaces in meters)."""

    n_periods: int
    """Number of toroidal periods of the load, the greatest common divisor of the
    numbers of field periods of the plasma boundary and the wall."""

    wall_area: pydantic.NonNegativeFloat
    """Area of the whole wall, in square meters."""

    mean_wall_load: pydantic.NonNegativeFloat
    """Total power divided by the wall area."""

    peak_wall_load: pydantic.NonNegativeFloat
    """Maximum of the wall load over the evaluation grid."""

    peaking_factor: pydantic.NonNegativeFloat
    """Peak wall load divided by the mean wall load."""

    power_balance_error: float
    """(integral of the wall load over the wall - total power) / total power. It is
    zero for the exact solution. Local errors can be larger, see the module
    docstring."""


def compute_neutron_wall_load(
    plasma_boundary: surface_rz_fourier.SurfaceRZFourier,
    wall: surface_rz_fourier.SurfaceRZFourier,
    total_power: float = 1.0,
    source_profile: SourceProfile | None = None,
    n_theta_wall: int = 32,
    n_phi_wall: int = 32,
    n_polar_directions: int = 16,
    n_azimuthal_directions: int = 64,
    n_steps_per_minor_radius: int = 40,
) -> NeutronWallLoad:
    """Compute the uncollided neutron wall load on a wall that encloses the plasma.

    The source density inside the plasma boundary is proportional to
    `source_profile(rho)`, where rho in [0, 1] is a geometric radial label defined by

        (R, Z)(rho, theta, phi) = (R, Z)_0(phi)
                                  + rho * ((R, Z)_boundary(theta, phi) - (R, Z)_0(phi))

    with (R, Z)_0(phi) the m = 0 part of the boundary (the centre of each cross
    section). rho is not a flux label; for nested, similar flux surfaces it is close
    to the square root of the normalized toroidal flux. The default is a uniform
    source. For a DT plasma, the fusion power density is roughly proportional to the
    square of the plasma pressure.

    The plasma boundary and the wall must have cross sections that are star-shaped
    about their centres, and the wall must enclose the plasma boundary; both are
    checked.

    Args:
        plasma_boundary: The boundary that contains the neutron source.
        wall: The wall surface, e.g. the first wall.
        total_power: The total neutron power emitted by the source, e.g. 80 % of the
            DT fusion power. The load is returned in units of this power per m^2.
        source_profile: Relative source density as a function of rho, evaluated on
            an array of rho values. Defaults to a uniform source.
        n_theta_wall: Number of poloidal points at which the wall load is evaluated.
        n_phi_wall: Number of toroidal points, over one period of the load, at which
            the wall load is evaluated.
        n_polar_directions: Number of Gauss-Legendre points in the cosine of the
            angle between the ray and the wall normal.
        n_azimuthal_directions: Number of azimuthal directions of the rays around the
            wall normal.
        n_steps_per_minor_radius: Number of integration steps along each ray per mean
            minor radius of the plasma boundary.

    Returns:
        The wall load over one period of the load, with summary values.
    """
    if total_power <= 0.0:
        raise ValueError("total_power must be positive.")

    plasma_sections = _StarShapedSections(plasma_boundary, "plasma_boundary")
    wall_sections = _StarShapedSections(wall, "wall")
    boundary_xyz = surface_rz_fourier.evaluate_points_xyz(
        plasma_boundary, surface_utils.make_theta_phi_grid(64, 64)
    )
    if np.any(wall_sections.normalized_radius(boundary_xyz) >= 1.0):
        raise ValueError("The wall does not enclose the plasma boundary.")

    def source_density(xyz: np.ndarray) -> np.ndarray:
        rho = plasma_sections.normalized_radius(xyz)
        inside = rho < 1.0
        density = np.zeros(rho.shape)
        if np.any(inside):
            density[inside] = _evaluate_profile(source_profile, rho[inside])
        return density

    # Normalize the source density so that it emits total_power.
    source_scale = total_power / _integrate_profile_over_volume(
        plasma_boundary, source_profile
    )

    n_periods = math.gcd(plasma_boundary.n_field_periods, wall.n_field_periods)
    theta_phi = surface_utils.make_theta_phi_grid(
        n_theta_wall, n_phi_wall, phi_upper_bound=2 * np.pi / n_periods
    )
    wall_xyz = surface_rz_fourier.evaluate_points_xyz(wall, theta_phi)
    wall_normal = np.asarray(surface_rz_fourier.evaluate_normal(wall, theta_phi))
    normal_norm = np.linalg.norm(wall_normal, axis=-1)
    # Orient the normal outwards: the enclosed volume, (1/3) * integral of x . n dA,
    # must be positive.
    if np.sum(wall_xyz * wall_normal) < 0.0:
        wall_normal = -wall_normal
    wall_unit_normal = wall_normal / normal_norm[..., np.newaxis]
    area_element = (
        normal_norm * (2 * np.pi / n_theta_wall) * (2 * np.pi / n_periods / n_phi_wall)
    )
    wall_area = float(np.sum(area_element) * n_periods)

    step = (
        surface_rz_fourier.evaluate_minor_radius(plasma_boundary)
        / n_steps_per_minor_radius
    )
    # No ray can stay inside the wall for longer than the wall's diameter.
    max_length = 2.0 * float(np.max(np.linalg.norm(wall_xyz, axis=-1)))
    directions, direction_weights = _inward_hemisphere_quadrature(
        n_polar_directions, n_azimuthal_directions
    )

    wall_load = _integrate_along_rays(
        origins=wall_xyz.reshape(-1, 3),
        unit_normals=wall_unit_normal.reshape(-1, 3),
        directions=directions,
        direction_weights=direction_weights,
        source_density=source_density,
        # The tolerance keeps the first steps of grazing rays, which start on the
        # wall, from being cut off by the interpolation error of the wall table.
        inside_wall=lambda xyz: wall_sections.normalized_radius(xyz) <= 1.0 + 1e-4,
        step=step,
        n_steps=int(np.ceil(max_length / step)),
    ).reshape(n_theta_wall, n_phi_wall) * (source_scale / (4 * np.pi))

    integrated_load = float(np.sum(wall_load * area_element) * n_periods)
    mean_wall_load = total_power / wall_area
    peak_wall_load = float(np.max(wall_load))
    return NeutronWallLoad(
        theta_phi=theta_phi,
        wall_load=wall_load,
        n_periods=n_periods,
        wall_area=wall_area,
        mean_wall_load=mean_wall_load,
        peak_wall_load=peak_wall_load,
        peaking_factor=peak_wall_load / mean_wall_load,
        power_balance_error=(integrated_load - total_power) / total_power,
    )


def _evaluate_profile(
    source_profile: SourceProfile | None, rho: np.ndarray
) -> np.ndarray:
    if source_profile is None:
        return np.ones_like(rho)
    profile = np.asarray(source_profile(rho), dtype=float)
    if profile.shape != rho.shape or np.any(profile < 0.0):
        raise ValueError(
            "source_profile must return one non-negative value per rho value."
        )
    return profile


def _integrate_profile_over_volume(
    plasma_boundary: surface_rz_fourier.SurfaceRZFourier,
    source_profile: SourceProfile | None,
    n_rho: int = 64,
    n_theta: int = 64,
    n_phi_per_period: int = 64,
) -> float:
    """Integral of source_profile(rho) over the volume inside the boundary, using the
    (rho, theta, phi) parametrization of `compute_neutron_wall_load`."""
    n_phi = n_phi_per_period * plasma_boundary.n_field_periods
    theta_phi = surface_utils.make_theta_phi_grid(n_theta, n_phi)
    rz = surface_rz_fourier.evaluate_points_rz(plasma_boundary, theta_phi)
    centre = _cross_section_centre(plasma_boundary, theta_phi[0, :, 1])
    delta = rz - centre[np.newaxis, :, :]
    drz_dtheta = _evaluate_drz_dtheta(plasma_boundary, theta_phi)
    # |d(R, Z) / d(rho, theta)| = rho * |cross|
    cross = np.abs(
        delta[..., 0] * drz_dtheta[..., 1] - delta[..., 1] * drz_dtheta[..., 0]
    )
    rho = (np.arange(n_rho) + 0.5) / n_rho
    r = centre[np.newaxis, np.newaxis, :, 0] + rho[:, None, None] * delta[..., 0]
    profile = _evaluate_profile(source_profile, rho)
    integrand = profile[:, None, None] * r * rho[:, None, None] * cross[None]
    return float(
        np.sum(integrand) * (1.0 / n_rho) * (2 * np.pi / n_theta) * (2 * np.pi / n_phi)
    )


def _inward_hemisphere_quadrature(
    n_polar: int, n_azimuthal: int
) -> tuple[
    jt.Float[np.ndarray, "n_directions 3"], jt.Float[np.ndarray, " n_directions"]
]:
    """Directions in the local frame of a wall point (the third axis is the outward
    normal) that point into the wall, and the weights of the integral of
    |cos(angle to the normal)| over that hemisphere."""
    mu, mu_weights = np.polynomial.legendre.leggauss(n_polar)
    # Map from [-1, 1] to |cos| in [0, 1].
    mu = 0.5 * (mu + 1.0)
    mu_weights = 0.5 * mu_weights
    azimuth = 2 * np.pi * (np.arange(n_azimuthal) + 0.5) / n_azimuthal
    mu_grid, azimuth_grid = np.meshgrid(mu, azimuth, indexing="ij")
    sin_angle = np.sqrt(1.0 - mu_grid**2)
    directions = np.stack(
        (
            sin_angle * np.cos(azimuth_grid),
            sin_angle * np.sin(azimuth_grid),
            -mu_grid,
        ),
        axis=-1,
    ).reshape(-1, 3)
    weights = (mu_weights[:, None] * mu_grid * (2 * np.pi / n_azimuthal)).reshape(-1)
    return directions, weights


def _integrate_along_rays(
    origins: jt.Float[np.ndarray, "n_points 3"],
    unit_normals: jt.Float[np.ndarray, "n_points 3"],
    directions: jt.Float[np.ndarray, "n_directions 3"],
    direction_weights: jt.Float[np.ndarray, " n_directions"],
    source_density: Callable[[np.ndarray], np.ndarray],
    inside_wall: Callable[[np.ndarray], np.ndarray],
    step: float,
    n_steps: int,
    n_points_per_chunk: int = 32,
    n_steps_per_block: int = 16,
) -> jt.Float[np.ndarray, " n_points"]:
    """For each wall point, the weighted sum over directions of the integral of the
    source density along the ray, up to where the ray first leaves the wall."""
    result = np.zeros(len(origins))
    for start in range(0, len(origins), n_points_per_chunk):
        chunk = slice(start, start + n_points_per_chunk)
        rays = _rotate_to_normals(directions, unit_normals[chunk])
        origin = origins[chunk][:, np.newaxis, :]
        line_integral = np.zeros(rays.shape[:2])
        still_inside = np.ones(rays.shape[:2], dtype=bool)
        for first_step in range(0, n_steps, n_steps_per_block):
            k = np.arange(first_step, min(first_step + n_steps_per_block, n_steps))
            s = (k + 0.5) * step
            points = (
                origin[:, :, np.newaxis, :]
                + s[np.newaxis, np.newaxis, :, np.newaxis] * rays[:, :, np.newaxis, :]
            )
            inside = np.logical_and.accumulate(inside_wall(points), axis=-1)
            inside &= still_inside[..., np.newaxis]
            density = np.zeros(inside.shape)
            density[inside] = source_density(points[inside])
            line_integral += np.sum(density, axis=-1) * step
            still_inside = inside[..., -1]
            if not np.any(still_inside):
                break
        result[chunk] = line_integral @ direction_weights
    return result


def _rotate_to_normals(
    directions: jt.Float[np.ndarray, "n_directions 3"],
    unit_normals: jt.Float[np.ndarray, "n_points 3"],
) -> jt.Float[np.ndarray, "n_points n_directions 3"]:
    """Express local-frame directions (third axis = normal) in global coordinates."""
    helper = np.where(
        np.abs(unit_normals[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]]
    )
    tangent_1 = np.cross(helper, unit_normals)
    tangent_1 /= np.linalg.norm(tangent_1, axis=-1, keepdims=True)
    tangent_2 = np.cross(unit_normals, tangent_1)
    frame = np.stack((tangent_1, tangent_2, unit_normals), axis=-2)
    return np.einsum("dk,pkj->pdj", directions, frame)


def _cross_section_centre(
    surface: surface_rz_fourier.SurfaceRZFourier, phi: jt.Float[np.ndarray, " n_phi"]
) -> jt.Float[np.ndarray, "n_phi 2"]:
    """The m = 0 part of the surface, (R, Z)_0(phi), i.e. the poloidal average of the
    cross section at each phi."""
    # A uniform grid with more points than the largest poloidal mode averages all
    # m > 0 terms to zero exactly.
    n_theta = 2 * surface.max_poloidal_mode + 1
    theta = 2 * np.pi * np.arange(n_theta) / n_theta
    theta_phi = np.stack(np.meshgrid(theta, phi, indexing="ij"), axis=-1)
    return np.mean(surface_rz_fourier.evaluate_points_rz(surface, theta_phi), axis=0)


def _evaluate_drz_dtheta(
    surface: surface_rz_fourier.SurfaceRZFourier,
    theta_phi: jt.Float[np.ndarray, "*dims 2"],
) -> jt.Float[np.ndarray, "*dims 2"]:
    dxyz_dtheta = surface_rz_fourier.evaluate_dxyz_dtheta(surface, theta_phi)
    phi = theta_phi[..., 1]
    dr_dtheta = dxyz_dtheta[..., 0] * np.cos(phi) + dxyz_dtheta[..., 1] * np.sin(phi)
    return np.stack((dr_dtheta, dxyz_dtheta[..., 2]), axis=-1)


class _StarShapedSections:
    """Polar-coordinate table of a toroidal surface whose cross sections are
    star-shaped about their centres, used to locate points relative to it."""

    def __init__(
        self,
        surface: surface_rz_fourier.SurfaceRZFourier,
        name: str,
        n_theta: int = 512,
        n_phi_per_period: int = 128,
        n_polar_angle: int = 512,
    ):
        self._period = 2 * np.pi / surface.n_field_periods
        self._n_phi = n_phi_per_period
        self._n_angle = n_polar_angle
        phi = self._period * np.arange(n_phi_per_period) / n_phi_per_period
        theta_phi = surface_utils.make_theta_phi_grid(
            n_theta, n_phi_per_period, phi_upper_bound=self._period
        )
        rz = surface_rz_fourier.evaluate_points_rz(surface, theta_phi)
        self._centre = _cross_section_centre(surface, phi)
        delta = rz - self._centre[np.newaxis, :, :]
        drz_dtheta = _evaluate_drz_dtheta(surface, theta_phi)
        cross = delta[..., 0] * drz_dtheta[..., 1] - delta[..., 1] * drz_dtheta[..., 0]
        if not (np.all(cross > 0.0) or np.all(cross < 0.0)):
            raise ValueError(
                f"The cross sections of {name} are not star-shaped about their "
                "centres (the m = 0 part of the surface), which this model requires."
            )
        angle = np.arctan2(delta[..., 1], delta[..., 0])
        radius = np.linalg.norm(delta, axis=-1)
        polar_angle = 2 * np.pi * np.arange(n_polar_angle) / n_polar_angle - np.pi
        self._radius = np.empty((n_polar_angle, n_phi_per_period))
        for j in range(n_phi_per_period):
            order = np.argsort(angle[:, j])
            self._radius[:, j] = np.interp(
                polar_angle, angle[order, j], radius[order, j], period=2 * np.pi
            )

    def normalized_radius(
        self, xyz: jt.Float[np.ndarray, "*dims 3"]
    ) -> jt.Float[np.ndarray, "*dims"]:
        """Distance of each point from the centre of the cross section at its phi,
        divided by the distance of the surface in the same direction: < 1 inside the
        surface and > 1 outside it."""
        phi_index = (
            np.mod(np.arctan2(xyz[..., 1], xyz[..., 0]), self._period)
            / self._period
            * self._n_phi
        )
        j0 = np.floor(phi_index).astype(int)
        tj = phi_index - j0
        j0 %= self._n_phi
        j1 = (j0 + 1) % self._n_phi
        centre_r = (1 - tj) * self._centre[j0, 0] + tj * self._centre[j1, 0]
        centre_z = (1 - tj) * self._centre[j0, 1] + tj * self._centre[j1, 1]
        delta_r = np.hypot(xyz[..., 0], xyz[..., 1]) - centre_r
        delta_z = xyz[..., 2] - centre_z
        angle_index = (
            (np.arctan2(delta_z, delta_r) + np.pi) / (2 * np.pi) * self._n_angle
        )
        i0 = np.floor(angle_index).astype(int)
        ti = angle_index - i0
        i0 %= self._n_angle
        i1 = (i0 + 1) % self._n_angle
        surface_radius = (
            (1 - ti) * (1 - tj) * self._radius[i0, j0]
            + ti * (1 - tj) * self._radius[i1, j0]
            + (1 - ti) * tj * self._radius[i0, j1]
            + ti * tj * self._radius[i1, j1]
        )
        return np.hypot(delta_r, delta_z) / surface_radius

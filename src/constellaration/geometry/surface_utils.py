import jaxtyping as jt
import numpy as np


def make_theta_phi_grid(
    n_theta: int,
    n_phi: int,
    phi_upper_bound: float = 2 * np.pi,
    include_endpoints: bool = False,
) -> jt.Float[np.ndarray, "n_theta n_phi 2"]:
    """Make a theta_phi grid from 0 to 2 Pi in the theta and phi angles with ij indexing
    and endpoints in the array not included.

    Args:
        n_theta: Number of theta points.
        n_phi: Number of phi points.
        phi_upper_bound: Upper limit of phi angle.
        include_endpoints: Whether or not to include the theta and phi endpoints in the
            grid generation.

    Returns:
        theta_phi: A grid of theta and phi angles.
    """

    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=include_endpoints)
    phis = np.linspace(0, phi_upper_bound, n_phi, endpoint=include_endpoints)
    thetas_grid, phis_grid = np.meshgrid(thetas, phis, indexing="ij")
    theta_phi = np.stack([thetas_grid, phis_grid], axis=-1)

    return theta_phi


def n_poloidal_toroidal_points_to_satisfy_nyquist_criterion(
    n_poloidal_modes: int,
    max_toroidal_mode: int,
) -> tuple[int, int]:
    """Provides the minimum number of poloidal and toroidal grid points to satisfy the
    Nyquist criterion.

    See also:
    https://github.com/jonathanschilling/vmec-internals/blob/master/scalars.pdf

    Args:
        n_poloidal_modes: The number of poloidal modes.
        max_toroidal_mode: The maximum toroidal mode number.

    Returns:
        A tuple with the number of poloidal and toroidal grid points.
    """
    n_poloidal_points = 2 * n_poloidal_modes + 6
    n_toroidal_points = 2 * max_toroidal_mode + 4
    return n_poloidal_points, n_toroidal_points


def make_s_theta_phi_grid(
    n_radial_points: int,
    n_poloidal_points: int,
    n_toroidal_points: int,
    phi_upper_bound: float = 2 * np.pi,
    include_endpoints: bool = False,
) -> jt.Float[np.ndarray, "n_radial_points n_poloidal_points n_toroidal_points 3"]:
    """Creates a grid of (s, theta, phi) points that can be used for accurate surface
    integrals.

    Args:
        n_radial_points: The number of radial grid points.
        n_poloidal_points: The number of poloidal grid points.
        n_toroidal_points: The number of toroidal grid points.
        phi_upper_bound: The upper bound of the phi coordinate.
        include_endpoints: Whether to include the endpoints in the poloidal and
            toroidal dimensions of the grid.
    """
    s = np.linspace(0, 1, n_radial_points, endpoint=True)
    theta = (
        np.linspace(0, 1.0, n_poloidal_points, endpoint=include_endpoints) * 2 * np.pi
    )
    phi = (
        np.linspace(0, 1.0, n_toroidal_points, endpoint=include_endpoints)
        * phi_upper_bound
    )
    return np.stack(np.meshgrid(s, theta, phi, indexing="ij"), axis=-1)

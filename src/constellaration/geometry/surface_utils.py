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

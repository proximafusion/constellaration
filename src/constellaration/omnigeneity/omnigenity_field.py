from typing import Annotated

import interpax
import jax
import jax.numpy as jnp
import numpy as np
import pydantic
from jaxtyping import Float, Scalar
from vmecpp import _pydantic_numpy as pydantic_numpy

FloatBetween0And1 = Annotated[float, pydantic.Field(ge=0, le=1)]


class OmnigenousField(pydantic_numpy.BaseModelWithNumpy):
    n_field_periods: int = 1
    """Number of field periods."""

    poloidal_winding: pydantic.NonNegativeInt = 0
    """The number of times a B contour wraps around the torus in the poloidal
    direction."""

    torodial_winding: pydantic.NonNegativeInt = 1
    """The number of times a B contour wraps around the torus in the toroidal
    direction."""

    x_lmn: Float[
        np.ndarray, "n_x_rho_coefficients n_x_eta_coefficinets n_x_alpha_coefficients"
    ]
    """The coefficients that parameterize the variation of the magnetic well shape on
    different field lines and flux surfaces.

    See Eq. 2.7 of Dut et al. (2024).
    """

    modB_spline_knot_coefficients: Float[
        np.ndarray, "n_modb_rho_coefficients n_modb_eta_spline_knots"
    ]
    """The spline knots coefficients that parameterize the magnetic well strength
    evaluated at any rho between 0 and 1.

    n_modb_rho_coefficients is the number of modes of a Chebyshev basis as Eq. 2.4 in
    Dudt et al. (2024).

    .. math::
        B_{well} = \\sum_{k=0}^{n_modB_rho_coefficients} modB_spline_knot_coefficients[k] \\dot T_k(\\eta)
        \\text{where } T_k(\\eta) = cos(k arccos(\\eta))

    The final magnetic well strength is given by a spline interpolation of the knot points
    conditioned on zero derivative at the boundaries (Eq. 2.3).
    """  # noqa: E501

    @property
    def n_x_rho_coefficients(self) -> int:
        return self.x_lmn.shape[0]

    @property
    def n_x_eta_coefficinets(self) -> int:
        return self.x_lmn.shape[1]

    @property
    def n_x_alpha_coefficients(self) -> int:
        return self.x_lmn.shape[2]

    @property
    def max_x_rho_coefficients(self) -> int:
        return self.x_lmn.shape[0] - 1

    @property
    def max_x_eta_coefficinets(self) -> int:
        return self.x_lmn.shape[1] - 1

    @property
    def max_x_alpha_coefficients(self) -> int:
        return (self.x_lmn.shape[2] - 1) // 2

    @property
    def n_modb_rho_coefficients(self) -> int:
        return self.modB_spline_knot_coefficients.shape[0]

    @property
    def n_modB_eta_spline_knots(self) -> int:
        return self.modB_spline_knot_coefficients.shape[1]

    @property
    def max_modB_rho_coefficient(self) -> int:
        return self.modB_spline_knot_coefficients.shape[0] - 1

    @property
    def helicity(self) -> tuple[int, int]:
        return self.poloidal_winding, self.torodial_winding


def get_theta_and_phi_boozer(
    field: OmnigenousField,
    rho: float = 1.0,
    n_alpha: int = 100,
    n_eta: int = 100,
    iota: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Gets the Boozer theta and phi coordinates for a given rho. Only omnigenous
    poloidal (OP) supported for now.

    Args:
        field: an omnigenous field
        rho: The radial coordinate to evaluate the Boozer coordinates
        n_alpha: The number of points in a linearly spaced grid for the $\alpha$
            coordiante
        n_eta: The number of points in a linearly spaced grid for the $\\eta$ coordinate
        iota: The rotational transform

    Returns:
        A tuple of 1D arrays of the Boozer coordinates of shape n_alpha * n_eta
    """

    eta = jnp.linspace(-np.pi / 2, jnp.pi / 2, n_eta, endpoint=False)
    alpha = jnp.linspace(0, 2 * jnp.pi / field.n_field_periods, n_alpha, endpoint=False)

    h3d = _evaluate_h(field, jnp.asarray([rho]), eta=eta, alpha=alpha)

    alpha = jnp.linspace(0, 2 * jnp.pi / field.n_field_periods, n_alpha, endpoint=False)

    theta_b = (
        field.torodial_winding * alpha[:, None] + (iota / field.torodial_winding) * h3d
    )  # noqa: E501
    phi_b = (1 / field.torodial_winding) * h3d

    theta_b = jnp.mod(theta_b.flatten(), 2 * np.pi)
    phi_b = jnp.mod(phi_b.flatten(), 2 * np.pi / field.n_field_periods)

    return theta_b, phi_b


def get_modb_boozer(
    field: OmnigenousField,
    rho: float = 1.0,
    n_alpha: int = 100,
    n_eta: int = 100,
) -> jnp.ndarray:
    """Gets the magnetic field strength in Boozer coordinates for a given rho on an
    evenly spaced eta and alpha grid.

    Args:
        field: an omnigenous field
        rho: The radial coordinate to evaluate the Boozer coordinates
        n_alpha: The number of points in a linearly spaced grid for the $\alpha$
            coordiante
        n_eta: The number of points in a linearly spaced grid for the $\\eta$ coordinate

    Returns:
        A 1D array of the Boozer coordinates of shape n_alpha * n_eta
    """

    eta = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, n_eta, endpoint=False)

    return (
        _compute_magnetic_well_at_rho_eta(field, rho=jnp.asarray([rho]), eta=eta)[
            0, None, :
        ]
        .repeat(n_alpha, axis=0)
        .flatten()
    )


def find_modb_at_theta_phi_boozer(
    field: OmnigenousField,
    theta_b: jnp.ndarray,
    phi_b: jnp.ndarray,
    rho: float = 1.0,
    iota: float = 0.0,
) -> jnp.ndarray:
    """Finds the magnetic field strength from the Boozer coordinates $\\theta$ and
    $\\phi$ by solving for $B$ in Eq. 7 from Dut et al. (2024).

    Args:
        field: an omnigenous field
        theta_b: The theta coordinate to evaluate the Boozer coordinates
        phi_b: The phi coordinate to evaluate the Boozer coordinates

    Returns:
        A 1D array of the modB evaluated at (theta_b, phi_b).
    """
    if len(theta_b) != len(phi_b):
        raise ValueError("theta_b and phi_b must have the same length")

    alpha = theta_b / field.torodial_winding - iota * phi_b / field.torodial_winding
    h = field.torodial_winding * phi_b

    rhos = jnp.asarray([rho] * len(theta_b))

    eta = _find_eta_from_h_rho_alpha(field, rho=rhos, alpha=alpha, h_target=h)
    return _compute_magnetic_well_at_rho_eta(
        field, rho=jnp.asarray([rho]), eta=eta
    ).flatten()


def get_mirror_ratio_from_field(field: OmnigenousField):
    min_ = field.modB_spline_knot_coefficients[0, 0]
    max_ = field.modB_spline_knot_coefficients[0, -1]
    return (max_ - min_) / (min_ + max_)


def _compute_magnetic_well_at_rho_eta(
    field: OmnigenousField,
    rho: jnp.ndarray = jnp.asarray([1.0]),
    eta: jnp.ndarray = jnp.asarray([0.0]),
) -> Float[jnp.ndarray, "n_rho n_eta"]:
    """Computes the magnetic well strength for all values of $\\rho$ at the specified
    $\\eta$ values. The magnetic well strength is given by a spline interpolation of the
    modB knot points. The knot points for an arbitrary rho are computed with a Chevyshev
    series where the coefficients are in field.modB_spline_knots.

    Args:
        field: an omnigenous field
        rho: The radial coordinate where to evaluate the magnetic well
        eta: The number of points in a linearly spaced grid for the $\\eta$ coordinate

    Returns:
        A 2D array of the magnetic well strength of shape (n_rho, n_eta)
    """

    n_rho = len(rho)

    # Compute chebyshev basis
    modes = jnp.arange(field.n_modb_rho_coefficients)[None, :]
    basis = jnp.cos(np.abs(modes) * jnp.arccos(2 * rho[:, None] - 1))

    def _project_onto_basis(x):
        return basis @ x

    B_input = jax.vmap(_project_onto_basis)(field.modB_spline_knot_coefficients.T)
    B_input = jnp.sort(B_input, axis=0)  # ensure monotonicity

    eta_input = jnp.linspace(
        0, jnp.pi / 2, num=field.n_modB_eta_spline_knots, endpoint=True
    )  # Where the spline knots are defined

    eta2d = eta[None, :].repeat(n_rho, axis=0)

    def _interpolate(x, B):
        return interpax.interp1d(x, eta_input, B, method="monotonic-0")

    return jax.vmap(_interpolate)(jnp.abs(eta2d).reshape(n_rho, -1), B_input.T)


def _evaluate_h(
    field: OmnigenousField,
    rho: jnp.ndarray = jnp.asarray([1.0]),
    eta: jnp.ndarray = jnp.asarray([0.0]),
    alpha: jnp.ndarray = jnp.asarray([0.0]),
) -> Float[jnp.ndarray, "n_rho_points n_eta n_alpha"]:
    """Evaluates the h computational coordinate on a grid of rho, eta, and alpha points.
    Following Eq. 7 of Dut et al. (2024) is used to compute h.

    Args:
        field: an omnigenous field
        rhos: An array of radial coordinates to evaluate h
        n_eta: The number of points in a linearly spaced grid for the $\\eta$ coordinate
        n_alpha: The number of point in a linearly spaced grid for the $\alpha$
            coordiante
    """

    eta3d = eta[None, None, :]
    alpha3d = alpha[None, :, None]
    rho3d = rho[:, None, None]

    h_lmn = field.x_lmn

    xm = (
        jnp.arange(field.n_x_eta_coefficinets)[None, :, None]
        .repeat(field.n_x_alpha_coefficients, axis=2)
        .repeat(field.n_x_rho_coefficients, axis=0)
    )
    xn = (
        jnp.arange(-field.max_x_alpha_coefficients, field.max_x_alpha_coefficients + 1)[
            None, None, :
        ]
        .repeat(field.n_x_eta_coefficinets, axis=1)
        .repeat(field.n_x_rho_coefficients, axis=0)
    )
    xl = (
        jnp.arange(field.n_x_rho_coefficients)[:, None, None]
        .repeat(field.n_x_eta_coefficinets, axis=1)
        .repeat(field.n_x_alpha_coefficients, axis=2)
    )

    xm = xm.flatten()
    xn = xn.flatten()
    xl = xl.flatten()

    # Eq. 8
    fm = jnp.cos(jnp.abs(xm)[:, None, None, None] * eta3d[None, :, :, :])
    fn_cos = jnp.cos(
        jnp.abs(xn)[:, None, None, None]
        * alpha3d[None, :, :, :]
        * field.n_field_periods
    )
    fn_sin = jnp.sin(
        jnp.abs(xn)[:, None, None, None]
        * alpha3d[None, :, :, :]
        * field.n_field_periods
    )
    fn = jnp.where(xn[:, None, None, None] < 0, fn_sin, fn_cos)
    tl = jnp.cos(
        jnp.abs(xl)[:, None, None, None] * jnp.arccos(2 * rho3d[None, :, :, :] - 1)
    )

    # Eq.7
    h3d = (
        2 * eta3d
        + jnp.pi
        + jnp.sum(h_lmn.flatten()[:, None, None, None] * tl * fm * fn, axis=0)
    )

    return h3d


def _find_eta_from_h_rho_alpha(
    field: OmnigenousField,
    rho: jnp.ndarray,
    alpha: jnp.ndarray,
    h_target: jnp.ndarray,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> jnp.ndarray:
    """Finds the $\\eta$ coordinate from the h computational coordinat, $\\alpha$, and
    $\\rho$ by solving for $\\eta$ in Eq. 7 from Dut et al. (2024). This is particularly
    useful for identifying the magnetic field streth at specific boozer coordinates.

    Args:
        field: an omnigenous field
        rho: The radial coordinate to evaluate the Boozer coordinates
        alpha: The alpha coordinate to evaluate the Boozer coordinates
        h: The h coordinate to evaluate the Boozer coordinates

    Returns:
        A 2D array of the eta coordinates of shape (n_rho, n_alpha)
    """

    if len(rho) != len(alpha) and len(rho) != len(h_target):
        raise ValueError("rho alpha and h must have the same length")

    def single_eta_solver(rho_val: Scalar, alpha_val: Scalar, h_val: Scalar) -> Scalar:
        """Finds the eta coordinate for a given rho, alpha, and h value."""

        def body(state: tuple[Scalar, int]) -> tuple[Scalar, int]:
            eta_val, i = state
            h_val_est = _evaluate_h(
                field,
                rho=jnp.asarray([rho_val]),
                eta=jnp.asarray([eta_val]),
                alpha=jnp.asarray([alpha_val]),
            )[0, 0, 0]

            residual_func = h_val_est - h_val

            grad_residual_func = jax.grad(
                lambda eta: _evaluate_h(
                    field,
                    jnp.asarray([rho_val]),
                    jnp.asarray([eta]),
                    jnp.asarray([alpha_val]),
                )[0, 0, 0]
            )(eta_val)

            eta_next = eta_val - residual_func / (grad_residual_func + 1e-12)
            return (eta_next, i + 1)

        def cond(state: tuple[Scalar, int]):
            """Condition to continue the while loop."""
            eta_val, i = state
            h_val_est = _evaluate_h(
                field,
                rho=jnp.asarray([rho_val]),
                eta=jnp.asarray([eta_val]),
                alpha=jnp.asarray([alpha_val]),
            )[0, 0, 0]
            return jnp.logical_and(jnp.abs(h_val_est - h_val) > tol, i < max_iter)

        init_eta = jnp.asarray(np.pi / 2)
        final_eta, _ = jax.lax.while_loop(cond, body, (init_eta, 0))
        return final_eta

    return jnp.asarray(
        jax.vmap(single_eta_solver, in_axes=(0, 0, 0))(rho, alpha, h_target)
    )

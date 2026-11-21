import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from jax.scipy.special import logsumexp

from constellaration.geometry import radial_profile, surface_utils
from constellaration.mhd import vmec_utils


def minimum_normalized_mercier_criterion(
    equilibrium: vmec_utils.VmecppWOut, temperature: float
) -> jax.Array | None:
    r"""Conservative soft minimum of the normalized Mercier criterion.

    For VMEC's interior full-grid values :math:`x_i=D_{\mathrm{Merc},i}
    \Phi_{\mathrm{edge}}^2`, return :math:`-\tau\log\sum_i e^{-x_i/\tau}`,
    where :math:`\tau` is ``temperature``. This is the log-mean-exp soft
    minimum minus :math:`\tau\log n`; it is no greater than the smallest
    sampled value. It excludes the axis and edge, and does not establish
    stability between grid points. VMEC values near the axis may be unreliable.

    Args:
        equilibrium: VMEC equilibrium supplying DMerc and total toroidal flux.
        temperature: Finite, strictly positive smoothing scale in normalized
            Mercier units. Smaller values approximate the sampled minimum more
            closely.

    Returns:
        The conservative soft minimum, or None if there are no interior samples
        and NaN if any normalized interior value is nonfinite.
    """
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    values = jnp.asarray(equilibrium.DMerc)[1:-1] * equilibrium.phi[-1] ** 2
    if not values.size:
        return None
    minimum = -temperature * logsumexp(-values / temperature)
    return jnp.where(jnp.all(jnp.isfinite(values)), minimum, jnp.nan)


def mirror_ratio_well(axis, edge) -> jax.Array:
    r"""Return the unsigned normalized difference of edge and axis mirror ratios.

    The metric :math:`2|M_\mathrm{edge}-M_\mathrm{axis}|/
    (M_\mathrm{edge}+M_\mathrm{axis})` favors radially flat mirror ratios.
    This is motivated by the flat-mirror QI fields of Velasco et al.,
    "Robust stellarator optimization via flat mirror magnetic fields" (2023),
    https://doi.org/10.1088/1741-4326/acfe8a. This axis-to-edge measure is only
    a max-J proxy, not a proof of max-J.

    Args:
        axis: Magnetic mirror ratio on the magnetic axis.
        edge: Magnetic mirror ratio at the plasma edge.

    Returns:
        The dimensionless well, or infinity if the denominator is nonfinite or
        no larger than 1e-14.
    """
    denominator = jnp.asarray(axis + edge)
    valid = jnp.isfinite(denominator) & (denominator > 1e-14)
    safe_denominator = jax.lax.select(valid, denominator, jnp.ones_like(denominator))
    return jnp.where(valid, 2 * jnp.abs(edge - axis) / safe_denominator, jnp.inf)


def beta_peaking_factor(equilibrium: vmec_utils.VmecppWOut) -> jax.Array | None:
    r"""Return axis beta divided by VMEC's volume-averaged beta.

    VMEC's ``beta_vol`` is a surface-averaged beta on a half grid with a
    padding value at index zero. Following VMEC, axis beta is linearly
    extrapolated as :math:`1.5\,\beta_1-0.5\,\beta_2` from the first two
    physical half-grid values. This is beta peaking, not pressure peaking.

    Args:
        equilibrium: VMEC equilibrium supplying `beta_vol` and `betatotal`.

    Returns:
        The dimensionless ratio, or None with fewer than two half-grid samples.
        Returns NaN for invalid total or axis beta.
    """
    beta = jnp.asarray(equilibrium.betatotal)
    profile = equilibrium.beta_vol
    if len(profile) < 3:
        return None
    axis = 1.5 * profile[1] - 0.5 * profile[2]
    valid = jnp.isfinite(beta) & (beta > 0) & jnp.isfinite(axis)
    return jnp.where(valid, axis / jnp.where(valid, beta, 1), jnp.nan)


def vacuum_well(
    equilibrium: vmec_utils.VmecppWOut,
) -> jax.Array:
    r"""Computes a single number that summarizes the vacuum magnetic well, given by the
    formula below.

    This function reproduces the `vacuum_well` function in Simsopt.

    The vacuum well is defined as:

    .. math::

        W = \frac{\partial_s V(s=0) - \partial_s V(s=1)}{\partial_s V(s=0)}

    where :math:`V` is the volume enclosed by the flux surface, and :math:`s` is the
    normalized toroidal flux. Positive values of :math:`W` are favorable for stability
    to interchange modes. This formula for :math:`W` is motivated by the fact that
    :math:`\tilde{W} = \frac{d^2 V}{d s^2} < 0` is favorable for stability. Integrating
    over :math:`\tilde{W}` from 0 to 1 and normalizing gives the above formula
    for :math:`W`.
    """

    # gmnc are the Fourier coefficients of the Jacobian in VMEC coordinates.
    # In VMEC, the radial derivative of the volume is (2pi)^2 * |g^{1/2}|.
    d_volume_d_s = 4 * np.pi * np.pi * jnp.abs(equilibrium.gmnc[0, 1:])

    # Extrapolate linearly to the magnetic axis and the LCFS.
    d_volume_d_s_at_magnetic_axis = 1.5 * d_volume_d_s[0] - 0.5 * d_volume_d_s[1]
    d_volume_d_s_at_lcfs = 1.5 * d_volume_d_s[-1] - 0.5 * d_volume_d_s[-2]

    return (
        d_volume_d_s_at_magnetic_axis - d_volume_d_s_at_lcfs
    ) / d_volume_d_s_at_magnetic_axis


def magnetic_mirror_ratio(
    equilibrium: vmec_utils.VmecppWOut,
) -> radial_profile.InterpolatedRadialProfile:
    """Mirror ratio on the full grid, including the magnetic axis and edge."""
    field = _magnetic_field_strength_nyquist_resolution(equilibrium)
    high, low = field.max(axis=(1, 2)), field.min(axis=(1, 2))
    return radial_profile.InterpolatedRadialProfile.model_construct(
        rho=jnp.sqrt(jnp.linspace(0, 1, field.shape[0])),
        values=(high - low) / (high + low),
    )


def _magnetic_field_strength_nyquist_resolution(
    equilibrium: vmec_utils.VmecppWOut,
) -> jax.Array:
    (
        ntheta,
        nphi,
    ) = surface_utils.n_poloidal_toroidal_points_to_satisfy_nyquist_criterion(
        equilibrium.mpol, equilibrium.ntor
    )
    theta, phi = jnp.moveaxis(
        jnp.asarray(
            surface_utils.make_theta_phi_grid(
                ntheta,
                nphi,
                2 * np.pi / equilibrium.nfp / (1 + int(not equilibrium.lasym)),
                include_endpoints=True,
            )
        ),
        -1,
        0,
    )
    b = jnp.asarray(equilibrium.bmnc)
    full = jnp.concatenate(
        (
            1.5 * b[:, 1:2] - 0.5 * b[:, 2:3],
            (b[:, 1:-1] + b[:, 2:]) / 2,
            1.5 * b[:, -1:] - 0.5 * b[:, -2:-1],
        ),
        axis=1,
    )
    phase = theta[..., None] * equilibrium.xm_nyq - phi[..., None] * equilibrium.xn_nyq
    values = full.T @ jnp.cos(phase.reshape(-1, b.shape[0])).T
    return values.reshape(b.shape[1], *theta.shape)


def normalized_magnetic_gradient_scale_length(
    equilibrium: vmec_utils.VmecppWOut,
    theta_phi: jt.Float[np.ndarray, "n_poloidal_points n_toroidal_points 2"],
) -> jt.Float[np.ndarray, "n_poloidal_points n_toroidal_points"]:
    """Computes the magnetic gradient scale length.

    The quantity correlates with the coils-plasma distance for a given coil
    "complexity".

    This quantity is discussed in arXiv:2309.11342v1:

    The Magnetic Gradient Scale Length Explains Why Certain Plasmas
    Require Close External Magnetic Coils
    by John Kappel, Matt Landreman, and Dhairya Malhotra

    The base of this code was written by John Kappel and provided to us by Alan.
    Simsopt now also has an implementation available in the `vmec_compute_geometry`
    function.

    Args:
        equilibrium: The VMEC equilibrium.
        theta_phi: A grid of poloidal and toroidal angles at which to
            evaluate the magnetic gradient scale length.
    """

    ns = equilibrium.ns
    xm = equilibrium.xm
    xn = equilibrium.xn
    xm_nyq = equilibrium.xm_nyq
    xn_nyq = equilibrium.xn_nyq
    rmnc = equilibrium.rmnc.T
    zmns = equilibrium.zmns.T
    gmnc = equilibrium.gmnc.T[1:, :]
    bmnc = equilibrium.bmnc.T[1:, :]
    bsupumnc = equilibrium.bsupumnc.T[1:, :]
    bsupvmnc = equilibrium.bsupvmnc.T[1:, :]

    s_full = np.linspace(0, 1, ns)
    ds = s_full[2] - s_full[1]

    def get_d_x_d_s_at_the_boundary(x: np.ndarray, is_full_mesh: bool) -> np.ndarray:
        """Returns the derivative of x with respect to s at the plasma boundary.

        We need to extrapolate the derivative at the boundary because some
        quantities are not defined at the boundary.
        Coefficients for higher order approximations are taken from:
        https://www.ams.org/journals/mcom/1988-51-184/S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf
        """  # noqa: E501
        if is_full_mesh:
            # Third order approximation of the derivative
            # at the boundary
            d_x_d_s = (
                11 / 6 * x[-1, :] - 3 * x[-2, :] + 3 / 2 * x[-3, :] - 1 / 3 * x[-4, :]
            ) / ds
        else:
            # Second order approximation of the derivative
            # at half mesh
            d_x_d_s_n = (1.5 * x[-1, :] - 2.0 * x[-2, :] + 0.5 * x[-3, :]) / ds
            d_x_d_s_n_minus_1 = (x[-1, :] - x[-3, :]) / (2 * ds)
            d_x_d_s_n_minus_2 = (x[-2, :] - x[-4, :]) / (2 * ds)
            # Second order extrapolation to the boundary
            d_x_d_s = 7 / 4 * d_x_d_s_n - d_x_d_s_n_minus_1 + 1 / 4 * d_x_d_s_n_minus_2
        return d_x_d_s

    d_rmnc_d_s = get_d_x_d_s_at_the_boundary(np.asarray(rmnc), True)
    d_zmns_d_s = get_d_x_d_s_at_the_boundary(np.asarray(zmns), True)
    d_bmnc_d_s = get_d_x_d_s_at_the_boundary(np.asarray(bmnc), False)
    d_bsupumnc_d_s = get_d_x_d_s_at_the_boundary(np.asarray(bsupumnc), False)
    d_bsupvmnc_d_s = get_d_x_d_s_at_the_boundary(np.asarray(bsupvmnc), False)

    def get_x_at_the_boundary(x: np.ndarray, is_full_mesh: bool) -> np.ndarray:
        """Returns the value of x at the plasma boundary."""
        if is_full_mesh:
            return x[-1, :]
        else:
            # Second order extrapolation to the boundary
            return 7 / 4 * x[-1, :] - x[-2, :] + 1 / 4 * x[-3, :]

    rmnc = get_x_at_the_boundary(np.asarray(rmnc), True)
    zmns = get_x_at_the_boundary(np.asarray(zmns), True)
    gmnc = get_x_at_the_boundary(np.asarray(gmnc), False)
    bmnc = get_x_at_the_boundary(np.asarray(bmnc), False)
    bsupumnc = get_x_at_the_boundary(np.asarray(bsupumnc), False)
    bsupvmnc = get_x_at_the_boundary(np.asarray(bsupvmnc), False)

    xm = xm[:, np.newaxis, np.newaxis]
    xn = xn[:, np.newaxis, np.newaxis]

    rmnc = rmnc[:, np.newaxis, np.newaxis]
    zmns = zmns[:, np.newaxis, np.newaxis]

    d_rmnc_d_s = d_rmnc_d_s[:, np.newaxis, np.newaxis]
    d_zmns_d_s = d_zmns_d_s[:, np.newaxis, np.newaxis]

    theta2d = theta_phi[np.newaxis, :, :, 0]
    phi2d = theta_phi[np.newaxis, :, :, 1]

    angle = xm * theta2d - xn * phi2d
    cos_of_angle = np.cos(angle)
    sin_of_angle = np.sin(angle)

    R = np.sum(rmnc * cos_of_angle, axis=0)

    d_R_d_theta = np.sum(rmnc * xm * (-sin_of_angle), axis=0)
    d_R_d_phi = np.sum(rmnc * (-xn) * (-sin_of_angle), axis=0)

    d2_R_d_theta2 = np.sum(rmnc * xm * xm * (-cos_of_angle), axis=0)
    d2_R_d_theta_d_phi = np.sum(rmnc * xm * (-xn) * (-cos_of_angle), axis=0)
    d2_R_d_phi2 = np.sum(rmnc * (-xn) * (-xn) * (-cos_of_angle), axis=0)

    d_Z_d_theta = np.sum(zmns * xm * (cos_of_angle), axis=0)
    d_Z_d_phi = np.sum(zmns * (-xn) * (cos_of_angle), axis=0)

    d2_Z_d_theta2 = np.sum(zmns * xm * xm * (-sin_of_angle), axis=0)
    d2_Z_d_theta_d_phi = np.sum(zmns * xm * (-xn) * (-sin_of_angle), axis=0)
    d2_Z_d_phi2 = np.sum(zmns * (-xn) * (-xn) * (-sin_of_angle), axis=0)

    d_R_d_s = np.sum(d_rmnc_d_s * (cos_of_angle), axis=0)
    d_Z_d_s = np.sum(d_zmns_d_s * (sin_of_angle), axis=0)

    d2_R_d_s_d_theta = np.sum(d_rmnc_d_s * xm * (-sin_of_angle), axis=0)
    d2_R_d_s_d_phi = np.sum(d_rmnc_d_s * (-xn) * (-sin_of_angle), axis=0)

    d2_Z_d_s_d_theta = np.sum(d_zmns_d_s * xm * (cos_of_angle), axis=0)
    d2_Z_d_s_d_phi = np.sum(d_zmns_d_s * (-xn) * (cos_of_angle), axis=0)

    xm_nyq = xm_nyq[:, np.newaxis, np.newaxis]
    xn_nyq = xn_nyq[:, np.newaxis, np.newaxis]

    bmnc = bmnc[:, np.newaxis, np.newaxis]
    gmnc = gmnc[:, np.newaxis, np.newaxis]

    bsupumnc = bsupumnc[:, np.newaxis, np.newaxis]
    bsupvmnc = bsupvmnc[:, np.newaxis, np.newaxis]

    d_bmnc_d_s = d_bmnc_d_s[:, np.newaxis, np.newaxis]

    d_bsupumnc_d_s = d_bsupumnc_d_s[:, np.newaxis, np.newaxis]
    d_bsupvmnc_d_s = d_bsupvmnc_d_s[:, np.newaxis, np.newaxis]

    nyq_angle = xm_nyq * theta2d - xn_nyq * phi2d
    cos_of_nyq_angle = np.cos(nyq_angle)
    sin_of_nyq_angle = np.sin(nyq_angle)

    B = np.sum(bmnc * cos_of_nyq_angle, axis=0)
    sqrt_g = np.sum(gmnc * cos_of_nyq_angle, axis=0)

    B_sup_theta = np.sum(bsupumnc * cos_of_nyq_angle, axis=0)
    B_sup_phi = np.sum(bsupvmnc * cos_of_nyq_angle, axis=0)

    d_B_sup_theta_d_theta = np.sum(bsupumnc * xm_nyq * (-sin_of_nyq_angle), axis=0)
    d_B_sup_phi_d_theta = np.sum(bsupvmnc * xm_nyq * (-sin_of_nyq_angle), axis=0)

    d_B_sup_theta_d_phi = np.sum(bsupumnc * (-xn_nyq) * (-sin_of_nyq_angle), axis=0)
    d_B_sup_phi_d_phi = np.sum(bsupvmnc * (-xn_nyq) * (-sin_of_nyq_angle), axis=0)

    d_B_sup_theta_d_s = np.sum(d_bsupumnc_d_s * cos_of_nyq_angle, axis=0)
    d_B_sup_phi_d_s = np.sum(d_bsupvmnc_d_s * cos_of_nyq_angle, axis=0)

    cos_of_phi = np.cos(theta_phi[..., 1])
    sin_of_phi = np.sin(theta_phi[..., 1])

    grad_s__R = -d_Z_d_theta * R / sqrt_g
    grad_s__phi = (d_R_d_phi * d_Z_d_theta - d_R_d_theta * d_Z_d_phi) / sqrt_g
    grad_s__Z = d_R_d_theta * R / sqrt_g
    grad_s__X = grad_s__R * cos_of_phi + grad_s__phi * -sin_of_phi
    grad_s__Y = grad_s__R * sin_of_phi + grad_s__phi * cos_of_phi

    grad_theta__R = d_Z_d_s * R / sqrt_g
    grad_theta__phi = (d_R_d_s * d_Z_d_phi - d_R_d_phi * d_Z_d_s) / sqrt_g
    grad_theta__Z = -d_R_d_s * R / sqrt_g
    grad_theta__X = grad_theta__R * cos_of_phi + grad_theta__phi * -sin_of_phi
    grad_theta__Y = grad_theta__R * sin_of_phi + grad_theta__phi * cos_of_phi

    grad_phi__R = 0 * sqrt_g
    grad_phi__phi = 1 / R
    grad_phi__Z = 0 * sqrt_g
    grad_phi__X = grad_phi__R * cos_of_phi + grad_phi__phi * -sin_of_phi
    grad_phi__Y = grad_phi__R * sin_of_phi + grad_phi__phi * cos_of_phi

    d_B_X_d_s = (
        d_B_sup_theta_d_s * d_R_d_theta * cos_of_phi
        + B_sup_theta * d2_R_d_s_d_theta * cos_of_phi
        + d_B_sup_phi_d_s * d_R_d_phi * cos_of_phi
        + B_sup_phi * d2_R_d_s_d_phi * cos_of_phi
        - d_B_sup_phi_d_s * R * sin_of_phi
        - B_sup_phi * d_R_d_s * sin_of_phi
    )

    d_B_X_d_theta = (
        d_B_sup_theta_d_theta * d_R_d_theta * cos_of_phi
        + B_sup_theta * d2_R_d_theta2 * cos_of_phi
        + d_B_sup_phi_d_theta * d_R_d_phi * cos_of_phi
        + B_sup_phi * d2_R_d_theta_d_phi * cos_of_phi
        - d_B_sup_phi_d_theta * R * sin_of_phi
        - B_sup_phi * d_R_d_theta * sin_of_phi
    )

    d_B_X_d_phi = (
        d_B_sup_theta_d_phi * d_R_d_theta * cos_of_phi
        + B_sup_theta * d2_R_d_theta_d_phi * cos_of_phi
        - B_sup_theta * d_R_d_theta * sin_of_phi
        + d_B_sup_phi_d_phi * d_R_d_phi * cos_of_phi
        + B_sup_phi * d2_R_d_phi2 * cos_of_phi
        - B_sup_phi * d_R_d_phi * sin_of_phi
        - d_B_sup_phi_d_phi * R * sin_of_phi
        - B_sup_phi * d_R_d_phi * sin_of_phi
        - B_sup_phi * R * cos_of_phi
    )

    d_B_Y_d_s = (
        d_B_sup_theta_d_s * d_R_d_theta * sin_of_phi
        + B_sup_theta * d2_R_d_s_d_theta * sin_of_phi
        + d_B_sup_phi_d_s * d_R_d_phi * sin_of_phi
        + B_sup_phi * d2_R_d_s_d_phi * sin_of_phi
        + d_B_sup_phi_d_s * R * cos_of_phi
        + B_sup_phi * d_R_d_s * cos_of_phi
    )

    d_B_Y_d_theta = (
        d_B_sup_theta_d_theta * d_R_d_theta * sin_of_phi
        + B_sup_theta * d2_R_d_theta2 * sin_of_phi
        + d_B_sup_phi_d_theta * d_R_d_phi * sin_of_phi
        + B_sup_phi * d2_R_d_theta_d_phi * sin_of_phi
        + d_B_sup_phi_d_theta * R * cos_of_phi
        + B_sup_phi * d_R_d_theta * cos_of_phi
    )

    d_B_Y_d_phi = (
        d_B_sup_theta_d_phi * d_R_d_theta * sin_of_phi
        + B_sup_theta * d2_R_d_theta_d_phi * sin_of_phi
        + B_sup_theta * d_R_d_theta * cos_of_phi
        + d_B_sup_phi_d_phi * d_R_d_phi * sin_of_phi
        + B_sup_phi * d2_R_d_phi2 * sin_of_phi
        + B_sup_phi * d_R_d_phi * cos_of_phi
        + d_B_sup_phi_d_phi * R * cos_of_phi
        + B_sup_phi * d_R_d_phi * cos_of_phi
        - B_sup_phi * R * sin_of_phi
    )

    d_B_Z_d_s = (
        d_B_sup_theta_d_s * d_Z_d_theta
        + B_sup_theta * d2_Z_d_s_d_theta
        + d_B_sup_phi_d_s * d_Z_d_phi
        + B_sup_phi * d2_Z_d_s_d_phi
    )

    d_B_Z_d_theta = (
        d_B_sup_theta_d_theta * d_Z_d_theta
        + B_sup_theta * d2_Z_d_theta2
        + d_B_sup_phi_d_theta * d_Z_d_phi
        + B_sup_phi * d2_Z_d_theta_d_phi
    )

    d_B_Z_d_phi = (
        d_B_sup_theta_d_phi * d_Z_d_theta
        + B_sup_theta * d2_Z_d_theta_d_phi
        + d_B_sup_phi_d_phi * d_Z_d_phi
        + B_sup_phi * d2_Z_d_phi2
    )

    grad_B__XX = (
        d_B_X_d_s * grad_s__X
        + d_B_X_d_theta * grad_theta__X
        + d_B_X_d_phi * grad_phi__X
    )
    grad_B__XY = (
        d_B_X_d_s * grad_s__Y
        + d_B_X_d_theta * grad_theta__Y
        + d_B_X_d_phi * grad_phi__Y
    )
    grad_B__XZ = (
        d_B_X_d_s * grad_s__Z
        + d_B_X_d_theta * grad_theta__Z
        + d_B_X_d_phi * grad_phi__Z
    )

    grad_B__YX = (
        d_B_Y_d_s * grad_s__X
        + d_B_Y_d_theta * grad_theta__X
        + d_B_Y_d_phi * grad_phi__X
    )
    grad_B__YY = (
        d_B_Y_d_s * grad_s__Y
        + d_B_Y_d_theta * grad_theta__Y
        + d_B_Y_d_phi * grad_phi__Y
    )
    grad_B__YZ = (
        d_B_Y_d_s * grad_s__Z
        + d_B_Y_d_theta * grad_theta__Z
        + d_B_Y_d_phi * grad_phi__Z
    )

    grad_B__ZX = (
        d_B_Z_d_s * grad_s__X
        + d_B_Z_d_theta * grad_theta__X
        + d_B_Z_d_phi * grad_phi__X
    )
    grad_B__ZY = (
        d_B_Z_d_s * grad_s__Y
        + d_B_Z_d_theta * grad_theta__Y
        + d_B_Z_d_phi * grad_phi__Y
    )
    grad_B__ZZ = (
        d_B_Z_d_s * grad_s__Z
        + d_B_Z_d_theta * grad_theta__Z
        + d_B_Z_d_phi * grad_phi__Z
    )

    grad_B_double_dot_grad_B = (
        grad_B__XX**2
        + grad_B__XY**2
        + grad_B__XZ**2
        + grad_B__YX**2
        + grad_B__YY**2
        + grad_B__YZ**2
        + grad_B__ZX**2
        + grad_B__ZY**2
        + grad_B__ZZ**2
    )

    magnetic_gradient_scale_length = B * (2 / (grad_B_double_dot_grad_B)) ** (1 / 2)
    return magnetic_gradient_scale_length / equilibrium.Aminor_p

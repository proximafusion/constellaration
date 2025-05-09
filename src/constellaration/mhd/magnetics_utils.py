import jaxtyping as jt
import numpy as np
from constellaration.geometry import radial_profile, surface_utils
from constellaration.mhd import vmec_utils


def vacuum_well(
    equilibrium: vmec_utils.VmecppWOut,
) -> float:
    r"""Computes a single number that summarizes the vacuum magnetic well, given by the
    formula.

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
    d_volume_d_s = 4 * np.pi * np.pi * np.abs(equilibrium.gmnc[:, 0])

    # Extrapolate linearly to the magnetic axis and the LCFS.
    d_volume_d_s_at_magnetic_axis = 1.5 * d_volume_d_s[0] - 0.5 * d_volume_d_s[1]
    d_volume_d_s_at_lcfs = 1.5 * d_volume_d_s[-1] - 0.5 * d_volume_d_s[-2]

    return (
        d_volume_d_s_at_magnetic_axis - d_volume_d_s_at_lcfs
    ) / d_volume_d_s_at_magnetic_axis


def magnetic_mirror_ratio(
    equilibrium: vmec_utils.VmecppWOut,
) -> radial_profile.InterpolatedRadialProfile:
    magnetic_field_strength = _magnetic_field_strength_nyquist_resolution(equilibrium)
    magnetic_field_strength_max = np.max(magnetic_field_strength, axis=(1, 2))
    magnetic_field_strength_min = np.min(magnetic_field_strength, axis=(1, 2))
    magnetic_mirror_ratio = (
        magnetic_field_strength_max - magnetic_field_strength_min
    ) / (magnetic_field_strength_max + magnetic_field_strength_min)
    return radial_profile.InterpolatedRadialProfile(
        rho=np.sqrt(equilibrium.normalized_toroidal_flux_full_grid_mesh),
        values=magnetic_mirror_ratio,
    )


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
    rmnc = equilibrium.rmnc
    zmns = equilibrium.zmns
    gmnc = equilibrium.gmnc
    bmnc = equilibrium.bmnc
    bsupumnc = equilibrium.bsupumnc
    bsupvmnc = equilibrium.bsupvmnc

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

    d_rmnc_d_s = get_d_x_d_s_at_the_boundary(rmnc, True)
    d_zmns_d_s = get_d_x_d_s_at_the_boundary(zmns, True)
    d_bmnc_d_s = get_d_x_d_s_at_the_boundary(bmnc, False)
    d_bsupumnc_d_s = get_d_x_d_s_at_the_boundary(bsupumnc, False)
    d_bsupvmnc_d_s = get_d_x_d_s_at_the_boundary(bsupvmnc, False)

    def get_x_at_the_boundary(x: np.ndarray, is_full_mesh: bool) -> np.ndarray:
        """Returns the value of x at the plasma boundary."""
        if is_full_mesh:
            return x[-1, :]
        else:
            # Second order extrapolation to the boundary
            return 7 / 4 * x[-1, :] - x[-2, :] + 1 / 4 * x[-3, :]

    rmnc = get_x_at_the_boundary(rmnc, True)
    zmns = get_x_at_the_boundary(zmns, True)
    gmnc = get_x_at_the_boundary(gmnc, False)
    bmnc = get_x_at_the_boundary(bmnc, False)
    bsupumnc = get_x_at_the_boundary(bsupumnc, False)
    bsupvmnc = get_x_at_the_boundary(bsupvmnc, False)

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


def _magnetic_field_strength_nyquist_resolution(
    equilibrium: vmec_utils.VmecppWOut,
) -> jt.Float[np.ndarray, "n_flux_surfaces n_poloidal_points n_toroidal_points 3"]:
    (
        n_poloidal_points,
        n_toroidal_points,
    ) = surface_utils.n_poloidal_toroidal_points_to_satisfy_nyquist_criterion(
        n_poloidal_modes=equilibrium.mpol,
        max_toroidal_mode=equilibrium.ntor,
    )
    phi_upper_bound = (
        2 * np.pi / equilibrium.n_field_periods / (1 + int(not equilibrium.lasym))
    )
    s_theta_phi = surface_utils.make_s_theta_phi_grid(
        n_radial_points=equilibrium.ns,
        n_poloidal_points=n_poloidal_points,
        n_toroidal_points=n_toroidal_points,
        phi_upper_bound=phi_upper_bound,
        include_endpoints=True,
    )
    return vmec_utils.magnetic_field_magnitude(
        equilibrium=equilibrium,
        s_theta_phi=s_theta_phi,
    )

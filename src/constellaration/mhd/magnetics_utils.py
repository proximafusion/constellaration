import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from jax.scipy.special import logsumexp

from constellaration.geometry import radial_profile, surface_utils
from constellaration.mhd import vmec_utils
from constellaration.utils.types import NpOrJaxArray


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
    theta_phi: jt.Float[NpOrJaxArray, "n_poloidal_points n_toroidal_points 2"],
) -> jax.Array:
    """Evaluate sqrt(2) |B| / (a ||∇B||_F) on the boundary grid.

    The radial jets and the Fourier Jacobian use the same extrapolation as the
    reporting metric. Angular and Cartesian derivatives are obtained by AD.
    """
    theta, phi = theta_phi[..., 0], theta_phi[..., 1]
    ns = equilibrium.rmnc.shape[1]
    ds = 1 / (ns - 1)
    m, n = equilibrium.xm, equilibrium.xn
    mn, nn = equilibrium.xm_nyq, equilibrium.xn_nyq

    def full_jet(coeff):
        return coeff[:, -1], coeff[:, -4:] @ jnp.array([-1 / 3, 1.5, -3, 11 / 6]) / ds

    def half_value(coeff):
        return coeff[:, -3:] @ jnp.array([1 / 4, -1.0, 7 / 4])

    def half_jet(coeff):
        derivative = coeff[:, -4:] @ jnp.array([-1 / 8, 11 / 8, -27 / 8, 17 / 8]) / ds
        return half_value(coeff), derivative

    r, dr = full_jet(equilibrium.rmnc)
    z, dz = full_jet(equilibrium.zmns)
    bu, dbu = half_jet(equilibrium.bsupumnc)
    bv, dbv = half_jet(equilibrium.bsupvmnc)
    mod_b = half_value(equilibrium.bmnc)
    jac = half_value(equilibrium.gmnc)

    def position(q):
        offset, t, p = q
        phase = m * t - n * p
        radius = (r + offset * dr) @ jnp.cos(phase)
        height = (z + offset * dz) @ jnp.sin(phase)
        return jnp.array([radius * jnp.cos(p), radius * jnp.sin(p), height])

    basis = jax.jacfwd(position)

    def field(q):
        offset, t, p = q
        cos = jnp.cos(mn * t - nn * p)
        contravariant = jnp.array(
            [0, (bu + offset * dbu) @ cos, (bv + offset * dbv) @ cos]
        )
        return basis(q) @ contravariant

    def length(t, p):
        q = jnp.array([0, t, p])
        e_s, e_theta, e_phi = basis(q).T
        cos = jnp.cos(mn * t - nn * p)
        volume_jacobian = jac @ cos
        radius = r @ jnp.cos(m * t - n * p)
        gradients = jnp.stack(
            (
                jnp.cross(e_theta, e_phi) / volume_jacobian,
                jnp.cross(e_phi, e_s) / volume_jacobian,
                jnp.array([-jnp.sin(p), jnp.cos(p), 0]) / radius,
            )
        )
        grad_b = jax.jacfwd(field)(q) @ gradients
        return (
            jnp.sqrt(2)
            * (mod_b @ cos)
            / (equilibrium.Aminor_p * jnp.linalg.norm(grad_b))
        )

    return jax.vmap(length)(theta.flatten(), phi.flatten()).reshape(theta.shape)

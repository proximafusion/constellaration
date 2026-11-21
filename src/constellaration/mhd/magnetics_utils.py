import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from scipy.special import logsumexp
from vmecpp import enable_x64  # pyright: ignore[reportAttributeAccessIssue]

from constellaration.geometry import radial_profile, surface_utils
from constellaration.mhd import jax_gradient_length, vmec_utils


def minimum_normalized_mercier_criterion(
    equilibrium: vmec_utils.VmecppWOut, temperature: float
) -> float | None:
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
        or any normalized interior value is nonfinite.
    """
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    values = np.asarray(equilibrium.DMerc)[1:-1] * equilibrium.phi[-1] ** 2
    if not len(values) or not np.all(np.isfinite(values)):
        return None
    return -temperature * float(np.asarray(logsumexp(-values / temperature)).item())


def mirror_ratio_well(axis: float, edge: float) -> float:
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
    denominator = axis + edge
    if not np.isfinite(denominator) or denominator <= 1e-14:
        return float(np.inf)
    return float(2 * abs(edge - axis) / denominator)


def beta_peaking_factor(equilibrium: vmec_utils.VmecppWOut) -> float | None:
    r"""Return axis beta divided by VMEC's volume-averaged beta.

    VMEC's ``beta_vol`` is a surface-averaged beta on a half grid with a
    padding value at index zero. Following VMEC, axis beta is linearly
    extrapolated as :math:`1.5\,\beta_1-0.5\,\beta_2` from the first two
    physical half-grid values. This is beta peaking, not pressure peaking.

    Args:
        equilibrium: VMEC equilibrium supplying `beta_vol` and `betatotal`.

    Returns:
        The dimensionless ratio, or None for nonpositive or nonfinite total
        beta, fewer than two physical half-grid values, or nonfinite axis beta.
    """
    beta = float(equilibrium.betatotal)
    profile = np.asarray(equilibrium.beta_vol)
    if not np.isfinite(beta) or beta <= 0 or len(profile) < 3:
        return None
    axis = 1.5 * profile[1] - 0.5 * profile[2]
    return float(axis / beta) if np.isfinite(axis) else None


def vacuum_well(
    equilibrium: vmec_utils.VmecppWOut,
) -> float:
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
    d_volume_d_s = 4 * np.pi * np.pi * np.abs(equilibrium.gmnc[0, 1:])

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


_gradient_length = jax.jit(jax_gradient_length.normalized_scale_length)


def normalized_magnetic_gradient_scale_length(
    equilibrium: vmec_utils.VmecppWOut,
    theta_phi: jt.Float[np.ndarray, "n_poloidal_points n_toroidal_points 2"],
) -> jt.Float[np.ndarray, "n_poloidal_points n_toroidal_points"]:
    """Return the local boundary gradient length, normalized by minor radius."""
    fields = ("rmnc", "zmns", "bsupumnc", "bsupvmnc", "bmnc", "gmnc", "Aminor_p")
    mode_names = ("xm", "xn", "xm_nyq", "xn_nyq")
    with enable_x64(True):
        wout = {name: jnp.asarray(getattr(equilibrium, name)) for name in fields}
        modes = {name: jnp.asarray(getattr(equilibrium, name)) for name in mode_names}
        values = _gradient_length(wout, modes, theta_phi[..., 0], theta_phi[..., 1])
        return np.asarray(values)


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

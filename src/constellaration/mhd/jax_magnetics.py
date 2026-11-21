"""Magnetic well and mirror ratio on VMEC radial and angular grids."""

import jax.numpy as jnp
from jax.scipy.special import logsumexp


def mirror_ratios(wout, modes, theta, phi):
    """Axis and edge mirror ratios with linear extrapolation from the half grid."""
    b = wout["bmnc"]
    coefficients = jnp.stack(
        (1.5 * b[:, 1] - 0.5 * b[:, 2], 1.5 * b[:, -1] - 0.5 * b[:, -2])
    )
    phase = theta[..., None] * modes["xm_nyq"] - phi[..., None] * modes["xn_nyq"]
    values = coefficients @ jnp.cos(phase.reshape(-1, b.shape[0])).T
    high, low = values.max(axis=-1), values.min(axis=-1)
    return (high - low) / (high + low)


def vacuum_well(wout):
    dvds = jnp.abs(wout["gmnc"][0])
    axis = 1.5 * dvds[1] - 0.5 * dvds[2]
    edge = 1.5 * dvds[-1] - 0.5 * dvds[-2]
    return 1 - edge / axis


def mercier_softmin(wout, temperature):
    normalized = wout["DMerc"][1:-1] * wout["phi"][-1] ** 2
    return -temperature * logsumexp(-normalized / temperature)

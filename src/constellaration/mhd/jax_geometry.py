"""Geometrical metrics evaluated on traced Fourier coefficients."""

import jax
import jax.numpy as jnp


def _boundary_modes(boundary):
    m = jnp.arange(boundary.shape[1])[:, None]
    n = jnp.arange(-(boundary.shape[2] // 2), boundary.shape[2] // 2 + 1)
    return m, n


def mean_cross_sectional_area(boundary: jax.Array, nfp: int) -> jax.Array:
    """Poloidal area averaged over the same quadrature as the reporting model."""
    m, n = _boundary_modes(boundary)
    theta = 2 * jnp.pi * jnp.arange(49) / 49
    phi = 2 * jnp.pi * jnp.arange(50) / 50
    phase = theta[:, None, None, None] * m - phi[None, :, None, None] * n * nfp
    cos = jnp.cos(phase)
    radius = jnp.sum(boundary[0] * cos, axis=(-2, -1))
    z_theta = jnp.sum(m * boundary[1] * cos, axis=(-2, -1))
    return 2 * jnp.pi * jnp.abs(jnp.mean(radius * z_theta))


def average_triangularity(boundary: jax.Array, nfp: int) -> jax.Array:
    """Triangularity at symmetry planes, using the continuous maximum of Z.

    A grid locates the top of the section; local Newton refinement of Z'=0
    supplies the dependence of its poloidal angle on the boundary coefficients.
    The centroid and radial width keep the reporting model's grid convention.
    """
    m, n = _boundary_modes(boundary)
    theta = jnp.linspace(0, 2 * jnp.pi, 201)

    def one_plane(phi):
        def rz(t):
            phase = m * t[..., None, None] - n * nfp * phi
            return (
                jnp.sum(boundary[0] * jnp.cos(phase), axis=(-2, -1)),
                jnp.sum(boundary[1] * jnp.sin(phase), axis=(-2, -1)),
            )

        r, z = rz(theta)
        start = theta[jnp.argmax(z)]

        def step(_, t):
            phase = m * t - n * nfp * phi
            slope = jnp.sum(m * boundary[1] * jnp.cos(phase))
            second = -jnp.sum(m**2 * boundary[1] * jnp.sin(phase))
            return t - slope / second

        top = jax.lax.fori_loop(0, 5, step, start)
        return 2 * (jnp.mean(r) - rz(top)[0]) / jnp.ptp(r)

    return (one_plane(0.0) + one_plane(jnp.pi / nfp)) / 2

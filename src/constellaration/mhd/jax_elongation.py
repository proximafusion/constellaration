"""Axis-normal elongation on a JAX equilibrium."""

import jax
import jax.numpy as jnp
import numpy as np

# Ellipse perimeter at fixed area is strictly increasing in log(a/b) >= 0.
# The table is computed once, independently of any equilibrium. Interpolating
# its inverse avoids differentiating a numerical root finder near circularity.
_LOG_ELONGATION = np.linspace(0, np.log(200), 16385)
_ANGLE = 2 * np.pi * np.arange(128) / 128
_PERIMETER_RATIO = np.sqrt(
    np.exp(_LOG_ELONGATION[:, None]) * np.sin(_ANGLE) ** 2
    + np.exp(-_LOG_ELONGATION[:, None]) * np.cos(_ANGLE) ** 2
).mean(axis=1)


def max_elongation(wout, xm, xn, nfp: int, n_theta: int, n_phi: int):
    """Maximum perimeter/area ellipse elongation on axis-normal sections.

    Unlike the legacy polygon, periodic quadrature integrates the perimeter
    and area from the tangent of each section, avoiding a zero-length edge.
    """
    theta = 2 * jnp.pi * jnp.arange(max(64, n_theta)) / max(64, n_theta)
    phi_grid = jnp.linspace(0, 2 * jnp.pi / nfp, n_phi)
    r_edge, z_edge = wout["rmnc"][:, -1], wout["zmns"][:, -1]
    r_axis, z_axis = wout["raxis_cc"], wout["zaxis_cs"]
    n_axis = xn[: r_axis.size]

    def cylindrical(r, z, dr, dz, phi):
        c, s = jnp.cos(phi), jnp.sin(phi)
        return jnp.stack([r * c, r * s, z]), jnp.stack(
            [dr * c - r * s, dr * s + r * c, dz]
        )

    def edge(t, p):
        # Match the reporting model's handedness convention for this metric.
        phase = xm * t + xn * p
        c, s = jnp.cos(phase), jnp.sin(phase)
        return cylindrical(
            r_edge @ c, z_edge @ s, -(r_edge * xn) @ s, (z_edge * xn) @ c, p
        )

    def section(phi):
        c, s = jnp.cos(n_axis * phi), jnp.sin(n_axis * phi)
        origin, axis_tangent = cylindrical(
            r_axis @ c, z_axis @ s, -(r_axis * n_axis) @ s, (z_axis * n_axis) @ c, phi
        )
        direction = axis_tangent / jnp.linalg.norm(axis_tangent)

        def sample(t):
            def refine(_, p):
                position, dp = edge(t, p)
                return p - ((position - origin) @ direction) / (dp @ direction)

            p = jax.lax.fori_loop(0, 10, refine, phi)
            position, dp = edge(t, p)
            phase = xm * t + xn * p
            rt = -(r_edge * xm) @ jnp.sin(phase)
            zt = (z_edge * xm) @ jnp.cos(phase)
            dt = jnp.array([rt * jnp.cos(p), rt * jnp.sin(p), zt])
            tangent = dt - dp * (dt @ direction) / (dp @ direction)
            return position - origin, tangent, (position - origin) @ direction

        pos, tangent, residual = jax.vmap(sample)(theta)
        perimeter = 2 * jnp.pi * jnp.mean(jnp.linalg.norm(tangent, axis=1))
        area = jnp.pi * jnp.abs(jnp.mean(jnp.cross(pos, tangent) @ direction))
        ratio = perimeter / jnp.sqrt(4 * jnp.pi * area)
        elongation = jnp.exp(jnp.interp(ratio, _PERIMETER_RATIO, _LOG_ELONGATION))
        # A missed plane intersection or out-of-range fit must be observable.
        return jnp.where(
            (jnp.max(jnp.abs(residual)) < 1e-9) & (ratio <= _PERIMETER_RATIO[-1]),
            elongation,
            jnp.nan,
        )

    return jnp.max(jax.vmap(section)(phi_grid))

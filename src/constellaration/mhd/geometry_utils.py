import jax
import jax.numpy as jnp
import numpy as np

from constellaration.geometry import surface_rz_fourier
from constellaration.mhd import vmec_utils

# Ellipse perimeter at fixed area is strictly increasing in log(a/b) >= 0.
# The table is computed once, independently of any equilibrium. Interpolating
# its inverse avoids differentiating a numerical root finder near circularity.
_LOG_ELONGATION = np.linspace(0, np.log(200), 16385)
_ANGLE = 2 * np.pi * np.arange(128) / 128
_PERIMETER_RATIO = np.sqrt(
    np.exp(_LOG_ELONGATION[:, None]) * np.sin(_ANGLE) ** 2
    + np.exp(-_LOG_ELONGATION[:, None]) * np.cos(_ANGLE) ** 2
).mean(axis=1)


def max_elongation(
    equilibrium: vmec_utils.VmecppWOut,
    n_poloidal_points: int,
    n_toroidal_points: int,
) -> jax.Array:
    """Maximum perimeter/area ellipse elongation on axis-normal sections.

    Unlike the legacy polygon, periodic quadrature integrates the perimeter
    and area from the tangent of each section, avoiding a zero-length edge.
    """
    if equilibrium.lasym:
        raise NotImplementedError("Non-stellarator symmetric equilibria not supported.")
    xm, xn = equilibrium.xm, equilibrium.xn
    nfp = equilibrium.nfp
    theta = (
        2 * jnp.pi * jnp.arange(max(64, n_poloidal_points)) / max(64, n_poloidal_points)
    )
    phi_grid = jnp.linspace(0, 2 * jnp.pi / nfp, n_toroidal_points)
    r_edge, z_edge = equilibrium.rmnc[:, -1], equilibrium.zmns[:, -1]
    r_axis, z_axis = equilibrium.raxis_cc, equilibrium.zaxis_cs
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
        # Reject missed intersections, allowing for roundoff in the active dtype.
        tolerance = jnp.maximum(
            1e-9, 32 * jnp.finfo(pos.dtype).eps * jnp.linalg.norm(origin)
        )
        return jnp.where(
            (jnp.max(jnp.abs(residual)) < tolerance) & (ratio <= _PERIMETER_RATIO[-1]),
            elongation,
            jnp.nan,
        )

    return jnp.max(jax.vmap(section)(phi_grid))


def average_triangularity(
    surface: surface_rz_fourier.SurfaceRZFourier,
    n_poloidal_points: int = 201,
) -> jax.Array:
    """Average triangularity of the two stellarator symmetry sections."""
    if not surface.is_stellarator_symmetric:
        raise NotImplementedError("Non-stellarator symmetric surfaces not supported.")
    return _triangularity(
        surface.r_cos.ravel(),
        surface.z_sin.ravel(),
        surface.poloidal_modes.ravel(),
        surface.toroidal_modes.ravel() * surface.n_field_periods,
        surface.n_field_periods,
        n_poloidal_points,
    )


def equilibrium_average_triangularity(equilibrium: vmec_utils.VmecppWOut) -> jax.Array:
    """Average triangularity of the equilibrium's outermost flux surface."""
    if equilibrium.lasym:
        raise NotImplementedError("Non-stellarator symmetric surfaces not supported.")
    return _triangularity(
        equilibrium.rmnc[:, -1],
        equilibrium.zmns[:, -1],
        equilibrium.xm,
        equilibrium.xn,
        equilibrium.nfp,
        201,
    )


def _triangularity(r_cos, z_sin, m, n, nfp, n_theta):
    # Keep the legacy centroid and radial width quadrature but refine the height
    # maximum continuously so its poloidal angle contributes to the derivative.
    theta = jnp.linspace(0, 2 * jnp.pi, n_theta)

    def one_plane(phi):
        def phase(t):
            return t[..., None] * m - phi * n

        angles = phase(theta)
        r = jnp.cos(angles) @ r_cos
        z = jnp.sin(angles) @ z_sin
        start = theta[jnp.argmax(z)]

        def step(_, t):
            slope = jnp.cos(phase(t)) @ (m * z_sin)
            second = -jnp.sin(phase(t)) @ (m**2 * z_sin)
            return t - slope / second

        top = jax.lax.fori_loop(0, 5, step, start)
        return 2 * (jnp.mean(r) - jnp.cos(phase(top)) @ r_cos) / jnp.ptp(r)

    return (one_plane(0.0) + one_plane(jnp.pi / nfp)) / 2

"""Magnetic gradient scale length from the VMEC boundary jet."""

import jax
import jax.numpy as jnp


def normalized_scale_length(wout, modes, theta, phi):
    """Evaluate sqrt(2) |B| / (a ||∇B||_F) on the boundary grid.

    The radial jets and the Fourier Jacobian use the same extrapolation as the
    reporting metric. Angular and Cartesian derivatives are obtained by AD.
    """
    ns = wout["rmnc"].shape[1]
    ds = 1 / (ns - 1)
    m, n = modes["xm"], modes["xn"]
    mn, nn = modes["xm_nyq"], modes["xn_nyq"]

    def full_jet(coeff):
        return coeff[:, -1], coeff[:, -4:] @ jnp.array([-1 / 3, 1.5, -3, 11 / 6]) / ds

    def half_value(coeff):
        return coeff[:, -3:] @ jnp.array([1 / 4, -1.0, 7 / 4])

    def half_jet(coeff):
        derivative = coeff[:, -4:] @ jnp.array([-1 / 8, 11 / 8, -27 / 8, 17 / 8]) / ds
        return half_value(coeff), derivative

    r, dr = full_jet(wout["rmnc"])
    z, dz = full_jet(wout["zmns"])
    bu, dbu = half_jet(wout["bsupumnc"])
    bv, dbv = half_jet(wout["bsupvmnc"])
    mod_b = half_value(wout["bmnc"])
    jac = half_value(wout["gmnc"])

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
            jnp.sqrt(2) * (mod_b @ cos) / (wout["Aminor_p"] * jnp.linalg.norm(grad_b))
        )

    return jax.vmap(length)(theta.flatten(), phi.flatten()).reshape(theta.shape)

"""Bad-curvature flux compression at straight-field-line quadrature points."""

import jax
import jax.numpy as jnp


def _radial_value(coefficients, s, *, half):
    """Local cubic interpolation on the uniform full or staggered VMEC mesh."""
    ns = coefficients.shape[-1]
    x = s * (ns - 1) + (0.5 if half else 0.0)
    start = jnp.clip(jnp.floor(x).astype(int) - 1, int(half), ns - 4)
    t = x - start
    weights = jnp.array(
        [
            -(t - 1) * (t - 2) * (t - 3) / 6,
            t * (t - 2) * (t - 3) / 2,
            -t * (t - 1) * (t - 3) / 2,
            t * (t - 1) * (t - 2) / 6,
        ]
    )
    values = jax.lax.dynamic_slice_in_dim(coefficients, start, 4, axis=-1)
    return values @ weights


def flux_compression(wout, modes, surfaces, nfp, n_lines, n_points, *, smoothing=0.02):
    """PEST-Jacobian-weighted compression, with a smooth bad-curvature mask.

    The sigmoid's width is ``smoothing`` times the RMS curvature drift on
    each surface. Unlike a step mask, it yields gradients as the boundary of
    the bad-curvature region moves through the quadrature grid. Set smoothing
    to zero for reporting with the historical hard mask.
    """
    alpha = jnp.linspace(-jnp.pi / nfp, jnp.pi / nfp, n_lines, endpoint=False)
    pest = jnp.linspace(-jnp.pi, jnp.pi, n_points, endpoint=False)
    m, n = modes["xm"], modes["xn"]
    mn, nn = modes["xm_nyq"], modes["xn_nyq"]
    edge_flux = wout["phipf"][-1]

    def iota(s):
        return _radial_value(wout["iotas"], s, half=True)

    def position(q):
        s, theta, phi = q
        phase = m * theta - n * phi
        r = _radial_value(wout["rmnc"], s, half=False) @ jnp.cos(phase)
        z = _radial_value(wout["zmns"], s, half=False) @ jnp.sin(phase)
        return jnp.array([r * jnp.cos(phi), r * jnp.sin(phi), z])

    def straight_theta(q):
        s, theta, phi = q
        phase = m * theta - n * phi
        return theta + _radial_value(wout["lmns"], s, half=True) @ jnp.sin(phase)

    def strength(q):
        s, theta, phi = q
        return _radial_value(wout["bmnc"], s, half=True) @ jnp.cos(
            mn * theta - nn * phi
        )

    def fieldline_label(q):
        return straight_theta(q) - q[2] * iota(q[0])

    def point(s, target, phi):
        def step(_, theta):
            q = jnp.array([s, theta, phi])
            return theta - (straight_theta(q) - target) / jax.grad(straight_theta)(q)[1]

        theta = jax.lax.fori_loop(0, 10, step, target)
        q = jnp.array([s, theta, phi])
        e_theta, e_phi = jax.jacfwd(position)(q).T[1:]
        phase = mn * theta - nn * phi
        cs, sn = jnp.cos(phase), jnp.sin(phase)
        jac = _radial_value(wout["gmnc"], s, half=True) @ cs
        gradient_s = jnp.cross(e_theta, e_phi) / jac
        covariant_b = jnp.array(
            [
                _radial_value(wout["bsubsmns"], s, half=False) @ sn,
                _radial_value(wout["bsubumnc"], s, half=True) @ cs,
                _radial_value(wout["bsubvmnc"], s, half=True) @ cs,
            ]
        )
        triple = jnp.cross(covariant_b, jax.grad(strength)(q)) @ jax.grad(
            fieldline_label
        )(q)
        pressure_gradient = jax.grad(
            lambda x: _radial_value(wout["pres"], x, half=True)
        )(s)
        # VMEC's signed Clebsch flux is minus the enclosed toroidal flux.
        curvature = (
            triple / (jac * strength(q))
            - 8e-7 * jnp.pi**2 * pressure_gradient / wout["phi"][-1]
        )
        b_sup_phi = _radial_value(wout["bsupvmnc"], s, half=True) @ cs
        weight = jnp.abs(1 / b_sup_phi)  # surface-constant factors cancel
        compression = wout["Aminor_p"] ** 2 * (gradient_s @ gradient_s) / (4 * s)
        residual = jnp.abs(straight_theta(q) - target)
        return compression, curvature * jnp.sign(edge_flux), weight, residual

    def one_surface(s):
        phi = (pest[None, :] - alpha[:, None]) / iota(s)
        target = jnp.broadcast_to(pest, phi.shape)
        compression, curvature, weight, residual = jax.vmap(
            point, in_axes=(None, 0, 0)
        )(s, target.flatten(), phi.flatten())
        if smoothing:
            width = smoothing * jnp.sqrt(jnp.mean(curvature**2))
            bad_curvature = jax.nn.sigmoid(curvature / width)
        else:
            bad_curvature = curvature > 0
        result = jnp.sum(weight * compression * bad_curvature) / jnp.sum(weight)
        return jnp.where(jnp.max(residual) < 1e-9, result, jnp.nan)

    return jax.vmap(one_surface)(jnp.asarray(surfaces))

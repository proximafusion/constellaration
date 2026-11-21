"""Differentiable Goodman (2023) squash–stretch–shuffle residual."""

import jax
import jax.numpy as jnp


def _well_residual(field, levels, x, output_x):
    """One sampled surface, with the field ordered (line, toroidal coordinate)."""
    field = (field - field.min()) / (field.max() - field.min())
    bottom = field.min(axis=-1, keepdims=True)
    # Locate the well bottom continuously, within ~one grid cell for a
    # well-resolved minimum. Width is measured in normalized |B| units.
    minimum_weights = jax.nn.softmax(-(field - bottom) / 0.001, axis=-1)
    centers = minimum_weights @ x

    def squash(line, low, center):
        left_side = x <= center
        absent = jnp.full_like(line, -jnp.inf)
        left_max = jnp.argmax(jax.lax.select(left_side, line, absent))
        right_max = jnp.argmax(jax.lax.select(left_side, absent, line))
        index = jnp.arange(x.size)
        left_peak, right_peak = line[left_max], line[right_max]
        left = jax.lax.associative_scan(
            jnp.minimum, jnp.where(index < left_max, left_peak, line)
        )
        right = jax.lax.associative_scan(
            jnp.minimum, jnp.where(index > right_max, right_peak, line), reverse=True
        )
        # Squash to monotone branches, then stretch each to [0, 1]. The small
        # width floor only regularizes degenerate (absent) well branches.
        left = (left - low) / jnp.maximum(left_peak - low, 1e-6)
        right = (right - low) / jnp.maximum(right_peak - low, 1e-6)
        return jnp.where(left_side, left, right)

    stretched = jax.vmap(squash)(field, bottom[:, 0], centers)
    squared_change = (field - stretched) ** 2
    change = (squared_change[:, :-1] + squared_change[:, 1:]).mean(axis=-1) / 2
    weights = 1 / (change + 1e-6)
    weights = weights / weights.sum()

    # The inverse of a monotone well branch can be written as the integral of
    # its above-level indicator. A narrow sigmoid smooths crossing switches.
    # Integrate at cell midpoints and blend the left/right split over one cell.
    dx = x[1] - x[0]
    midpoints = (x[:-1] + x[1:]) / 2
    split = jax.nn.sigmoid((centers[:, None] - midpoints) / (0.5 * dx))
    mid_field = (stretched[:, :-1] + stretched[:, 1:]) / 2

    def bounces(level):
        above = jax.nn.sigmoid((mid_field - level) / 0.002)
        left = dx * jnp.sum(above * split, axis=-1)
        right = 1 - dx * jnp.sum(above * (1 - split), axis=-1)
        return left, right

    left, right = jax.lax.map(bounces, levels, batch_size=16)
    left = left.at[0].set(centers).at[-1].set(0)
    right = right.at[0].set(centers).at[-1].set(1)
    half_width = ((right - left) @ weights) / 2
    half_width = (1 - 1e-6) * half_width + 0.5e-6 * levels
    midpoint = (right + left) / 2

    # Shuffle all lines to the weighted mean width at each bounce level.
    # Limit shifts of the midpoint to keep the shuffled wells nested, rather
    # than sorting bounce points (which can swap the field levels).
    limits = (1 - 1e-6) * jnp.diff(half_width)[:, None]

    def keep_nested(outer, pair):
        center, limit = pair
        inner = jnp.clip(center, outer - limit, outer + limit)
        return inner, inner

    _, inner = jax.lax.scan(
        keep_nested, midpoint[-1], (midpoint[:-1], limits), reverse=True
    )
    midpoint = jnp.concatenate((inner, midpoint[-1:]))
    left = midpoint - half_width[:, None]
    right = midpoint + half_width[:, None]
    shuffled_x = jnp.concatenate((left[::-1], right[1:]), axis=0).T
    shuffled_field = jnp.concatenate((levels[::-1], levels[1:]))
    target = jax.vmap(lambda grid: jnp.interp(output_x, grid, shuffled_field))(
        shuffled_x
    )
    original = jax.vmap(lambda line: jnp.interp(output_x, x, line))(field)
    return jnp.mean((target - original) ** 2)


def residual(spectrum, nfp, settings, axis_toroidal_harmonic):
    """Sum of per-surface mean squared residuals in normalized field units."""
    x = jnp.linspace(0, 1, settings.n_toroidal_points)
    levels = jnp.linspace(0, 1, settings.n_field_contours)
    output_x = jnp.linspace(0, 1, settings.n_output_toroidal_points)
    alpha = jnp.linspace(0, 2 * jnp.pi, settings.n_field_lines)
    phi = x * (2 * jnp.pi / nfp)
    # Select the stellarator-symmetry plane with the stronger axis |B|, as
    # in the original construction. The convention switches at zero harmonic.
    start = jnp.where(axis_toroidal_harmonic < 0, jnp.pi / nfp, 0)

    def one_surface(coefficients, iota):
        poloidal = alpha[:, None] * spectrum.xm_b
        toroidal = phi[:, None] * (iota * spectrum.xm_b - spectrum.xn_b)
        toroidal -= start * spectrum.xn_b
        field = (jnp.cos(poloidal) * coefficients) @ jnp.cos(toroidal).T
        field -= (jnp.sin(poloidal) * coefficients) @ jnp.sin(toroidal).T
        return _well_residual(field, levels, x, output_x)

    return jax.vmap(one_surface)(spectrum.bmnc_b.T, spectrum.iota).sum()

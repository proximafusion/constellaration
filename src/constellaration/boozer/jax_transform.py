"""Adapt VMEC's staggered radial grids to the functional JAX Boozer API."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from booz_xform_jax.jax_api import (
    BoozerConfig,
    booz_xform_jax_impl,
    prepare_booz_xform_plan,
)


class BoozerSpectrum(NamedTuple):
    """Selected half-grid surfaces. Spectra are indexed (Fourier mode, surface)."""

    xm_b: jax.Array
    xn_b: jax.Array
    s: jax.Array
    iota: jax.Array
    bmnc_b: jax.Array
    bmns_b: jax.Array | None


def make_transform(
    modes: Mapping[str, jax.typing.ArrayLike],
    *,
    nfp: int,
    ns: int,
    mboz: int,
    nboz: int,
    surfaces: Sequence[float] | None,
    asym: bool = False,
    magnetic_only: bool = False,
) -> Callable[[Mapping[str, jax.Array]], tuple[BoozerSpectrum, dict]]:
    """Prepare the quadrature and static surface selection outside tracing."""
    mode_arrays = {
        name: jnp.asarray(modes[name]) for name in ("xm", "xn", "xm_nyq", "xn_nyq")
    }
    plan = prepare_booz_xform_plan(
        nfp=nfp,
        asym=asym,
        xm=np.asarray(modes["xm"]).tolist(),
        xn=np.asarray(modes["xn"]).tolist(),
        xm_nyq=np.asarray(modes["xm_nyq"]).tolist(),
        xn_nyq=np.asarray(modes["xn_nyq"]).tolist(),
        config=BoozerConfig(mboz=mboz, nboz=nboz, trig_f32=False),
    )
    s_full = np.linspace(0, 1, ns)
    s_half = (s_full[1:] + s_full[:-1]) / 2
    selected = (
        np.arange(ns - 1)
        if surfaces is None
        else np.unique([np.argmin(np.abs(s_half - s)) for s in surfaces])
    )
    s = jnp.asarray(s_half[selected])
    indices = jnp.asarray(selected)
    rho = jnp.asarray(np.sqrt(s_full)).at[0].set(1)
    rho_half = jnp.asarray(np.sqrt(s_half))
    odd = mode_arrays["xm"] % 2 != 0

    def half_geometry(coefficients):
        c = jnp.asarray(coefficients).T
        regular = c / rho[:, None]
        mean = (c[:-1] + c[1:]) / 2
        odd_mean = (regular[:-1] + regular[1:]) * (rho_half[:, None] / 2)
        half = jnp.where(odd, odd_mean, mean)
        # VMEC's m=1 coefficients are linear in rho at the axis; extrapolate
        # c/rho using the two first interior points, as in BOOZ_XFORM.
        first = rho_half[0] * (1.5 * regular[1] - 0.5 * regular[2])
        return half.at[0].set(jnp.where(mode_arrays["xm"] == 1, first, half[0]))

    def transform(wout):
        fields: dict[str, Any] = {
            name: half_geometry(wout[name]) for name in ("rmnc", "zmns")
        }
        half_names = ["lmns", "bmnc", "bsubumnc", "bsubvmnc"]
        if asym:
            fields.update(
                {name: half_geometry(wout[name]) for name in ("rmns", "zmnc")}
            )
            half_names += ["lmnc", "bmns", "bsubumns", "bsubvmns"]
        fields.update({name: jnp.asarray(wout[name]).T[1:] for name in half_names})
        out = booz_xform_jax_impl(
            **fields,
            iota=jnp.asarray(wout["iotas"])[1:],
            xm=mode_arrays["xm"],
            xn=mode_arrays["xn"],
            xm_nyq=mode_arrays["xm_nyq"],
            xn_nyq=mode_arrays["xn_nyq"],
            constants=plan.constants,
            grids=plan.grids,
            surface_indices=indices,
            plan=plan,
            magnetic_only=magnetic_only,
        )
        return (
            BoozerSpectrum(
                out["ixm_b"],
                out["ixn_b"],
                s,
                out["iota_b"],
                out["bmnc_b"].T,
                out["bmns_b"].T if asym else None,
            ),
            out if magnetic_only else {**out, **fields},
        )

    return transform

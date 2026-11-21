"""Fixed-boundary VMEC++ equilibrium and output stage for use under JAX."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import vmecpp
from vmecpp import autodiff, autodiff_wout

# Only fields used by the ConStellaration metrics and Boozer transform. The
# exponents restore physical units after a solve at the template's edge flux.
_FIELD_POWER = {
    "rmnc": 0,
    "zmns": 0,
    "lmns": 0,
    "lmns_full": 0,
    "raxis_cc": 0,
    "zaxis_cs": 0,
    "iotaf": 0,
    "iotas": 0,
    "aspect": 0,
    "Aminor_p": 0,
    "vp": 0,
    "gmnc": 0,
    "beta_vol": 0,
    "betatotal": 0,
    "betaxis": 0,
    "bmnc": 1,
    "bsubumnc": 1,
    "bsubvmnc": 1,
    "bsubsmns": 1,
    "bsupumnc": 1,
    "bsupvmnc": 1,
    "phi": 1,
    "phipf": 1,
    "pres": 2,
    "DMerc": -2,
}


def resize_boundary(boundary: jax.Array, mpol: int, ntor: int) -> jax.Array:
    """Resize stacked RBC/ZBS, keeping toroidal mode zero in the center."""
    old_ntor = (boundary.shape[-1] - 1) // 2
    m = min(mpol, boundary.shape[-2])
    n = min(ntor, old_ntor)
    clipped = boundary[:, :m, old_ntor - n : old_ntor + n + 1]
    return jnp.pad(clipped, ((0, 0), (0, mpol - m), (ntor - n, ntor - n)))


def make_equilibrium_solver(
    template: vmecpp.VmecInput,
) -> Callable[[jax.Array, jax.Array, jax.Array], dict[str, jax.Array]]:
    """Prepare a float64 zero-current, fixed-boundary differentiable solve.

    The closure accepts stacked RBC/ZBS, pressure in pascals (ascending
    power-series coefficients), and signed edge toroidal flux. Resolution is
    fixed by the input template, not inferred from a traced boundary.

    VMEC's exact VJP supports pressure profiles but not edge flux itself.
    For zero toroidal current the MHD equation is invariant under simultaneous
    scaling of B by k and pressure by k². Solve with the template's flux and
    p/k², then restore dimensional outputs, to differentiate through both.
    """
    if jnp.asarray(0.0).dtype != jnp.float64:
        raise ValueError("Differentiable VMEC requires JAX_ENABLE_X64=1")
    mpol, ntor = template.mpol, template.ntor
    if not isinstance(mpol, int) or not isinstance(ntor, int):
        raise ValueError("Differentiable VMEC requires a fixed Fourier resolution")
    if (
        template.ncurr != 1
        or template.curtor != 0
        or template.pmass_type != "power_series"
        or template.gamma != 0
        or template.signgs != -1
        or template.phiedge == 0
    ):
        raise ValueError("Expected zero-current, left-handed power-series VMEC input")
    solver = autodiff.make_solver(template)

    def solve(boundary, pressure, edge_flux):
        boundary = resize_boundary(jnp.asarray(boundary, dtype=jnp.float64), mpol, ntor)
        scale = jnp.asarray(edge_flux, dtype=jnp.float64) / template.phiedge
        profiles = {
            "am": jnp.asarray(pressure, dtype=jnp.float64) / scale**2,
            "pres_scale": jnp.asarray(1.0, dtype=jnp.float64),
        }
        geometry = solver(boundary, profiles)
        mass, _, _ = autodiff_wout.half_grid_profiles(template, solver.ns, profiles)
        wout = autodiff_wout.wout_quantities(geometry, template, mass_half=mass)
        result = {}
        for name, power in _FIELD_POWER.items():
            value = wout[name]
            assert value is not None
            result[name] = value * scale**power
        return result

    return solve

"""Reference and directional-derivative checks shared by the metric tests."""

from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import vmecpp
from vmecpp import autodiff_wout

from constellaration.geometry import surface_utils
from constellaration.mhd import differentiable_vmec
from tests.equilibrium_reference_test import REFERENCES, reference_inputs
from tests.mhd.differentiable_vmec_test import finite_beta_input


def grid(mpol, ntor, nfp):
    (
        ntheta,
        nphi,
    ) = surface_utils.n_poloidal_toroidal_points_to_satisfy_nyquist_criterion(
        mpol, ntor
    )
    return jnp.moveaxis(
        jnp.asarray(
            surface_utils.make_theta_phi_grid(
                ntheta, nphi, np.pi / nfp, include_endpoints=True
            )
        ),
        -1,
        0,
    )


def pinned_case(name):
    root = REFERENCES / name
    boundary, _, settings = reference_inputs(name)
    loaded = vmecpp.VmecWOut.from_wout_file(root / "wout.nc")
    wout = dict(loaded)
    arrays = {
        k: jnp.asarray(v) for k, v in wout.items() if isinstance(v, (np.ndarray, float))
    }
    return boundary, settings, wout, arrays


@contextmanager
def resolved_case():
    """Yield one finite-beta family with pressure and boundary perturbations."""
    with vmecpp.enable_x64(True):
        template = finite_beta_input()
        solve = differentiable_vmec.make_equilibrium_solver(template)
        modes = {
            k: jnp.asarray(v)
            for k, v in autodiff_wout.static_fields(template).items()
            if k in ("xm", "xn", "xm_nyq", "xn_nyq")
        }
        b = jnp.stack([template.rbc, template.zbs])
        p = jnp.array([3000.0, -3450.0, 450.0])

        def family(t):
            b_t = b.at[0, 1, template.ntor].add(0.06 * t)
            return b_t, solve(b_t, p * (1 + 0.2 * t), template.phiedge * (1 + 0.1 * t))

        yield template, family, modes


def assert_directional_derivative(
    function,
    *,
    steps=(2e-3, 1e-3),
    atol=2e-7,
    rtol=1e-3,
):
    """Test local autodiff against two independent central re-solves."""
    f = jax.jit(function)
    result = f(0.0)
    derivative = jax.jit(jax.jacrev(function))(0.0)
    assert np.all(np.isfinite(result))
    assert np.all(np.isfinite(derivative))
    for step in steps:
        fd = (f(step) - f(-step)) / (2 * step)
        np.testing.assert_allclose(derivative, fd, rtol=rtol, atol=atol)

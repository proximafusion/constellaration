import jax
import jax.numpy as jnp
import numpy as np
import pytest
import vmecpp

from constellaration.mhd import differentiable_vmec, vmec_settings, vmec_utils
from tests.equilibrium_reference_test import NAMES, REFERENCES, reference_inputs


def finite_beta_input():
    boundary, mhd, _ = reference_inputs("finite_beta")
    settings = vmec_settings.VmecSettings(
        n_poloidal_modes=4,
        max_toroidal_mode=3,
        n_poloidal_grid_points=20,
        n_toroidal_grid_points=18,
        multigrid_steps=[
            vmec_settings.VmecMultiGridStep(
                n_flux_surfaces=ns, force_tolerance=1e-19, n_max_iterations=18000
            )
            for ns in (15, 29)
        ],
        time_step=0.5,
    )
    return vmec_utils.build_vmecpp_indata(mhd, boundary, settings)


@pytest.mark.parametrize("name", NAMES)
def test_solved_equilibrium_against_pinned_wout(name):
    boundary, mhd, settings = reference_inputs(name)
    resolution = vmec_settings.create_vmec_settings_from_preset(
        boundary, settings.vmec_preset_settings
    )
    template = vmec_utils.build_vmecpp_indata(mhd, boundary, resolution)
    legacy = vmecpp.VmecWOut.from_wout_file(REFERENCES / name / "wout.nc")
    with vmecpp.enable_x64(True):
        solve = differentiable_vmec.make_equilibrium_solver(template)

        @jax.jit
        def evaluate(b):
            out = solve(
                b, jnp.asarray(mhd.pressure.coefficients), jnp.asarray(template.phiedge)
            )
            return (
                out["aspect"],
                out["phi"],
                out["betatotal"],
                out["bmnc"][0, 1:],
                out["iotaf"],
            )

        aspect, flux, beta, b00, iota = evaluate(
            jnp.stack((template.rbc, template.zbs))
        )
    np.testing.assert_allclose(aspect, legacy.aspect, rtol=1e-10)
    np.testing.assert_allclose(flux, legacy.phi, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(beta, legacy.betatotal, rtol=3e-5)
    np.testing.assert_allclose(b00, legacy.bmnc[0, 1:], rtol=3e-4)
    # The old VMEC++ version ran with a free m=1 gauge. At finite resolution
    # pinned-gauge iota differs slightly, most visibly at the finite-beta edge.
    np.testing.assert_allclose(iota, legacy.iotaf, rtol=5e-3)


def test_unconverged_equilibrium_is_not_a_penalty():
    template = finite_beta_input().model_copy(
        update={
            "ns_array": np.array([15]),
            "niter_array": np.array([1]),
            "ftol_array": np.array([1e-19]),
        }
    )
    with vmecpp.enable_x64(True):
        solve = differentiable_vmec.make_equilibrium_solver(template)
        boundary = jnp.stack((template.rbc, template.zbs))
        with pytest.raises(RuntimeError, match="VMEC\\+\\+ did not converge"):
            jax.jit(solve)(
                boundary, jnp.array([3000.0, -3000.0]), jnp.asarray(template.phiedge)
            )["aspect"].block_until_ready()


def test_equilibrium_gradients():
    # Vary a boundary mode, pressure amplitude/shape, and the physical edge flux.
    template = finite_beta_input()
    boundary = np.stack([template.rbc, template.zbs])
    pressure = np.array([3000.0, -3450.0, 450.0])

    def parameters(t):
        b = jnp.asarray(boundary).at[0, 1, template.ntor].add(0.06 * t)
        p = pressure + t * np.array([300.0, -200.0, -100.0])
        return b, p, template.phiedge * (1 + 0.1 * t)

    keys = ("aspect", "betatotal", "betaxis", "edge_iota")

    def sample(wout):
        return jnp.stack(
            [wout[k] if k != "edge_iota" else wout["iotaf"][-1] for k in keys]
        )

    with vmecpp.enable_x64(True):
        solve = differentiable_vmec.make_equilibrium_solver(template)

        def traced(t):
            return sample(solve(*parameters(t)))

        def independently_resolved(t):
            b, p, flux = parameters(t)
            inp = template.model_copy(
                update={
                    "rbc": np.asarray(b[0]),
                    "zbs": np.asarray(b[1]),
                    "am": np.asarray(p),
                    "pres_scale": 1.0,
                    "phiedge": float(flux),
                    "always_fix_m1_gauge": True,
                }
            )
            out = vmecpp.run(inp, max_threads=1, verbose=False).wout
            return sample(
                {**{k: getattr(out, k) for k in keys[:-1]}, "iotaf": out.iotaf}
            )

        value = jax.jit(traced)(0.0)
        expected = independently_resolved(0.0)
        np.testing.assert_allclose(value, expected, rtol=2e-4, atol=2e-7)
        derivative = jax.jit(jax.jacrev(traced))(0.0)
        assert np.all(np.isfinite(derivative))
        for step in (2e-3, 1e-3):
            finite_difference = (
                independently_resolved(step) - independently_resolved(-step)
            ) / (2 * step)
            np.testing.assert_allclose(
                derivative, finite_difference, rtol=1e-3, atol=2e-7
            )

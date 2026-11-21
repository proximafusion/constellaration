import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
import vmecpp
from vmecpp import autodiff_wout

from constellaration.boozer import boozer
from constellaration.mhd import vmec_utils
from constellaration.utils import visualization
from tests.equilibrium_reference_test import NAMES, REFERENCES, reference_inputs
from tests.utils.vmec_utils_test import finite_beta_input


@pytest.mark.parametrize("name", NAMES)
def test_pinned_boozer_spectrum(name):
    root = REFERENCES / name
    _, _, settings = reference_inputs(name)
    assert settings.boozer_preset_settings is not None
    wout = vmec_utils.vmecppwout_from_wout(
        vmecpp.VmecWOut.from_wout_file(root / "wout.nc")
    )
    result = boozer.run_boozer(
        wout,
        boozer.create_boozer_settings_from_equilibrium_resolution(
            wout, settings.boozer_preset_settings
        ),
    )
    with np.load(root / "boozer.npz") as expected:
        for key in expected:
            np.testing.assert_allclose(
                getattr(result, key), expected[key], rtol=1e-9, atol=2e-11
            )


def test_plot_boozer_surfaces_uses_jax_transform(tmp_path):
    source = REFERENCES / "finite_beta" / "wout.nc"
    wout = vmec_utils.vmecppwout_from_wout(vmecpp.VmecWOut.from_wout_file(source))
    plots = visualization.plot_boozer_surfaces(
        wout,
        boozer.BoozerSettings(
            normalized_toroidal_flux=[1.0], n_poloidal_modes=12, max_toroidal_mode=5
        ),
        tmp_path,
    )
    try:
        assert len(plots) == 1
        assert plots[0].axes[0].get_title().startswith("|B| on surface s=")
        assert (tmp_path / "surface_plot_0.png").stat().st_size > 0
    finally:
        for fig in plots:
            plt.close(fig)


def test_boozer_gradient_through_resolved_equilibrium():
    template = finite_beta_input()
    with vmecpp.enable_x64(True):
        solve = vmec_utils.make_equilibrium_solver(template)
        transform = boozer.make_transform(
            autodiff_wout.static_fields(template),
            nfp=template.nfp,
            ns=int(template.ns_array[-1]),
            mboz=12,
            nboz=5,
            surfaces=[0.5, 1.0],
            magnetic_only=True,
        )
        b = jnp.stack([template.rbc, template.zbs])
        p = jnp.array([3000.0, -3450.0, 450.0])

        @jax.jit
        def objective(t):
            boundary = b.at[0, 1, template.ntor].add(0.06 * t)
            wout = solve(boundary, p * (1 + 0.2 * t), jnp.asarray(template.phiedge))
            spectrum, _ = transform(wout)
            return jnp.sum((spectrum.bmnc_b[1:] / spectrum.bmnc_b[:1]) ** 2)

        derivative = jax.grad(objective)(0.0)
        for step in (2e-3, 1e-3):
            fd = (objective(step) - objective(-step)) / (2 * step)
            np.testing.assert_allclose(derivative, fd, rtol=1e-3, atol=2e-7)

import json

import jax.numpy as jnp
import numpy as np
import pytest
import vmecpp

from constellaration import forward_model, initial_guess
from constellaration.mhd import ideal_mhd_parameters, magnetics_utils, vmec_utils
from tests.equilibrium_reference_test import NAMES, REFERENCES
from tests.utils.vmec_utils_test import (
    assert_directional_derivative,
    grid,
    pinned_case,
    resolved_case,
)


def test_mercier_softmin():
    equilibrium = vmec_utils.VmecppWOut.model_construct(
        DMerc=np.array([-100.0, 2, 3, -100]), phi=np.array([0.0, 2.0])
    )
    value = magnetics_utils.minimum_normalized_mercier_criterion(equilibrium, 0.1)
    assert value is not None
    assert value <= 8
    assert value == pytest.approx(8, abs=1e-10)
    equilibrium.DMerc[1] = -1
    value = magnetics_utils.minimum_normalized_mercier_criterion(equilibrium, 1e-3)
    assert value is not None
    assert value < 0
    equilibrium.DMerc[1] = np.nan
    value = magnetics_utils.minimum_normalized_mercier_criterion(equilibrium, 1e-3)
    assert value is not None
    assert np.isnan(float(value))
    with pytest.raises(ValueError, match="temperature"):
        magnetics_utils.minimum_normalized_mercier_criterion(equilibrium, 0)


def test_mirror_and_beta():
    assert magnetics_utils.mirror_ratio_well(0.1, 0.11) == pytest.approx(2 / 21)
    assert np.isinf(magnetics_utils.mirror_ratio_well(0, 0))
    equilibrium = vmec_utils.VmecppWOut.model_construct(
        betatotal=0.03, beta_vol=np.array([0.0, 0.09, 0.07])
    )
    assert magnetics_utils.beta_peaking_factor(equilibrium) == pytest.approx(10 / 3)
    equilibrium.betatotal = 0
    peaking = magnetics_utils.beta_peaking_factor(equilibrium)
    assert peaking is not None
    assert np.isnan(float(peaking))
    equilibrium.betatotal = 0.03
    equilibrium.beta_vol = np.array([0.0, 0.09])
    assert magnetics_utils.beta_peaking_factor(equilibrium) is None


def test_metrics_from_real_vmec():
    boundary = initial_guess.generate_nae(
        aspect_ratio=10,
        max_elongation=5,
        rotational_transform=0.75,
        mirror_ratio=0.25,
        n_field_periods=3,
        max_poloidal_mode=1,
        max_toroidal_mode=1,
    )
    mhd = ideal_mhd_parameters.boundary_to_ideal_mhd_parameters(boundary)
    mhd.pressure.coefficients = [3000, -3000]
    settings = forward_model.ConstellarationSettings(
        qi_settings=None, boozer_preset_settings=None, turbulent_settings=None
    )
    metrics, wout = forward_model.forward_model(boundary, mhd, settings)
    assert metrics.volume_averaged_beta == pytest.approx(wout.betatotal)
    assert metrics.beta_peaking_factor == pytest.approx(
        (1.5 * wout.beta_vol[1] - 0.5 * wout.beta_vol[2]) / wout.betatotal
    )
    interior = wout.DMerc[1:-1] * wout.phi[-1] ** 2
    value = metrics.minimum_normalized_mercier_criterion
    assert value is not None
    assert value <= np.min(interior)


def _magnetic_values(equilibrium):
    mirror = magnetics_utils.magnetic_mirror_ratio(equilibrium).values
    axis, edge = mirror[0], mirror[-1]
    mercier = magnetics_utils.minimum_normalized_mercier_criterion(equilibrium, 1e-3)
    assert mercier is not None
    return jnp.stack(
        (
            axis,
            edge,
            magnetics_utils.vacuum_well(equilibrium),
            mercier,
        )
    )


@pytest.mark.parametrize("name", NAMES)
def test_magnetic_values(name):
    with vmecpp.enable_x64(True):
        equilibrium = vmec_utils.VmecppWOut.model_construct(**pinned_case(name)[2])
        actual = _magnetic_values(equilibrium)
    expected = json.loads((REFERENCES / name / "metrics.json").read_text())
    np.testing.assert_allclose(
        actual,
        [
            expected[key]
            for key in (
                "axis_magnetic_mirror_ratio",
                "edge_magnetic_mirror_ratio",
                "vacuum_well",
                "minimum_normalized_mercier_criterion",
            )
        ],
        rtol=1e-10,
        atol=1e-11,
    )


def test_magnetic_gradients():
    with resolved_case() as (template, family, modes):

        def metric(t):
            _, out = family(t)
            return _magnetic_values(
                vmec_utils.VmecppWOut.model_construct(
                    _fields_set=None,
                    **out,
                    **modes,
                    nfp=template.nfp,
                    lasym=False,
                    mpol=template.mpol,
                    ntor=template.ntor,
                )
            )

        assert_directional_derivative(metric)


@pytest.mark.parametrize("name", NAMES)
def test_gradient_length_value(name):
    with vmecpp.enable_x64(True):
        equilibrium = vmec_utils.VmecppWOut.model_construct(**pinned_case(name)[2])
        theta, phi = grid(equilibrium.mpol, equilibrium.ntor, equilibrium.nfp)
        actual = (
            magnetics_utils.normalized_magnetic_gradient_scale_length(
                equilibrium, jnp.stack((theta, phi), axis=-1)
            ).min()
            * equilibrium.nfp
        )
    expected = json.loads((REFERENCES / name / "metrics.json").read_text())
    np.testing.assert_allclose(
        actual, expected["minimum_normalized_magnetic_gradient_scale_length"], rtol=1e-9
    )


def test_gradient_length_derivative():
    with resolved_case() as (template, family, modes):
        theta, phi = grid(template.mpol, template.ntor, template.nfp)

        def metric(t):
            _, out = family(t)
            return (
                magnetics_utils.normalized_magnetic_gradient_scale_length(
                    vmec_utils.VmecppWOut.model_construct(
                        _fields_set=None, **out, **modes
                    ),
                    jnp.stack((theta, phi), axis=-1),
                ).min()
                * template.nfp
            )

        assert_directional_derivative(metric)

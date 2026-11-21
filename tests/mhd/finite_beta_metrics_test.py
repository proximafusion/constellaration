import numpy as np
import pytest

from constellaration import forward_model, initial_guess
from constellaration.mhd import ideal_mhd_parameters, magnetics_utils, vmec_utils


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
    assert (
        magnetics_utils.minimum_normalized_mercier_criterion(equilibrium, 1e-3) is None
    )
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
    assert magnetics_utils.beta_peaking_factor(equilibrium) is None
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

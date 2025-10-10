import numpy as np

from constellaration import initial_guess
from constellaration.mhd import ideal_mhd_parameters_utils


def test_get_ideal_mhd_parameters_for_vacuum_beta():
    boundary = initial_guess.generate_rotating_ellipse(
        aspect_ratio=6.0,
        elongation=1.5,
        rotational_transform=0.4,
        n_field_periods=4,
    )
    volume_averaged_beta = 0.0

    mhd_parameters = (
        ideal_mhd_parameters_utils.get_ideal_mhd_parameters_for_volume_averaged_beta(
            boundary=boundary,
            volume_averaged_beta=volume_averaged_beta,
        )
    )

    # Check that pressure coefficients are normalized
    assert mhd_parameters.pressure.coefficients[0] == 1.0
    assert np.allclose(mhd_parameters.boundary_toroidal_flux, 0.09, rtol=0.1)


def test_get_ideal_mhd_parameters_for_finite_beta():
    boundary = initial_guess.generate_rotating_ellipse(
        aspect_ratio=6.0,
        elongation=1.5,
        rotational_transform=0.4,
        n_field_periods=4,
    )
    volume_averaged_beta = 0.02

    mhd_parameters = (
        ideal_mhd_parameters_utils.get_ideal_mhd_parameters_for_volume_averaged_beta(
            boundary=boundary,
            volume_averaged_beta=volume_averaged_beta,
        )
    )

    # Check that pressure profile is scaled appropriately
    assert np.allclose(mhd_parameters.pressure.coefficients[0], 18000, rtol=0.1)
    assert np.allclose(mhd_parameters.boundary_toroidal_flux, 0.09, rtol=0.1)

import jax.numpy as jnp
import numpy as np
import pytest

from constellaration.omnigeneity import omnigenity_field


@pytest.fixture
def omnigenous_field_constant_rho() -> omnigenity_field.OmnigenousField:
    x_lmn = np.array(
        [
            [
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
                [1.69468503e-01, -1.04972797e-05, 3.28821224e-01],
            ],
            [
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
                [9.78318700e-02, 1.16823997e-05, 3.28696034e-01],
            ],
        ],
    )

    modB_spline_knot_coefficients = np.array(
        [
            [0.8, 0.9, 1.1, 1.2],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    nfp = 3
    return omnigenity_field.OmnigenousField(
        n_field_periods=nfp,
        poloidal_winding=0,
        torodial_winding=nfp,
        x_lmn=x_lmn,
        modB_spline_knot_coefficients=modB_spline_knot_coefficients,
    )


@pytest.fixture
def omnigenous_field_varying_rho() -> omnigenity_field.OmnigenousField:
    x_lmn = np.array(
        [
            [
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
                [1.69468503e-01, -1.04972797e-05, 3.28821224e-01],
            ],
            [
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
                [9.78318700e-02, 1.16823997e-05, 3.28696034e-01],
            ],
        ],
    )

    modB_spline_knot_coefficients = np.array(
        [
            [0.7554605, 1.62107776, 1.17689637],
            [-0.0445395, 0.00637115, -0.02310363],
        ]
    )
    nfp = 3
    return omnigenity_field.OmnigenousField(
        n_field_periods=nfp,
        poloidal_winding=0,
        torodial_winding=nfp,
        x_lmn=x_lmn,
        modB_spline_knot_coefficients=modB_spline_knot_coefficients,
    )


@pytest.fixture
def stell_symmetric_omnigenous_field() -> omnigenity_field.OmnigenousField:
    n_x_rho_coefficients = 1
    n_x_eta_coefficinets = 2
    n_x_alpha_coefficients = 5

    seed = 7
    rng = np.random.default_rng(seed)

    x_lmn = rng.standard_normal(
        (
            n_x_rho_coefficients,
            n_x_eta_coefficinets,
            n_x_alpha_coefficients,
        )
    )
    x_lmn[:, :, n_x_alpha_coefficients // 2 :] = 0
    x_lmn[:, ::2, :] = 0

    modB_spline_knot_coefficients = np.array([[0.8, 0.9, 1.1, 1.2], [0, 0, 0, 0]])

    nfp = 4
    return omnigenity_field.OmnigenousField(
        n_field_periods=nfp,
        poloidal_winding=0,
        torodial_winding=nfp,
        x_lmn=x_lmn,
        modB_spline_knot_coefficients=modB_spline_knot_coefficients,
    )


def test_compute_magnetic_well_at_rho_eta_constant_rho(omnigenous_field_constant_rho):
    field = omnigenous_field_constant_rho
    eta = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 10, endpoint=False)
    actual_B = omnigenity_field._compute_magnetic_well_at_rho_eta(
        field,
        rho=jnp.asarray([0.0, 0.25, 1.0]),  # length 3
        eta=eta,
    )

    expected_shape = (3, 10)
    assert expected_shape == actual_B.shape

    # All wells across values of rhos are different
    assert np.array_equal(actual_B[0, :], actual_B[1, :])
    assert np.array_equal(actual_B[1, :], actual_B[2, :])
    assert np.array_equal(actual_B[2, :], actual_B[3, :])
    assert np.array_equal(actual_B[3, :], actual_B[4, :])


def test_compute_magnetic_well_at_rho_eta_varying_rho(omnigenous_field_varying_rho):
    field = omnigenous_field_varying_rho
    eta = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 4, endpoint=False)
    actual_B = omnigenity_field._compute_magnetic_well_at_rho_eta(
        field,
        rho=jnp.asarray([0.0, 0.25, 0.5, 0.75, 1.0]),  # length 5
        eta=eta,
    )

    expected_shape = (5, 4)
    assert expected_shape == actual_B.shape

    # All wells across values of rhos are different
    assert not np.array_equal(actual_B[0, :], actual_B[1, :])
    assert not np.array_equal(actual_B[1, :], actual_B[2, :])
    assert not np.array_equal(actual_B[2, :], actual_B[3, :])
    assert not np.array_equal(actual_B[3, :], actual_B[4, :])


def test_find_modb_from_theta_phi_boozer(stell_symmetric_omnigenous_field):
    field = stell_symmetric_omnigenous_field
    rho = 1.0
    # Expected by running the forward function to get mob
    expected_modb = omnigenity_field.get_modb_boozer(
        field, rho=rho, n_alpha=10, n_eta=10
    )

    # Get theta and phi boozer
    theta_b, phi_b = omnigenity_field.get_theta_and_phi_boozer(
        field, n_alpha=10, n_eta=10
    )

    # Estimate modb at theta and phi boozer
    computed_modb = omnigenity_field.find_modb_at_theta_phi_boozer(
        field,
        rho=rho,
        theta_b=theta_b,
        phi_b=phi_b,
    )

    assert np.allclose(computed_modb, expected_modb)

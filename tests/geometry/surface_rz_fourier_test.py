import numpy as np

from constellaration.geometry import surface_rz_fourier


def test_infinity_norm_scaling_basic():
    poloidal_modes = np.array([0, 1, 2, 3])
    toroidal_modes = np.array([0, 1, 2, 3])
    alpha = 1.0
    expected = np.exp(-1.0 * alpha * np.array([0, 1, 2, 3]))
    result = surface_rz_fourier.compute_infinity_norm_spectrum_scaling_fun(
        poloidal_modes, toroidal_modes, alpha
    )
    np.testing.assert_allclose(result, expected)


def test_infinity_norm_scaling_negative_modes():
    toroidal_modes = np.array([-2, -1, 0, 1, 2])
    poloidal_modes = np.array([0, 0, 0, 0, 0])
    alpha = 0.5
    expected = np.exp(-0.5 * np.abs(toroidal_modes))
    result = surface_rz_fourier.compute_infinity_norm_spectrum_scaling_fun(
        poloidal_modes, toroidal_modes, alpha
    )
    np.testing.assert_allclose(result, expected)


def test_infinity_norm_scaling_mixed_modes():
    poloidal_modes = np.array([1, 2, 3, 4])
    toroidal_modes = np.array([4, 3, 2, 1])
    alpha = 2.0
    # max(|m|, |n|) = [4, 3, 3, 4]
    expected = np.exp(-2.0 * np.array([4, 3, 3, 4]))
    result = surface_rz_fourier.compute_infinity_norm_spectrum_scaling_fun(
        poloidal_modes, toroidal_modes, alpha
    )
    np.testing.assert_allclose(result, expected)


def test_infinity_norm_scaling_zero_alpha():
    poloidal_modes = np.array([0, 1, 2, 3])
    toroidal_modes = np.array([3, 2, 1, 0])
    alpha = 0.0
    expected = np.ones_like(poloidal_modes, dtype=float)
    result = surface_rz_fourier.compute_infinity_norm_spectrum_scaling_fun(
        poloidal_modes, toroidal_modes, alpha
    )
    np.testing.assert_allclose(result, expected)


def test_infinity_norm_scaling_large_alpha():
    poloidal_modes = np.array([0, 1, 2])
    toroidal_modes = np.array([2, 1, 0])
    alpha = 100.0
    expected = np.exp(-100.0 * np.array([2, 1, 2]))
    result = surface_rz_fourier.compute_infinity_norm_spectrum_scaling_fun(
        poloidal_modes, toroidal_modes, alpha
    )
    np.testing.assert_allclose(result, expected)


def test_infinity_norm_scaling_empty():
    poloidal_modes = np.array([], dtype=int)
    toroidal_modes = np.array([], dtype=int)
    alpha = 1.0
    expected = np.array([])
    result = surface_rz_fourier.compute_infinity_norm_spectrum_scaling_fun(
        poloidal_modes, toroidal_modes, alpha
    )
    np.testing.assert_allclose(result, expected)


def test_infinity_norm_scaling_varied_signs():
    poloidal_modes = np.array([-3, 0, 3])
    toroidal_modes = np.array([0, -3, 3])
    alpha = 0.7
    expected = np.exp(-0.7 * np.array([3, 3, 3]))
    result = surface_rz_fourier.compute_infinity_norm_spectrum_scaling_fun(
        poloidal_modes, toroidal_modes, alpha
    )
    np.testing.assert_allclose(result, expected)

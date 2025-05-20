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


def test_mask_basic():
    surface = _make_surface(3, 5)  # m=0,1,2; n=-2,-1,0,1,2
    mask = surface_rz_fourier.build_mask(
        surface, max_poloidal_mode=2, max_toroidal_mode=2
    )
    # m=0, n>=1; m>0, |n|<=2
    expected = np.zeros((3, 5), dtype=bool)
    # m=0, n=1,2
    expected[0, 3:] = True
    # m=1,2, all n
    expected[1:, :] = True
    assert np.array_equal(mask.r_cos, expected)
    assert np.array_equal(mask.z_sin, expected)


def test_mask_toroidal_limit():
    surface = _make_surface(4, 7)  # m=0..3, n=-3..3
    mask = surface_rz_fourier.build_mask(
        surface, max_poloidal_mode=3, max_toroidal_mode=1
    )
    expected = np.zeros((4, 7), dtype=bool)
    # m=0, n=1
    expected[0, 4] = True
    # m=1..3, n=-1,0,1 (indices 2,3,4)
    expected[1:, 2:5] = True
    assert np.array_equal(mask.r_cos, expected)


def test_mask_poloidal_limit():
    surface = _make_surface(2, 3)  # m=0,1; n=-1,0,1
    mask = surface_rz_fourier.build_mask(
        surface, max_poloidal_mode=0, max_toroidal_mode=1
    )
    expected = np.zeros((2, 3), dtype=bool)
    # m=0, n=1
    expected[0, 2] = True
    assert np.array_equal(mask.r_cos, expected)
    assert np.array_equal(mask.z_sin, expected)


def test_mask_all_modes():
    surface = _make_surface(2, 3)
    mask = surface_rz_fourier.build_mask(
        surface, max_poloidal_mode=10, max_toroidal_mode=10
    )
    # All m>0 and m=0, n>=1
    expected = np.zeros((2, 3), dtype=bool)
    expected[0, 2] = True
    expected[1, :] = True
    assert np.array_equal(mask.r_cos, expected)


def test_mask_none():
    surface = _make_surface(2, 3)
    mask = surface_rz_fourier.build_mask(
        surface, max_poloidal_mode=0, max_toroidal_mode=0
    )
    expected = np.zeros((2, 3), dtype=bool)
    assert np.array_equal(mask.r_cos, expected)


def test_mask_shape_consistency():
    # Check that mask shape matches input
    for mpol, ntor in [(1, 1), (2, 3), (4, 5)]:
        surface = _make_surface(mpol, ntor)
        mask = surface_rz_fourier.build_mask(
            surface, max_poloidal_mode=mpol - 1, max_toroidal_mode=(ntor - 1) // 2
        )
        assert mask.r_cos.shape == (mpol, ntor)
        assert mask.z_sin.shape == (mpol, ntor)


def _make_surface(
    n_poloidal_modes: int, n_toroidal_modes: int
) -> surface_rz_fourier.SurfaceRZFourier:
    # mpol: number of poloidal modes, ntor: number of toroidal modes (must be odd)
    shape = (n_poloidal_modes, n_toroidal_modes)
    r_cos = np.zeros(shape)
    z_sin = np.zeros(shape)
    return surface_rz_fourier.SurfaceRZFourier(r_cos=r_cos, z_sin=z_sin)

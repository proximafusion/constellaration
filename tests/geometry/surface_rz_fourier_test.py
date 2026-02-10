import dataclasses

import jaxtyping as jt
import numpy as np
import pytest

from constellaration.geometry import surface_rz_fourier


@pytest.fixture
def two_mode_surface() -> surface_rz_fourier.SurfaceRZFourier:
    r_cos = np.array(
        [
            [0.0, 0.0, 1.17402691, 0.29914774, 0.01072749],
            [0.00175617, -0.00545333, 0.14967022, -0.0997095, -0.01166579],
            [0.00274853, -0.00707297, 0.01615058, 0.03462255, 0.01252142],
        ]
    )
    z_sin = np.array(
        [
            [0.0, 0.0, 0.0, -0.24121311, -0.01337604],
            [0.00701798, -0.02070752, 0.26476756, 0.13379441, 0.00483678],
            [0.00229368, -0.01336017, 0.02373038, 0.026118, -0.00429497],
        ]
    )
    return surface_rz_fourier.SurfaceRZFourier(
        r_cos=r_cos, z_sin=z_sin, n_field_periods=3, is_stellarator_symmetric=True
    )


@pytest.fixture
def three_mode_surface() -> surface_rz_fourier.SurfaceRZFourier:
    r_cos = np.array(
        [
            [0.0, 0.0, 0.0, 1.17402691, 0.29914774, 0.01072749, 0.001],
            [
                0.001,
                0.00175617,
                -0.00545333,
                0.14967022,
                -0.0997095,
                -0.01166579,
                0.001,
            ],
            [0.001, 0.00274853, -0.00707297, 0.01615058, 0.03462255, 0.01252142, 0.001],
            [0.001, 0.00027485, -0.00070797, 0.00161058, 0.00346255, 0.00252142, 0.001],
        ]
    )
    z_sin = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, -0.24121311, -0.01337604, -0.001],
            [
                0.001,
                0.00701798,
                -0.02070752,
                0.26476756,
                0.13379441,
                0.00483678,
                -0.002,
            ],
            [0.001, 0.00229368, -0.01336017, 0.02373038, 0.026118, -0.00429497, 0.002],
            [
                -0.001,
                0.00027485,
                -0.00070797,
                0.00161058,
                0.00346255,
                0.00252142,
                0.002,
            ],
        ]
    )
    return surface_rz_fourier.SurfaceRZFourier(
        r_cos=r_cos, z_sin=z_sin, n_field_periods=3, is_stellarator_symmetric=True
    )


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


def test_get_named_mode_values_stellarator_symmetric():
    """Test get_named_mode_values with a stellarator symmetric surface."""
    # Create a SurfaceRZFourier object with stellarator symmetry
    r_cos = np.array(
        [
            [0.0, 1.0, 0.5],
            [0.0, 0.2, 0.1],
            [0.0, 0.05, 0.02],
        ]
    )
    z_sin = np.array(
        [
            [0.0, 0.0, 0.1],
            [0.0, 0.2, 0.05],
            [0.0, 0.03, 0.01],
        ]
    )

    # Ensure coefficients satisfy stellarator symmetry constraints
    # For m=0 and n<0, r_cos[0, n<0] must be zero (already zero)
    # For m=0 and n<=0, z_sin[0, n<=0] must be zero (already zero)

    surface = surface_rz_fourier.SurfaceRZFourier(
        r_cos=r_cos,
        z_sin=z_sin,
        n_field_periods=2,
        is_stellarator_symmetric=True,
    )

    mode_values = surface_rz_fourier.get_named_mode_values(surface)

    expected_mode_values = {
        # r_cos modes (excluding m=0, n<0)
        "r_cos(0, 0)": 1.0,
        "r_cos(0, 1)": 0.5,
        "r_cos(1, -1)": 0.0,
        "r_cos(1, 0)": 0.2,
        "r_cos(1, 1)": 0.1,
        "r_cos(2, -1)": 0.0,
        "r_cos(2, 0)": 0.05,
        "r_cos(2, 1)": 0.02,
        # z_sin modes (excluding m=0, n<=0)
        "z_sin(0, 1)": 0.1,
        "z_sin(1, -1)": 0.0,
        "z_sin(1, 0)": 0.2,
        "z_sin(1, 1)": 0.05,
        "z_sin(2, -1)": 0.0,
        "z_sin(2, 0)": 0.03,
        "z_sin(2, 1)": 0.01,
    }

    assert mode_values == expected_mode_values


def test_get_named_mode_values_zero_coefficients():
    """Test get_named_mode_values with zero coefficients."""
    # Create a SurfaceRZFourier object with zero coefficients
    r_cos = np.zeros((2, 3))  # Shape (2, 3)
    z_sin = np.zeros((2, 3))  # Shape (2, 3)

    # Ensure coefficients satisfy stellarator symmetry constraints
    # r_cos[0, n<0] and z_sin[0, n<=0] are zeros

    surface = surface_rz_fourier.SurfaceRZFourier(
        r_cos=r_cos,
        z_sin=z_sin,
        n_field_periods=1,
        is_stellarator_symmetric=True,
    )

    mode_values = surface_rz_fourier.get_named_mode_values(surface)

    expected_mode_values = {
        "r_cos(0, 0)": 0.0,
        "r_cos(0, 1)": 0.0,
        "r_cos(1, -1)": 0.0,
        "r_cos(1, 0)": 0.0,
        "r_cos(1, 1)": 0.0,
        "z_sin(0, 1)": 0.0,
        "z_sin(1, -1)": 0.0,
        "z_sin(1, 0)": 0.0,
        "z_sin(1, 1)": 0.0,
    }

    assert mode_values == expected_mode_values


def test_get_named_mode_values_single_mode():
    """Test get_named_mode_values with a single Fourier mode."""
    r_cos = np.array([[1.0]])  # Shape (1, 1)
    z_sin = np.array([[0.0]])  # Shape (1, 1)

    # Ensure coefficients satisfy stellarator symmetry constraints
    # Only m=0, n=0 mode

    surface = surface_rz_fourier.SurfaceRZFourier(
        r_cos=r_cos,
        z_sin=z_sin,
        n_field_periods=1,
        is_stellarator_symmetric=True,
    )

    mode_values = surface_rz_fourier.get_named_mode_values(surface)

    expected_mode_values = {
        "r_cos(0, 0)": 1.0,
    }

    assert mode_values == expected_mode_values


def test_get_named_mode_values_large_modes():
    """Test get_named_mode_values with larger mode indices."""
    # Coefficient arrays must satisfy stellarator symmetry constraints
    # We'll create arrays with zeros where required

    # Number of poloidal and toroidal modes
    n_poloidal_modes = 5  # m from 0 to 4
    n_toroidal_modes = 9  # n from -4 to 4 (must be odd)

    # Initialize arrays with zeros
    r_cos = np.zeros((n_poloidal_modes, n_toroidal_modes))
    z_sin = np.zeros((n_poloidal_modes, n_toroidal_modes))

    # Set random values where allowed
    max_toroidal_mode = (n_toroidal_modes - 1) // 2

    for m in range(n_poloidal_modes):
        rng = np.random.default_rng()
        for n_idx, n in enumerate(range(-max_toroidal_mode, max_toroidal_mode + 1)):
            # Skip r_cos when m=0 and n<0
            if not (m == 0 and n < 0):
                r_cos[m, n_idx] = rng.random()
            # Skip z_sin when m=0 and n<=0
            if not (m == 0 and n <= 0):
                z_sin[m, n_idx] = rng.random()

    surface = surface_rz_fourier.SurfaceRZFourier(
        r_cos=r_cos,
        z_sin=z_sin,
        n_field_periods=2,
        is_stellarator_symmetric=True,
    )

    mode_values = surface_rz_fourier.get_named_mode_values(surface)

    expected_keys = []
    for m in range(n_poloidal_modes):
        for n_idx, n in enumerate(range(-max_toroidal_mode, max_toroidal_mode + 1)):
            if not (m == 0 and n < 0):
                expected_keys.append(f"r_cos({m}, {n})")
            if not (m == 0 and n <= 0):
                expected_keys.append(f"z_sin({m}, {n})")

    assert set(mode_values.keys()) == set(expected_keys)


def test_get_named_mode_values_with_negative_toroidal_modes():
    """Test get_named_mode_values correctly handles negative toroidal modes."""
    r_cos = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.5, 0.2],
        ]
    )
    z_sin = np.array(
        [
            [0.0, 0.0, 0.1],
            [0.0, 0.05, 0.02],
        ]
    )

    # Ensure coefficients satisfy stellarator symmetry constraints
    # r_cos[0, n<0] and z_sin[0, n<=0] are zeros

    surface = surface_rz_fourier.SurfaceRZFourier(
        r_cos=r_cos,
        z_sin=z_sin,
        n_field_periods=1,
        is_stellarator_symmetric=True,
    )

    mode_values = surface_rz_fourier.get_named_mode_values(surface)

    expected_mode_values = {
        "r_cos(0, 0)": 0.0,
        "r_cos(0, 1)": 1.0,
        "r_cos(1, -1)": 0.0,
        "r_cos(1, 0)": 0.5,
        "r_cos(1, 1)": 0.2,
        "z_sin(0, 1)": 0.1,
        "z_sin(1, -1)": 0.0,
        "z_sin(1, 0)": 0.05,
        "z_sin(1, 1)": 0.02,
    }

    assert mode_values == expected_mode_values


# Tests for boundary_from_named_modes


def test_boundary_from_named_modes_stellarator_symmetric():
    """Test reconstruction of a stellarator symmetric SurfaceRZFourier object."""
    named_modes = {
        "r_cos(0, 0)": 1.0,
        "r_cos(1, 1)": 0.5,
        "z_sin(0, 1)": 0.3,
        "z_sin(1, 1)": 0.7,
    }
    n_field_periods = 2
    is_stellarator_symmetric = True

    surface = surface_rz_fourier.boundary_from_named_modes(
        named_modes, is_stellarator_symmetric, n_field_periods
    )

    assert surface.n_field_periods == n_field_periods
    assert surface.is_stellarator_symmetric
    assert np.allclose(surface.r_cos[0, 1], 1.0)  # r_cos(0, 0)
    assert np.allclose(surface.r_cos[1, 2], 0.5)  # r_cos(1, 1)
    assert np.allclose(surface.z_sin[0, 2], 0.3)  # z_sin(0, 1)
    assert np.allclose(surface.z_sin[1, 2], 0.7)  # z_sin(1, 1)
    assert surface.r_sin is None
    assert surface.z_cos is None


def test_boundary_from_named_modes_non_symmetric():
    """Test reconstruction of a non-stellarator symmetric SurfaceRZFourier object."""
    named_modes = {
        "r_cos(0, 0)": 1.0,
        "r_cos(1, -1)": 0.4,
        "r_sin(1, 1)": 0.6,
        "z_sin(1, -1)": 0.3,
        "z_cos(0, 0)": 0.9,
    }
    n_field_periods = 3
    is_stellarator_symmetric = False

    surface = surface_rz_fourier.boundary_from_named_modes(
        named_modes, is_stellarator_symmetric, n_field_periods
    )

    assert surface.n_field_periods == n_field_periods
    assert not surface.is_stellarator_symmetric
    assert np.allclose(surface.r_cos[0, 1], 1.0)  # r_cos(0, 0)
    assert np.allclose(surface.r_cos[1, 0], 0.4)  # r_cos(1, -1)
    assert surface.r_sin is not None
    assert np.allclose(surface.r_sin[1, 2], 0.6)  # r_sin(1, 1)
    assert np.allclose(surface.z_sin[1, 0], 0.3)  # z_sin(1, -1)
    assert surface.z_cos is not None
    assert np.allclose(surface.z_cos[0, 1], 0.9)  # z_cos(0, 0)


def test_boundary_from_named_modes_round_trip_get_boundary_named_modes():
    """Test round trip: Surface -> Named Modes -> Surface."""
    r_cos = np.array([[1.0, 0.0, 0.5], [0.0, 0.4, 0.0]])
    z_sin = np.array([[0.0, 0.3, 0.0], [0.0, 0.0, 0.7]])
    r_sin = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.6]])
    z_cos = np.array([[0.9, 0.0, 0.0], [0.0, 0.0, 0.0]])
    n_field_periods = 2
    is_stellarator_symmetric = False

    original_surface = surface_rz_fourier.SurfaceRZFourier(
        r_cos=r_cos,
        z_sin=z_sin,
        r_sin=r_sin,
        z_cos=z_cos,
        n_field_periods=n_field_periods,
        is_stellarator_symmetric=is_stellarator_symmetric,
    )

    named_modes = surface_rz_fourier.get_named_mode_values(original_surface)
    reconstructed_surface = surface_rz_fourier.boundary_from_named_modes(
        named_modes, is_stellarator_symmetric, n_field_periods
    )
    assert np.allclose(original_surface.r_cos, reconstructed_surface.r_cos)
    assert np.allclose(original_surface.z_sin, reconstructed_surface.z_sin)
    assert original_surface.r_sin is not None
    assert reconstructed_surface.r_sin is not None
    assert np.allclose(original_surface.r_sin, reconstructed_surface.r_sin)
    assert original_surface.z_cos is not None
    assert reconstructed_surface.z_cos is not None
    assert np.allclose(original_surface.z_cos, reconstructed_surface.z_cos)
    assert original_surface.n_field_periods == reconstructed_surface.n_field_periods
    assert (
        original_surface.is_stellarator_symmetric
        == reconstructed_surface.is_stellarator_symmetric
    )


@dataclasses.dataclass
class SpectralWidthTestCase:
    fourier_coefficients: list[
        jt.Float[np.ndarray, "n_poloidal_modes n_toroidal_modes"]
    ]
    p: int
    q: int
    expected_spectral_width: float
    normalize: bool = True


SPECTRAL_WIDTH_TEST_CASES: list[SpectralWidthTestCase] = [
    # toroidal curve (it has only m=0 coefficients)
    # For this curve, the spectral width should be 1 since the curve
    # does not have any m>1 coefficient contributions.
    SpectralWidthTestCase(
        fourier_coefficients=[np.arange(1, 4, dtype=float).reshape(1, 3)],
        p=4,
        q=1,
        expected_spectral_width=1.0,
    ),
    # mimic 1D curve with xm = [1.0, 2.0, 3.0]
    SpectralWidthTestCase(
        fourier_coefficients=[np.arange(1, 4, dtype=float).reshape(3, 1)],
        p=1,
        q=1,
        expected_spectral_width=(1 * 4 + 4 * 9) / (1 * 4 + 2 * 9),
    ),
    # same as above with p=4, q=1
    SpectralWidthTestCase(
        fourier_coefficients=[np.arange(1, 4, dtype=float).reshape(3, 1)],
        p=4,
        q=1,
        expected_spectral_width=(1 * 4 + 32 * 9) / (1 * 4 + 16 * 9),
    ),
    # same as above with p=5, q=5
    SpectralWidthTestCase(
        fourier_coefficients=[np.arange(1, 4, dtype=float).reshape(3, 1)],
        p=5,
        q=5,
        expected_spectral_width=(1 * 4 + 1024 * 9) / (1 * 4 + 32 * 9),
    ),
    # mimic 2D curve with xm = ym = [1.0, 2.0, 3.0]
    SpectralWidthTestCase(
        fourier_coefficients=[
            np.arange(1, 4, dtype=float).reshape(3, 1),
            np.arange(1, 4, dtype=float).reshape(3, 1),
        ],
        p=1,
        q=1,
        expected_spectral_width=(1 * 4 + 4 * 9) / (1 * 4 + 2 * 9),
    ),
    # same as above with p=4, q=1
    SpectralWidthTestCase(
        fourier_coefficients=[
            np.arange(1, 4, dtype=float).reshape(3, 1),
            np.arange(1, 4, dtype=float).reshape(3, 1),
        ],
        p=4,
        q=1,
        expected_spectral_width=(1 * 4 + 32 * 9) / (1 * 4 + 16 * 9),
    ),
    # mimic 2D curve of two coordinates with
    # xmn = ymn = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]].
    # For example, xmn and ymn can be rc(m, n) and zs(m, n)
    SpectralWidthTestCase(
        fourier_coefficients=[
            np.arange(1, 7, dtype=float).reshape(3, 2),
            np.arange(1, 7, dtype=float).reshape(3, 2),
        ],
        p=1,
        q=1,
        expected_spectral_width=(1 * (9 + 16) + 4 * (25 + 36))
        / (1 * (9 + 16) + 2 * (25 + 36)),
    ),
    # same as above without normalization
    SpectralWidthTestCase(
        fourier_coefficients=[
            np.arange(1, 7, dtype=float).reshape(3, 2),
            np.arange(1, 7, dtype=float).reshape(3, 2),
        ],
        p=1,
        q=1,
        normalize=False,
        expected_spectral_width=2 * (1 * (9 + 16) + 4 * (25 + 36)),
    ),
    # test cases from:
    # https://github.com/proximafusion/DESCUR/blob/master/src/python/lambdaDescur.py
    # dshape
    SpectralWidthTestCase(
        fourier_coefficients=[
            np.array([3.0, 0.991, 0.136, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(-1, 1),
            np.array([0.0, 1.409, -0.118, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(-1, 1),
        ],
        p=4,
        q=1,
        expected_spectral_width=1.1487974178461664,
    ),
    # square
    SpectralWidthTestCase(
        fourier_coefficients=[
            np.array([3.0, 0.4268, 0.0, 0.07322, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(
                -1, 1
            ),
            np.array(
                [0.0, 0.4268, 0.0, -0.07322, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ).reshape(-1, 1),
        ],
        p=4,
        q=1,
        expected_spectral_width=2.408973284653641,
    ),
    # belt
    SpectralWidthTestCase(
        fourier_coefficients=[
            np.array([3.0, 0.453, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(
                -1, 1
            ),
            np.array([0.0, 0.6, 0.0, 0.196, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(
                -1, 1
            ),
        ],
        p=4,
        q=1,
        expected_spectral_width=2.6925626307995447,
    ),
    # ellipse
    SpectralWidthTestCase(
        fourier_coefficients=[
            np.array([3.0, 1.0, 0.0, 0.0]).reshape(-1, 1),
            np.array([0.0, 0.6, 0.0, 0.0]).reshape(-1, 1),
        ],
        p=4,
        q=1,
        expected_spectral_width=1.0,
    ),
]


@pytest.mark.parametrize("test_case", SPECTRAL_WIDTH_TEST_CASES)
def test_spectral_width(test_case: SpectralWidthTestCase):
    computed_spectral_width = surface_rz_fourier.spectral_width(
        test_case.fourier_coefficients,
        p=test_case.p,
        q=test_case.q,
        normalize=test_case.normalize,
    )
    np.testing.assert_allclose(
        computed_spectral_width, test_case.expected_spectral_width, atol=0, rtol=0
    )


def test_from_points() -> None:
    named_modes = {
        "r_cos(0, 0)": 1.0,
        "r_cos(0, 1)": 0.3,
        "r_cos(1, -1)": 0.1,
        "r_cos(1, 1)": 0.05,
        "z_sin(0, 0)": 0.0,
        "z_sin(0, 1)": 0.3,
        "z_sin(1, -1)": 0.1,
    }
    n_field_periods = 2
    is_stellarator_symmetric = True
    surface = surface_rz_fourier.boundary_from_named_modes(
        named_modes, is_stellarator_symmetric, n_field_periods
    )
    thetas = np.linspace(
        0.0,
        2 * np.pi,
        7,
        endpoint=True,
    )
    phis = np.linspace(
        0.0,
        2 * np.pi,
        13,
        endpoint=True,
    )
    thetas_grid, phis_grid = np.meshgrid(thetas, phis, indexing="ij")
    theta_phi = np.stack([thetas_grid, phis_grid], axis=-1)
    fitted_surface, residual = surface_rz_fourier.from_points(
        surface_rz_fourier.evaluate_points_xyz(
            surface,
            theta_phi,
        ),
        theta_phi=theta_phi,
        n_field_periods=2,
        n_poloidal_modes=2,
        n_toroidal_modes=3,
    )
    np.testing.assert_allclose(
        fitted_surface.r_cos,
        surface.r_cos,
        atol=1e-14,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        fitted_surface.z_sin,
        surface.z_sin,
        atol=1e-14,
        rtol=0.0,
    )
    assert surface.is_stellarator_symmetric == fitted_surface.is_stellarator_symmetric


def test_generate_stellarator_symmetric_augmentations_when_flipping_z_coeffs(
    two_mode_surface,
):
    """Equivalent to flipping on the z axis."""

    # Get grid points for comparison
    surface = two_mode_surface
    theta = np.linspace(0, 2 * np.pi, 50)
    phi = np.linspace(0, 2 * np.pi * surface.n_field_periods / 2, 50)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    theta_phi_grid = np.stack([theta_grid, phi_grid], axis=-1)
    grid = surface_rz_fourier.evaluate_points_xyz(surface, theta_phi_grid).reshape(
        -1, 3
    )

    # Flip the grid on the z axis
    grid[:, 2] = -grid[:, 2]

    # Generate data augmentations
    augmentation = surface_rz_fourier.generate_stellarator_symmetric_augmentation(
        surface, [True]
    )
    augmentation_grid = surface_rz_fourier.evaluate_points_xyz(
        augmentation, theta_phi_grid
    ).reshape(-1, 3)

    np.testing.assert_allclose(grid, augmentation_grid, atol=1e-14)


def test_generate_stellarator_symmetric_augmentation_from_named_modes_no_augmentation():
    named_modes = {
        "r_cos(0, 0)": 1.0,
        "r_cos(1, 1)": 0.5,
        "z_sin(0, 1)": 0.3,
        "z_sin(1, 1)": 0.7,
    }
    augmentation_switches = [False, False, False]

    augmented_modes = surface_rz_fourier._generate_stellarator_symmetric_augmentation_from_named_modes(  # noqa: E501
        named_modes, augmentation_switches=augmentation_switches
    )

    assert augmented_modes == named_modes


def test_generate_stellarator_symmetric_augmentation_from_named_modes_flip_z_sin():
    named_modes = {
        "r_cos(0, 0)": 1.0,
        "r_cos(1, 1)": 0.5,
        "z_sin(0, 1)": 0.3,
        "z_sin(1, 1)": 0.7,
    }
    augmentation_switches = [True, False, False]

    augmented_modes = surface_rz_fourier._generate_stellarator_symmetric_augmentation_from_named_modes(  # noqa: E501
        named_modes, augmentation_switches=augmentation_switches
    )

    expected_modes = {
        "r_cos(0, 0)": 1.0,
        "r_cos(1, 1)": 0.5,
        "z_sin(0, 1)": -0.3,
        "z_sin(1, 1)": -0.7,
    }
    assert augmented_modes == expected_modes


def test_generate_stellarator_symmetric_augmentation_from_named_modes_flip_abs_n1():
    named_modes = {
        "r_cos(0, 0)": 1.0,
        "r_cos(1, -1)": 0.4,
        "r_cos(1, 1)": 0.5,
        "z_sin(1, -1)": 0.3,
        "z_sin(1, 1)": 0.7,
    }
    augmentation_switches = [False, True, False]

    augmented_modes = surface_rz_fourier._generate_stellarator_symmetric_augmentation_from_named_modes(  # noqa: E501
        named_modes, augmentation_switches=augmentation_switches
    )

    expected_modes = {
        "r_cos(0, 0)": 1.0,
        "r_cos(1, -1)": -0.4,
        "r_cos(1, 1)": -0.5,
        "z_sin(1, -1)": -0.3,
        "z_sin(1, 1)": -0.7,
    }
    assert augmented_modes == expected_modes


def test_generate_stellarator_symmetric_augmentation_from_named_modes_flip_m1():
    named_modes = {
        "r_cos(0, 0)": 1.0,
        "r_cos(1, -1)": 0.4,
        "r_cos(1, 1)": 0.5,
        "r_cos(2, 1)": 0.2,
        "z_sin(1, -1)": 0.3,
        "z_sin(1, 1)": 0.7,
    }
    augmentation_switches = [False, False, True]

    augmented_modes = surface_rz_fourier._generate_stellarator_symmetric_augmentation_from_named_modes(  # noqa: E501
        named_modes, augmentation_switches=augmentation_switches
    )

    expected_modes = {
        "r_cos(0, 0)": 1.0,
        "r_cos(1, -1)": -0.4,
        "r_cos(1, 1)": -0.5,
        "r_cos(2, 1)": 0.2,
        "z_sin(1, -1)": -0.3,
        "z_sin(1, 1)": -0.7,
    }
    assert augmented_modes == expected_modes


def test_generate_stellarator_symmetric_augmentation_from_named_modes_multiple_switches():  # noqa: E501
    named_modes = {
        "r_cos(0, 0)": 1.0,
        "r_cos(1, -1)": 0.4,
        "r_cos(1, 1)": 0.5,
        "r_cos(2, 1)": 0.2,
        "z_sin(1, -1)": 0.3,
        "z_sin(1, 1)": 0.7,
    }
    augmentation_switches = [True, False, True]

    augmented_modes = surface_rz_fourier._generate_stellarator_symmetric_augmentation_from_named_modes(  # noqa: E501
        named_modes, augmentation_switches=augmentation_switches
    )

    expected_modes = {
        "r_cos(0, 0)": 1.0,
        "r_cos(1, -1)": -0.4,
        "r_cos(1, 1)": -0.5,
        "r_cos(2, 1)": 0.2,
        "z_sin(1, -1)": 0.3,
        "z_sin(1, 1)": 0.7,
    }
    assert augmented_modes == expected_modes


def test_generate_stellarator_symmetric_augmentation_from_named_modes_random_seed():
    named_modes = {
        "r_cos(0, 0)": 1.0,
        "r_cos(1, 1)": 0.5,
        "z_sin(0, 1)": 0.3,
        "z_sin(1, 1)": 0.7,
    }
    seed = 42

    augmented_modes1 = surface_rz_fourier._generate_stellarator_symmetric_augmentation_from_named_modes(  # noqa: E501
        named_modes, seed=seed
    )
    augmented_modes2 = surface_rz_fourier._generate_stellarator_symmetric_augmentation_from_named_modes(  # noqa: E501
        named_modes, seed=seed
    )

    assert augmented_modes1 == augmented_modes2


def test_generate_stellarator_symmetric_augmentation_from_named_modes_non_stellarator_symmetric():  # noqa: E501
    named_modes = {
        "r_cos(0, 0)": 1.0,
        "z_cos(1, 1)": 0.5,
    }
    with pytest.raises(
        ValueError, match="The input named modes should be stellarator symmetric."
    ):
        surface_rz_fourier._generate_stellarator_symmetric_augmentation_from_named_modes(
            named_modes
        )


def test_generate_stellarator_symmetric_augmentation_from_named_modes_missing_r_cos_0_0():  # noqa: E501
    named_modes = {
        "r_cos(1, 1)": 0.5,
        "z_sin(0, 1)": 0.3,
    }
    with pytest.raises(
        ValueError,
        match="The input named modes should contain the mode r_cos\\(0, 0\\)",
    ):
        surface_rz_fourier._generate_stellarator_symmetric_augmentation_from_named_modes(
            named_modes
        )


def test_generate_stellarator_symmetric_augmentation_from_named_modes_example():
    named_modes = {
        "r_cos(0, 0)": 1.17402691,
        "r_cos(0, 1)": 0.29914774,
        "r_cos(1, -1)": -0.00545333,
        "r_cos(1, 0)": 0.14967022,
        "r_cos(1, 1)": -0.0997095,
        "z_sin(0, 1)": -0.24121311,
        "z_sin(1, -1)": -0.02070752,
        "z_sin(1, 0)": 0.26476756,
        "z_sin(1, 1)": 0.13379441,
    }
    augmentation_switches = [True, False, True]

    augmented_modes = surface_rz_fourier._generate_stellarator_symmetric_augmentation_from_named_modes(  # noqa: E501
        named_modes, augmentation_switches
    )

    expected_modes = {
        "r_cos(0, 0)": 1.17402691,
        "r_cos(0, 1)": 0.29914774,
        "r_cos(1, -1)": 0.00545333,
        "r_cos(1, 0)": -0.14967022,
        "r_cos(1, 1)": 0.0997095,
        "z_sin(0, 1)": 0.24121311,
        "z_sin(1, -1)": -0.02070752,
        "z_sin(1, 0)": 0.26476756,
        "z_sin(1, 1)": 0.13379441,
    }
    assert augmented_modes == expected_modes


@pytest.mark.parametrize(
    "surface_fixture",
    ["three_mode_surface", "two_mode_surface"],
)
def test_generate_stellarator_symmetric_augmentations_when_flipping_abs_n_is_1_coeffs(
    request, surface_fixture
):
    """Equivalent to rotating toroidaly by pi/nfp."""

    # Get grid points for comparison
    surface = request.getfixturevalue(surface_fixture)
    theta = np.linspace(0, 2 * np.pi, 50)
    phi = np.linspace(0, 2 * np.pi * surface.n_field_periods / 2, 50)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    theta_phi_grid = np.stack([theta_grid, phi_grid], axis=-1)

    # Shift the surface by pi / nfp in the toroidal direction
    shifted_surface = surface_rz_fourier.shift_in_angular_variables(
        surface, 0, np.pi / surface.n_field_periods
    )
    shifted_grid = surface_rz_fourier.evaluate_points_xyz(
        shifted_surface, theta_phi_grid
    ).reshape(-1, 3)

    # Generate data augmentations
    augmentation = surface_rz_fourier.generate_stellarator_symmetric_augmentation(
        surface, [False, True, False, False]
    )
    augmentation_grid = surface_rz_fourier.evaluate_points_xyz(
        augmentation, theta_phi_grid
    ).reshape(-1, 3)

    np.testing.assert_allclose(shifted_grid, augmentation_grid, atol=1e-14)


@pytest.mark.parametrize(
    "surface_fixture",
    ["three_mode_surface", "two_mode_surface"],
)
def test_generate_stellarator_symmetric_augmentations_when_flipping_m_is_1_coeffs(
    request, surface_fixture
):
    """Equivalent to shifting the surface by pi poloidally."""
    # Get grid points for comparison
    surface = request.getfixturevalue(surface_fixture)
    theta = np.linspace(0, 2 * np.pi, 50)
    phi = np.linspace(0, 2 * np.pi * surface.n_field_periods / 2, 50)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    theta_phi_grid = np.stack([theta_grid, phi_grid], axis=-1)

    # Shift the surface by pi in the poloidal direction
    shifted_surface = surface_rz_fourier.shift_in_angular_variables(surface, np.pi, 0)
    shifted_grid = surface_rz_fourier.evaluate_points_xyz(
        shifted_surface, theta_phi_grid
    ).reshape(-1, 3)

    # Generate data augmentations
    augmentation = surface_rz_fourier.generate_stellarator_symmetric_augmentation(
        surface, [False, False, True, False]
    )
    third_augmentation_grid = surface_rz_fourier.evaluate_points_xyz(
        augmentation, theta_phi_grid
    ).reshape(-1, 3)

    np.testing.assert_allclose(shifted_grid, third_augmentation_grid, atol=1e-14)


def test_compute_normal_displacement_zero_for_identical_surfaces() -> None:
    surface = surface_rz_fourier.SurfaceRZFourier(
        r_cos=np.array([[3.0], [0.5]]),
        z_sin=np.array([[0.0], [0.5]]),
        n_field_periods=1,
        is_stellarator_symmetric=True,
    )

    distances = surface_rz_fourier.compute_normal_displacement(
        target_surface=surface,
        comparison_surface=surface,
        n_poloidal_points=4,
        n_toroidal_points=3,
    )

    np.testing.assert_allclose(np.asarray(distances), 0.0, atol=1e-12)


def test_compute_normal_displacement_matches_radial_offset() -> None:
    delta = 0.2
    target_surface = surface_rz_fourier.SurfaceRZFourier(
        r_cos=np.array([[3.0], [0.4]]),
        z_sin=np.array([[0.0], [0.4]]),
        n_field_periods=1,
        is_stellarator_symmetric=True,
    )
    comparison_surface = surface_rz_fourier.SurfaceRZFourier(
        r_cos=np.array([[3.0 + delta], [0.4]]),
        z_sin=np.array([[0.0], [0.4]]),
        n_field_periods=1,
        is_stellarator_symmetric=True,
    )

    distances = surface_rz_fourier.compute_normal_displacement(
        target_surface=target_surface,
        comparison_surface=comparison_surface,
        n_poloidal_points=5,
        n_toroidal_points=4,
    )

    np.testing.assert_allclose(np.max(np.abs(distances)), delta, atol=1e-7)


def test_compute_normal_displacement_raises_for_mismatched_parameters() -> None:
    base_surface = surface_rz_fourier.SurfaceRZFourier(
        r_cos=np.array([[3.0], [0.5]]),
        z_sin=np.array([[0.0], [0.5]]),
        n_field_periods=1,
        is_stellarator_symmetric=True,
    )

    mismatched_field_periods = base_surface.model_copy(update=dict(n_field_periods=2))
    with pytest.raises(ValueError, match="field periods"):
        surface_rz_fourier.compute_normal_displacement(
            target_surface=base_surface,
            comparison_surface=mismatched_field_periods,
            n_poloidal_points=3,
            n_toroidal_points=2,
        )

    mismatched_symmetry = base_surface.model_copy(
        update=dict(is_stellarator_symmetric=False)
    )
    with pytest.raises(ValueError, match="stellarator symmetry"):
        surface_rz_fourier.compute_normal_displacement(
            target_surface=base_surface,
            comparison_surface=mismatched_symmetry,
            n_poloidal_points=3,
            n_toroidal_points=2,
        )


# ---------- to_standard_orientation tests ----------


def test_to_standard_orientation_torus_external_ccw() -> None:
    """Surface already in standard orientation is returned unchanged."""
    major_radius = 1.5
    minor_radius = 0.5
    r_cos = np.array([[major_radius], [minor_radius]])
    z_sin = np.array([[0.0], [minor_radius]])
    surface = surface_rz_fourier.SurfaceRZFourier(
        r_cos=r_cos, z_sin=z_sin, is_stellarator_symmetric=True
    )
    assert surface_rz_fourier._is_theta_external(surface)
    assert surface_rz_fourier._is_theta_counterclockwise(surface)
    surface_std = surface_rz_fourier.to_standard_orientation(surface)
    np.testing.assert_allclose(surface_std.r_cos, surface.r_cos)
    np.testing.assert_allclose(surface_std.z_sin, surface.z_sin)
    assert surface_rz_fourier._is_theta_external(surface_std)
    assert surface_rz_fourier._is_theta_counterclockwise(surface_std)


def test_to_standard_orientation_torus_internal_ccw() -> None:
    """theta=0 on the inside, counterclockwise: needs a theta shift by pi."""
    major_radius = 1.5
    minor_radius = 0.5
    r_cos = np.array([[major_radius], [-minor_radius]])
    z_sin = np.array([[0.0], [-minor_radius]])
    surface = surface_rz_fourier.SurfaceRZFourier(
        r_cos=r_cos, z_sin=z_sin, is_stellarator_symmetric=True
    )
    assert not surface_rz_fourier._is_theta_external(surface)
    assert surface_rz_fourier._is_theta_counterclockwise(surface)
    surface_std = surface_rz_fourier.to_standard_orientation(surface)
    assert surface_rz_fourier._is_theta_external(surface_std)
    assert surface_rz_fourier._is_theta_counterclockwise(surface_std)
    # Verify geometric equivalence: original at theta evaluates to same xyz
    # as standardized at theta+pi
    rng = np.random.default_rng(42)
    theta_phi = rng.random((7, 8, 2)) * 2 * np.pi
    points = surface_rz_fourier.evaluate_points_xyz(surface, theta_phi)
    theta_phi_std = theta_phi.copy()
    theta_phi_std[:, :, 0] += np.pi
    points_std = surface_rz_fourier.evaluate_points_xyz(surface_std, theta_phi_std)
    np.testing.assert_allclose(points, points_std, atol=1e-12)


def test_to_standard_orientation_torus_external_cw() -> None:
    """theta=0 on the outside, clockwise: needs theta sign flip."""
    major_radius = 1.5
    minor_radius = 0.5
    r_cos = np.array([[major_radius], [minor_radius]])
    z_sin = np.array([[0.0], [-minor_radius]])
    surface = surface_rz_fourier.SurfaceRZFourier(
        r_cos=r_cos, z_sin=z_sin, is_stellarator_symmetric=True
    )
    assert surface_rz_fourier._is_theta_external(surface)
    assert not surface_rz_fourier._is_theta_counterclockwise(surface)
    surface_std = surface_rz_fourier.to_standard_orientation(surface)
    assert surface_rz_fourier._is_theta_external(surface_std)
    assert surface_rz_fourier._is_theta_counterclockwise(surface_std)
    # Verify geometric equivalence: original at theta = standardized at -theta
    rng = np.random.default_rng(42)
    theta_phi = rng.random((7, 8, 2)) * 2 * np.pi
    points = surface_rz_fourier.evaluate_points_xyz(surface, theta_phi)
    theta_phi_std = theta_phi.copy()
    theta_phi_std[:, :, 0] = -theta_phi_std[:, :, 0]
    points_std = surface_rz_fourier.evaluate_points_xyz(surface_std, theta_phi_std)
    np.testing.assert_allclose(points, points_std, atol=1e-12)


def test_to_standard_orientation_torus_internal_cw() -> None:
    """theta=0 on the inside, clockwise: needs both shift and sign flip."""
    major_radius = 1.5
    minor_radius = 0.5
    r_cos = np.array([[major_radius], [-minor_radius]])
    z_sin = np.array([[0.0], [minor_radius]])
    surface = surface_rz_fourier.SurfaceRZFourier(
        r_cos=r_cos, z_sin=z_sin, is_stellarator_symmetric=True
    )
    assert not surface_rz_fourier._is_theta_external(surface)
    assert not surface_rz_fourier._is_theta_counterclockwise(surface)
    surface_std = surface_rz_fourier.to_standard_orientation(surface)
    assert surface_rz_fourier._is_theta_external(surface_std)
    assert surface_rz_fourier._is_theta_counterclockwise(surface_std)
    # Verify geometric equivalence: original at theta = standardized at -theta+pi
    rng = np.random.default_rng(42)
    theta_phi = rng.random((7, 8, 2)) * 2 * np.pi
    points = surface_rz_fourier.evaluate_points_xyz(surface, theta_phi)
    theta_phi_std = theta_phi.copy()
    theta_phi_std[:, :, 0] = -theta_phi_std[:, :, 0] + np.pi
    points_std = surface_rz_fourier.evaluate_points_xyz(surface_std, theta_phi_std)
    np.testing.assert_allclose(points, points_std, atol=1e-12)

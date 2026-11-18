import json
import pathlib

import numpy as np
import pytest

from constellaration.geometry import neutron_wall_load, surface_rz_fourier

_TEST_DATA = pathlib.Path(__file__).parent / "test_data"


def _circular_torus(
    major_radius: float, minor_radius: float
) -> surface_rz_fourier.SurfaceRZFourier:
    r_cos = np.array([[0.0, major_radius, 0.0], [0.0, minor_radius, 0.0]])
    z_sin = np.array([[0.0, 0.0, 0.0], [0.0, minor_radius, 0.0]])
    return surface_rz_fourier.SurfaceRZFourier(r_cos=r_cos, z_sin=z_sin)


def _stellarator_boundary(scale: float) -> surface_rz_fourier.SurfaceRZFourier:
    with (_TEST_DATA / "test_boundary_1.json").open() as f:
        data = json.load(f)
    return surface_rz_fourier.SurfaceRZFourier(
        r_cos=scale * np.array(data["r_cos"]),
        z_sin=scale * np.array(data["z_sin"]),
        n_field_periods=data["n_field_periods"],
        is_stellarator_symmetric=data["is_stellarator_symmetric"],
    )


def _scaled_about_centre(
    surface: surface_rz_fourier.SurfaceRZFourier, factor: float
) -> surface_rz_fourier.SurfaceRZFourier:
    """Scale every cross section about its centre (the m = 0 part)."""
    r_cos = surface.r_cos.copy()
    z_sin = surface.z_sin.copy()
    r_cos[1:] *= factor
    z_sin[1:] *= factor
    return surface_rz_fourier.SurfaceRZFourier(
        r_cos=r_cos, z_sin=z_sin, n_field_periods=surface.n_field_periods
    )


@pytest.mark.parametrize("major_radius", [3.0, 10.0])
def test_wall_load_integrates_to_total_power_on_a_torus(major_radius: float) -> None:
    result = neutron_wall_load.compute_neutron_wall_load(
        plasma_boundary=_circular_torus(major_radius, 1.0),
        wall=_circular_torus(major_radius, 1.3),
        total_power=2.0,
        n_theta_wall=16,
        n_phi_wall=2,
    )

    assert result.n_periods == 1
    np.testing.assert_allclose(
        result.wall_area, 4 * np.pi**2 * major_radius * 1.3, rtol=1e-6
    )
    np.testing.assert_allclose(result.mean_wall_load, 2.0 / result.wall_area)
    assert abs(result.power_balance_error) < 5e-3


def test_wall_load_on_a_torus_is_axisymmetric_and_peaks_outboard() -> None:
    result = neutron_wall_load.compute_neutron_wall_load(
        plasma_boundary=_circular_torus(3.0, 1.0),
        wall=_circular_torus(3.0, 1.3),
        n_theta_wall=16,
        n_phi_wall=3,
    )

    load = result.wall_load
    # The direction quadrature is aligned differently at each wall point, so the
    # load is axisymmetric only to within its error.
    np.testing.assert_allclose(load, load[:, :1].repeat(3, axis=1), rtol=1e-2)
    # Up-down symmetry: theta -> -theta.
    np.testing.assert_allclose(load[1:, 0], load[:0:-1, 0], rtol=1e-2)
    # theta = 0 is the outboard midplane, theta = pi the inboard one.
    assert np.argmax(load[:, 0]) == 0
    assert np.argmin(load[:, 0]) == 8


def test_wall_load_approaches_uniform_at_large_aspect_ratio() -> None:
    """For a thin torus the wall sees a straight cylinder, on which a coaxial source
    gives a uniform load."""
    result = neutron_wall_load.compute_neutron_wall_load(
        plasma_boundary=_circular_torus(100.0, 1.0),
        wall=_circular_torus(100.0, 1.3),
        n_theta_wall=8,
        n_phi_wall=1,
    )

    assert abs(result.power_balance_error) < 5e-3
    assert result.peaking_factor < 1.04
    assert np.min(result.wall_load) / result.mean_wall_load > 0.98


def test_wall_load_on_a_stellarator() -> None:
    plasma_boundary = _stellarator_boundary(scale=15.0)
    result = neutron_wall_load.compute_neutron_wall_load(
        plasma_boundary=plasma_boundary,
        wall=_scaled_about_centre(plasma_boundary, 1.2),
        n_theta_wall=16,
        n_phi_wall=8,
    )

    assert result.n_periods == plasma_boundary.n_field_periods
    assert abs(result.power_balance_error) < 5e-3
    assert result.peaking_factor > 1.0
    # Stellarator symmetry: load(theta, phi) = load(-theta, -phi).
    load = result.wall_load
    mirrored = load[(-np.arange(16)) % 16][:, (-np.arange(8)) % 8]
    np.testing.assert_allclose(load, mirrored, rtol=1e-2)


def test_wall_load_is_linear_in_total_power_and_conserves_peaked_sources() -> None:
    plasma_boundary = _circular_torus(3.0, 1.0)
    wall = _circular_torus(3.0, 1.3)

    uniform = neutron_wall_load.compute_neutron_wall_load(
        plasma_boundary, wall, total_power=1.0, n_theta_wall=8, n_phi_wall=1
    )
    doubled = neutron_wall_load.compute_neutron_wall_load(
        plasma_boundary, wall, total_power=2.0, n_theta_wall=8, n_phi_wall=1
    )
    peaked = neutron_wall_load.compute_neutron_wall_load(
        plasma_boundary,
        wall,
        total_power=1.0,
        source_profile=lambda rho: (1 - rho**2) ** 2,
        n_theta_wall=8,
        n_phi_wall=1,
    )

    np.testing.assert_allclose(doubled.wall_load, 2 * uniform.wall_load)
    assert abs(peaked.power_balance_error) < 5e-3
    # A source concentrated near the centre loads the wall more evenly.
    assert peaked.peaking_factor < uniform.peaking_factor


def test_wall_must_enclose_the_plasma() -> None:
    with pytest.raises(ValueError, match="does not enclose"):
        neutron_wall_load.compute_neutron_wall_load(
            plasma_boundary=_circular_torus(3.0, 1.0),
            wall=_circular_torus(3.0, 0.9),
        )


def test_cross_sections_must_be_star_shaped() -> None:
    # A strongly indented (kidney-shaped) cross section.
    r_cos = np.array([[0.0, 3.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    z_sin = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    r_cos[2, 1] = 0.9
    kidney = surface_rz_fourier.SurfaceRZFourier(r_cos=r_cos, z_sin=z_sin)

    with pytest.raises(ValueError, match="star-shaped"):
        neutron_wall_load.compute_neutron_wall_load(
            plasma_boundary=kidney, wall=_circular_torus(3.0, 3.0)
        )


def test_source_profile_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        neutron_wall_load.compute_neutron_wall_load(
            plasma_boundary=_circular_torus(3.0, 1.0),
            wall=_circular_torus(3.0, 1.3),
            source_profile=lambda rho: rho - 0.5,
            n_theta_wall=4,
            n_phi_wall=1,
        )

import jaxtyping as jt
import numpy as np
from desc import geometry as desc_geometry

from constellaration.geometry import surface_rz_fourier, surface_utils_desc


def test_from_desc_fourier_rz_toroidal_surface() -> None:
    desc_surface = desc_geometry.FourierRZToroidalSurface.from_qp_model(
        major_radius=1,
        aspect_ratio=9,
        elongation=2,
        mirror_ratio=0.2,
        torsion=0,
        NFP=2,
        sym=True,
    )  # Model based off of section III of Goodman et. al 2023

    surface = surface_utils_desc.from_desc_fourier_rz_toroidal_surface(desc_surface)

    assert surface.r_sin is None
    assert surface.z_cos is None
    assert surface.n_field_periods == 2
    assert surface.is_stellarator_symmetric
    assert surface.max_poloidal_mode == 1
    assert surface.max_toroidal_mode == 2
    assert isinstance(desc_surface.R_lmn, jt.Array)
    assert isinstance(desc_surface.Z_lmn, jt.Array)
    assert np.prod(surface.r_cos.shape) - surface.max_toroidal_mode == np.prod(
        desc_surface.R_lmn.shape
    )
    assert np.prod(surface.z_sin.shape) - (surface.max_toroidal_mode + 1) == np.prod(
        desc_surface.Z_lmn.shape
    )


def test_round_trip_from_and_to_desc() -> None:
    desc_surface = desc_geometry.FourierRZToroidalSurface.from_qp_model(
        major_radius=1,
        aspect_ratio=9,
        elongation=2,
        mirror_ratio=0.2,
        torsion=0,
        NFP=2,
        sym=True,
    )  # Model based off of section III of Goodman et. al 2023

    # Create SurfaceRZFourier surface from desc
    surface = surface_utils_desc.from_desc_fourier_rz_toroidal_surface(desc_surface)

    actual_desc_surface = surface_utils_desc.to_desc_fourier_rz_toroidal_surface(
        surface
    )

    assert desc_surface.NFP == actual_desc_surface.NFP
    assert desc_surface.sym == actual_desc_surface.sym
    np.testing.assert_allclose(actual_desc_surface.R_lmn, desc_surface.R_lmn)
    np.testing.assert_allclose(actual_desc_surface.Z_lmn, desc_surface.Z_lmn)


def test_from_qp_model_aspect_ratio() -> None:
    surface = surface_utils_desc.from_qp_model(
        aspect_ratio=10.0,
        elongation=1.5,
        mirror_ratio=0.2,
        torsion=0.0,
        major_radius=1.0,
    )
    aspect_ratio = 1.0 / surface_rz_fourier.evaluate_minor_radius(surface=surface)
    np.testing.assert_allclose(aspect_ratio, 10.0, atol=0.0, rtol=1.0e-1)

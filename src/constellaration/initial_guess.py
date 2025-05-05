from constellaration.geometry import surface_rz_fourier
from simsopt import geo

_EDGE_IOTA_OVER_N_FIELD_PERIODS_SCALING_PARAMETERS = (0.4, 10.0)


def generate_rotating_ellipse(
    aspect_ratio: float,
    elongation: float,
    rotational_transform: float,
    n_field_periods: int,
) -> surface_rz_fourier.SurfaceRZFourier:
    """Generates a rotating ellipse boundary.

    Args:
        aspect_ratio: The aspect ratio of the plasma.
        elongation: The elongation of the plasma.
        rotational_transform: The rotational transform at the edge of the plasma.
        n_field_periods: The number of field periods.

    Returns:
        The rotating ellipse boundary.
    """
    simsopt_surface = geo.SurfaceRZFourier(
        nfp=n_field_periods,
        stellsym=True,
        mpol=1,
        ntor=1,
    )
    torsion = _get_torsion_at_rotational_transform_over_n_field_periods(
        rotational_transform_over_n_field_periods=rotational_transform
        / n_field_periods,
        aspect_ratio=aspect_ratio,
        elongation=elongation,
    )
    simsopt_surface.make_rotating_ellipse(
        major_radius=1.0,
        minor_radius=1.0 / aspect_ratio,
        elongation=elongation,
        torsion=torsion,  # type: ignore
    )
    return surface_rz_fourier.from_simsopt(surface=simsopt_surface)


def _get_torsion_at_rotational_transform_over_n_field_periods(
    rotational_transform_over_n_field_periods: float,
    aspect_ratio: float,
    elongation: float,
) -> float:
    """Get the torsion required to achieve a given rotational transform over the number
    of field periods."""
    c0, c1 = _EDGE_IOTA_OVER_N_FIELD_PERIODS_SCALING_PARAMETERS
    inverse_aspect = 1.0 / aspect_ratio
    return (
        rotational_transform_over_n_field_periods / c0 / inverse_aspect
        - (elongation - 1)
    ) / c1

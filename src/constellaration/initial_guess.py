from constellaration.geometry import surface_rz_fourier
from constellaration.mhd import near_axis_configuration
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


def generate_nae(
    aspect_ratio: float,
    max_elongation: float,
    rotational_transform: float,
    mirror_ratio: float,
    n_field_periods: int,
    max_poloidal_mode: int = 1,
    max_toroidal_mode: int = 1,
) -> surface_rz_fourier.SurfaceRZFourier:
    """Generates a ~QI boundary generated with the near-axis expansion (NAE) framework.

    Args:
        aspect_ratio: The aspect ratio of the plasma.
        max_elongation: The maximum elongation of the plasma.
        rotational_transform: The rotational transform at the edge of the plasma.
        mirror_ratio: The mirror ratio of the plasma.
        n_field_periods: The number of field periods.
        max_poloidal_mode: The maximum poloidal mode number of the resulting boundary.
        max_toroidal_mode: The maximum toroidal mode number of the resulting boundary.

    Returns:
        The NAE boundary.
    """

    near_axis_expansion = near_axis_configuration.generate(
        mirror_ratio=mirror_ratio,
        min_iota=rotational_transform,
        max_elongation=max_elongation,
        # Empirical scaling.
        torsion=1.33 / aspect_ratio,
        n_field_periods=n_field_periods,
        major_radius=1.0,
        # We require at least n=2 to satisfy QI conditions on axis.
        max_toroidal_mode=max(3, max_toroidal_mode),
    )

    first_order_to_plasma_settings = (
        near_axis_configuration.NearAxisToPlasmaBoundarySettings(
            minor_radius=1.0 / aspect_ratio
        )
    )

    boundary = near_axis_configuration.near_axis_configuration_to_plasma_boundary(
        near_axis_configuration=near_axis_expansion,
        settings=first_order_to_plasma_settings,
    )

    low_mode_boundary = near_axis_configuration.smooth_and_set_max_mode_numbers(
        boundary,
        max_poloidal_mode,
        max_toroidal_mode,
    )
    return low_mode_boundary


def _get_torsion_at_rotational_transform_over_n_field_periods(
    rotational_transform_over_n_field_periods: float,
    aspect_ratio: float,
    elongation: float,
) -> float:
    r"""Get the torsion required to achieve a given rotational transform over the number
    of field periods.

    This function is based on a simple linear fit to map the aspect ratio, elongation,
    and torsion to the rotational transform over the number of field periods. The linear
    mapping is:

    .. math::
        \frac{\iota}{N_{\text{fp}}} = \frac{c_0}{A} (c_1 \kappa  + (\epsilon - 1))

    where :math:`\iota` is the rotational transform, :math:`N_{\text{fp}}` is the number
    of field periods, :math:`A` is the aspect ratio, :math:`\kappa` is the torsion,
    :math:`\epsilon` is the elongation, and :math:`c_0` and :math:`c_1` are constants
    determined by the linear fit.

    Args:
        rotational_transform_over_n_field_periods: The desired rotational transform over
            the number of field periods.
        aspect_ratio: The aspect ratio of the plasma.
        elongation: The elongation of the plasma.
    """
    c0, c1 = _EDGE_IOTA_OVER_N_FIELD_PERIODS_SCALING_PARAMETERS
    inverse_aspect = 1.0 / aspect_ratio
    return (
        rotational_transform_over_n_field_periods / c0 / inverse_aspect
        - (elongation - 1)
    ) / c1

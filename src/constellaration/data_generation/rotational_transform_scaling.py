EDGE_IOTA_OVER_N_FIELD_PERIODS_SCALING_PARAMETERS = (0.4, 10.0)
"""Scaling parameters for the linear fit between inverse aspect ratio,
elongation, and torsion to estimate the rotational transform over
the number of field periods."""


def rotational_transform_over_n_field_periods(
    aspect: float, elongation: float, torsion: float, c0: float, c1: float
) -> float:
    """Simple function to estimate the rotational transform given an aspect ratio,
    elongation, and torsion."""
    inverse_aspect = 1.0 / aspect
    return c0 * inverse_aspect * ((elongation - 1) + c1 * torsion)


def get_torsion_at_rotational_transform_over_n_field_periods(
    rotational_transform_over_n_field_periods: float,
    aspect_ratio: float,
    elongation: float,
) -> float:
    """Get the torsion required to achieve a given rotational transform over the number
    of field periods."""
    c0, c1 = EDGE_IOTA_OVER_N_FIELD_PERIODS_SCALING_PARAMETERS
    inverse_aspect = 1.0 / aspect_ratio
    return (
        rotational_transform_over_n_field_periods / c0 / inverse_aspect
        - (elongation - 1)
    ) / c1

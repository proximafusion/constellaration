import numpy as np
import pytest

from constellaration.data_generation import desc_optimization
from constellaration.omnigeneity import omnigenity_field


@pytest.fixture
def dummy_omnigenous_field():
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
            [0.0, 0.0, 0.0],
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


def test_get_mirror_ratio_from_field_simple(dummy_omnigenous_field):
    # (max - min)/(max + min)
    dummy_omnigenous_field.modB_spline_knot_coefficients = np.array(
        [[0.2, 0.8], [0.0, 0.0]]
    )
    ratio = desc_optimization._get_mirror_ratio_from_field(dummy_omnigenous_field)
    assert pytest.approx((0.8 - 0.2) / (0.8 + 0.2)) == ratio

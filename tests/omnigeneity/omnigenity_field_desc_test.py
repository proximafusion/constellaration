import numpy as np
import pytest

from constellaration.omnigeneity import omnigenity_field, omnigenity_field_desc


@pytest.fixture
def omnigenous_field() -> omnigenity_field.OmnigenousField:
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


def test_round_trip_from_to_desc(omnigenous_field):
    field_desc = omnigenity_field_desc.omnigenous_field_to_desc(omnigenous_field)
    recovered_omnigenous_field = omnigenity_field_desc.omnigenous_field_from_desc(
        field_desc
    )

    assert (
        omnigenous_field.n_field_periods == recovered_omnigenous_field.n_field_periods
    )
    assert omnigenous_field.helicity == recovered_omnigenous_field.helicity
    np.testing.assert_allclose(
        omnigenous_field.modB_spline_knot_coefficients,
        recovered_omnigenous_field.modB_spline_knot_coefficients,
    )
    np.testing.assert_allclose(omnigenous_field.x_lmn, recovered_omnigenous_field.x_lmn)

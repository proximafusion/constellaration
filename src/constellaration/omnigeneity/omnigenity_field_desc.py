import numpy as np
from desc import magnetic_fields as desc_magnetic_fields

from constellaration.omnigeneity import omnigenity_field


def omnigenous_field_from_desc(
    field_desc: desc_magnetic_fields.OmnigenousField,
) -> omnigenity_field.OmnigenousField:
    """Convert a DESC omnigenous field into a omnigenity_field.OmnigenousField.

    Args:
        field_desc: A DESC magnetic field object.

    Returns:
        An omnigenous field object.
    """

    assert isinstance(field_desc.NFP, int)
    assert field_desc.L_B is not None
    assert field_desc.M_B is not None
    assert field_desc.L_x is not None
    assert field_desc.N_x is not None
    assert field_desc.M_x is not None
    assert isinstance(field_desc.x_lmn, np.ndarray)

    n_field_periods = field_desc.NFP
    poloidal_winding, toroidal_winding = field_desc.helicity

    modB_spline_knot_coefficients = field_desc.B_lm.reshape(
        field_desc.L_B + 1, field_desc.M_B
    )

    x_lmn = field_desc.x_lmn.reshape(
        2 * field_desc.N_x + 1, field_desc.L_x + 1, field_desc.N_x + 1
    ).transpose(1, 2, 0)

    return omnigenity_field.OmnigenousField(
        n_field_periods=n_field_periods,
        poloidal_winding=poloidal_winding,
        torodial_winding=toroidal_winding,
        x_lmn=x_lmn,
        modB_spline_knot_coefficients=modB_spline_knot_coefficients,
    )


def omnigenous_field_to_desc(
    field: omnigenity_field.OmnigenousField,
) -> desc_magnetic_fields.OmnigenousField:
    """Convert a omnigenity_field_types.OmnigenousField into a DESC omnigenous field.

    Args:
        field: An omnigenous field object.

    Returns:
        A DESC magnetic field object.
    """
    helicity = field.helicity

    L_B = field.max_modB_rho_coefficient
    M_B = field.n_modB_eta_spline_knots
    B_lm = field.modB_spline_knot_coefficients.flatten()

    L_x = field.max_x_rho_coefficients
    N_x = field.max_x_alpha_coefficients
    M_x = field.max_x_eta_coefficinets

    x_lmn = field.x_lmn.transpose(2, 0, 1).flatten()

    return desc_magnetic_fields.OmnigenousField(
        NFP=field.n_field_periods,
        L_B=L_B,
        M_B=M_B,
        L_x=L_x,
        N_x=N_x,
        M_x=M_x,
        B_lm=B_lm,
        x_lmn=x_lmn,
        helicity=helicity,
    )

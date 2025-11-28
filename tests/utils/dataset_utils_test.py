import numpy as np

from constellaration import forward_model, initial_guess
from constellaration.geometry import surface_rz_fourier
from constellaration.utils import dataset_utils

# Expected structure of a dataset row with field names and types
EXPECTED_ROW_STRUCTURE = {
    "boundary": {
        "r_cos": list,
        "z_sin": list,
        "n_field_periods": int,
        "json": str,
    },
    "metrics": {
        "aspect_ratio": float,
        "max_elongation": float,
        "json": str,
    },
    "misc": {
        "has_neurips_2025_forward_model_error": bool,
    },
}


def test_boundary_to_dataset_row_success() -> None:
    """Test that boundary_to_dataset_row works with a valid boundary."""
    boundary = initial_guess.generate_rotating_ellipse(
        aspect_ratio=3, elongation=0.5, rotational_transform=0.4, n_field_periods=3
    )
    row = dataset_utils.boundary_to_dataset_row(
        boundary, settings=forward_model.ConstellarationSettings(qi_settings=None)
    )

    assert "boundary" in row
    assert "metrics" in row
    assert "misc" in row

    assert row["misc"]["has_neurips_2025_forward_model_error"] is False

    for field, expected_type in EXPECTED_ROW_STRUCTURE["boundary"].items():
        assert field in row["boundary"]
        assert isinstance(row["boundary"][field], expected_type)

    for field, expected_type in EXPECTED_ROW_STRUCTURE["metrics"].items():
        assert field in row["metrics"]
        assert isinstance(row["metrics"][field], expected_type)


def test_boundary_to_dataset_row_failure() -> None:
    """Test that boundary_to_dataset_row handles VMEC failures gracefully."""
    boundary = surface_rz_fourier.SurfaceRZFourier(
        r_cos=np.array([[0.0, 0.0, 0.001], [0.0, 0.0, 0.0]]),
        z_sin=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.001]]),
        n_field_periods=1,
        is_stellarator_symmetric=True,
    )
    row = dataset_utils.boundary_to_dataset_row(boundary)

    assert "boundary" in row
    assert "misc" in row
    assert "metrics" not in row

    assert row["misc"]["has_neurips_2025_forward_model_error"] is True

    for field, expected_type in EXPECTED_ROW_STRUCTURE["boundary"].items():
        assert field in row["boundary"]
        assert isinstance(row["boundary"][field], expected_type)

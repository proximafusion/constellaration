from typing import Any

from constellaration import forward_model
from constellaration.geometry import surface_rz_fourier
from constellaration.mhd import ideal_mhd_parameters, vmec_settings


def boundary_to_dataset_row(
    boundary: surface_rz_fourier.SurfaceRZFourier,
    ideal_mhd_parameters: ideal_mhd_parameters.IdealMHDParameters | None = None,
    settings: forward_model.ConstellarationSettings | None = None,
) -> dict[str, Any]:
    """Convert a plasma boundary to a dataset row in JSON format.

    This function takes a SurfaceRZFourier boundary, computes all relevant metrics
    using the forward model, and produces a dictionary matching the Constellaration
    dataset schema. Only boundary-related and metrics-related columns are populated.

    If the forward model fails, the function will still return a row with the boundary
    data and the misc.has_neurips_2025_forward_model_error flag set to True.

    Args:
        boundary: The plasma boundary surface represented as a SurfaceRZFourier object.
        ideal_mhd_parameters: Optional ideal-MHD parameters. If None, default vacuum
            parameters are used.
        settings: Optional ConstellarationSettings for the forward model. If None,
            high fidelity settings are used.

    Returns:
        A dictionary with the same structure as the dataset, i.e.:
        {
            "boundary": { ... boundary data ... },
            "metrics": { ... computed metrics ... } (only if forward model succeeds),
            "misc": {
                "has_neurips_2025_forward_model_error": bool
            }
        }
    """
    if settings is None:
        settings = forward_model.ConstellarationSettings(
            vmec_preset_settings=vmec_settings.VmecPresetSettings(
                fidelity="high_fidelity"
            )
        )
    boundary_data = boundary.model_dump(mode="json")
    boundary_data["json"] = boundary.model_dump_json()

    # Try to run the forward model to compute metrics
    has_error = False
    metrics_data = None

    try:
        metrics, _ = forward_model.forward_model(
            boundary=boundary,
            ideal_mhd_parameters=ideal_mhd_parameters,
            settings=settings,
        )

        metrics_dict = metrics.model_dump()
        metrics_data = {
            **metrics_dict,
        }
        metrics_data["json"] = metrics.model_dump_json()

    except Exception:
        has_error = True

    misc_data = {
        "has_neurips_2025_forward_model_error": has_error,
    }

    dataset_row = {
        "boundary": boundary_data,
        "misc": misc_data,
    }

    # Only include metrics if forward model succeeded
    if not has_error:
        dataset_row["metrics"] = metrics_data

    return dataset_row

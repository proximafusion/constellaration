import numpy as np
import pydantic
from constellaration.boozer import boozer as boozer_module
from constellaration.geometry import (
    radial_profile,
    surface_rz_fourier,
    surface_utils,
)
from constellaration.mhd import geometry_utils
from constellaration.mhd import (
    ideal_mhd_parameters as ideal_mhd_parameters_module,
)
from constellaration.mhd import magnetics_utils, turbulent_transport
from constellaration.mhd import vmec_settings as vmec_settings_module
from constellaration.mhd import vmec_utils
from constellaration.omnigeneity import qi


class ConstellarationMetrics(pydantic.BaseModel):
    aspect_ratio: float

    aspect_ratio_over_edge_rotational_transform: float

    max_elongation: float

    axis_rotational_transform_over_n_field_periods: float

    edge_rotational_transform_over_n_field_periods: float

    axis_magnetic_mirror_ratio: float

    edge_magnetic_mirror_ratio: float

    average_triangularity: float

    vacuum_well: float

    minimum_normalized_magnetic_gradient_scale_length: float

    qi: float | None = None

    flux_compression_in_regions_of_bad_curvature: float | None = None


class ConstellarationSettings(pydantic.BaseModel):
    vmec_preset_settings: vmec_settings_module.VmecPresetSettings = (
        vmec_settings_module.VmecPresetSettings(
            fidelity="low_fidelity",
        )
    )
    boozer_preset_settings: boozer_module.BoozerPresetSettings | None = (
        boozer_module.BoozerPresetSettings(
            normalized_toroidal_flux=[1.0],
        )
    )
    qi_settings: qi.QISettings | None = qi.QISettings()
    turbulent_settings: (
        turbulent_transport.IdealMHDTurbulentTransportMetricsSettings | None
    ) = turbulent_transport.IdealMHDTurbulentTransportMetricsSettings(
        normalized_toroidal_flux=np.array([0.7**2]),
        n_field_lines=101,
        n_toroidal_points=64,
    )

    @staticmethod
    def default_high_fidelity() -> "ConstellarationSettings":
        return ConstellarationSettings(
            vmec_preset_settings=vmec_settings_module.VmecPresetSettings(
                fidelity="high_fidelity",
            ),
        )

    @staticmethod
    def default_high_fidelity_skip_qi() -> "ConstellarationSettings":
        return ConstellarationSettings(
            vmec_preset_settings=vmec_settings_module.VmecPresetSettings(
                fidelity="high_fidelity",
            ),
            boozer_preset_settings=None,
            qi_settings=None,
        )


def forward_model(
    boundary: surface_rz_fourier.SurfaceRZFourier,
    ideal_mhd_parameters: ideal_mhd_parameters_module.IdealMHDParameters | None = None,
    settings: ConstellarationSettings | None = None,
) -> tuple[ConstellarationMetrics, vmec_utils.VmecppWOut]:
    """Runs the forward model.

    Args:
        boundary: the boundary surface of the plasma.
        ideal_mhd_parameters: the ideal-MHD parameters; if None, default parameters in
            vacuum are used.
        settings: the settings for the forward model; if None, default settings are
            used.

    Returns:
        The computed metrics for the boundary.
    """
    if ideal_mhd_parameters is None:
        ideal_mhd_parameters = (
            ideal_mhd_parameters_module.boundary_to_ideal_mhd_parameters(boundary)
        )
    if settings is None:
        settings = ConstellarationSettings()
    vmec_settings = vmec_settings_module.create_vmec_settings_from_preset(
        boundary,
        settings=settings.vmec_preset_settings,
    )
    equilibrium = vmec_utils.run_vmec(
        boundary=boundary,
        mhd_parameters=ideal_mhd_parameters,
        vmec_settings=vmec_settings,
    )

    # Geometrical metrics
    (
        n_poloidal_points,
        n_toroidal_points,
    ) = surface_utils.n_poloidal_toroidal_points_to_satisfy_nyquist_criterion(
        n_poloidal_modes=equilibrium.mpol,
        max_toroidal_mode=equilibrium.ntor,
    )

    max_elongation = geometry_utils.max_elongation(
        equilibrium=equilibrium,
        n_poloidal_points=n_poloidal_points,
        n_toroidal_points=n_toroidal_points,
    )

    average_triangularity = geometry_utils.average_triangularity(
        surface=boundary,
    )

    # Magnetic metrics
    normalized_effective_radius_on_full_grid_mesh = np.sqrt(
        equilibrium.normalized_toroidal_flux_full_grid_mesh
    )
    iota = radial_profile.InterpolatedRadialProfile(
        rho=normalized_effective_radius_on_full_grid_mesh,
        values=equilibrium.iota_full,
    )
    axis_rotational_transform = float(
        radial_profile.evaluate_at_normalized_effective_radius(iota, np.array([0.0]))
    )
    edge_rotational_transform = float(
        radial_profile.evaluate_at_normalized_effective_radius(iota, np.array([1.0]))
    )

    vacuum_well = magnetics_utils.vacuum_well(equilibrium)

    magnetic_mirror_ratio = magnetics_utils.magnetic_mirror_ratio(equilibrium)
    axis_magnetic_mirror_ratio = float(
        radial_profile.evaluate_at_normalized_effective_radius(
            magnetic_mirror_ratio, np.array([0.0])
        )
    )
    edge_magnetic_mirror_ratio = float(
        radial_profile.evaluate_at_normalized_effective_radius(
            magnetic_mirror_ratio, np.array([1.0])
        )
    )

    phi_upper_bound = (
        2 * np.pi / equilibrium.n_field_periods / (1 + int(not equilibrium.lasym))
    )
    theta_phi = surface_utils.make_theta_phi_grid(
        n_theta=n_poloidal_points,
        n_phi=n_toroidal_points,
        phi_upper_bound=phi_upper_bound,
        include_endpoints=True,
    )
    # In a QI/QP configuration, the magnetic gradient scale length should scale
    # proportionally to (a A) / n_field_periods,
    # where a is the minor radius and A is the aspect ratio.
    # We normalize by the number of field periods to make the metric independent of
    # the configuration number of field periods.
    minimum_normalized_magnetic_gradient_scale_length = (
        np.min(
            magnetics_utils.normalized_magnetic_gradient_scale_length(
                equilibrium, theta_phi
            )
        )
        * equilibrium.n_field_periods
    )

    # QI metrics
    if settings.qi_settings is not None and settings.boozer_preset_settings is not None:
        boozer_settings = (
            boozer_module.create_boozer_settings_from_equilibrium_resolution(
                mhd_equilibrium=equilibrium, settings=settings.boozer_preset_settings
            )
        )
        boozer = boozer_module.run_boozer(
            equilibrium=equilibrium,
            settings=boozer_settings,
        )
        qi_metrics = qi.quasi_isodynamicity_residual(
            boozer=boozer,
            settings=settings.qi_settings,
        )
        qi_residuals = float(np.sum(qi_metrics.residuals**2))
    else:
        qi_residuals = None

    if settings.turbulent_settings is not None:
        flux_compression_in_regions_of_bad_curvature = (
            turbulent_transport.compute_flux_compression_in_regions_of_bad_curvature(
                equilibrium=equilibrium,
                settings=settings.turbulent_settings,
            )
        )
    else:
        flux_compression_in_regions_of_bad_curvature = None

    metrics = ConstellarationMetrics(
        aspect_ratio=equilibrium.aspect,
        aspect_ratio_over_edge_rotational_transform=equilibrium.aspect
        / edge_rotational_transform,
        max_elongation=max_elongation,
        edge_rotational_transform_over_n_field_periods=edge_rotational_transform
        / equilibrium.n_field_periods,
        axis_rotational_transform_over_n_field_periods=axis_rotational_transform
        / equilibrium.n_field_periods,
        average_triangularity=average_triangularity,
        vacuum_well=vacuum_well,
        axis_magnetic_mirror_ratio=axis_magnetic_mirror_ratio,
        edge_magnetic_mirror_ratio=edge_magnetic_mirror_ratio,
        minimum_normalized_magnetic_gradient_scale_length=minimum_normalized_magnetic_gradient_scale_length,
        qi=qi_residuals,
        flux_compression_in_regions_of_bad_curvature=flux_compression_in_regions_of_bad_curvature,
    )

    return metrics, equilibrium

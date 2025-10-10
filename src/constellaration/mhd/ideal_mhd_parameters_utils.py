"""Utilities for ideal-magnetohydrodynamic (MHD) parameters."""

import numpy as np
from scipy import constants as scipy_constants

from constellaration.geometry import surface_rz_fourier
from constellaration.mhd import flux_power_series, ideal_mhd_parameters
from constellaration.mhd import vmec_settings as vmec_settings_module
from constellaration.mhd import vmec_utils


def get_ideal_mhd_parameters_for_volume_averaged_beta(
    boundary: surface_rz_fourier.SurfaceRZFourier,
    volume_averaged_beta: float,
) -> ideal_mhd_parameters.IdealMHDParameters:
    """Create ideal-magnetohydrodynamic (MHD) parameters from a surface at the target
    volume-averaged beta using a linear pressure profile.

    The pressure profile is simply scaled by the required factor to achieve the
    target volume-averaged beta.

    Args:
        boundary: The boundary surface of the plasma.
        volume_averaged_beta: The target volume-averaged plasma beta.

    Returns:
        The ideal-MHD parameters, including pressure, toroidal current, and boundary
        toroidal flux.
    """
    # should we do like this?
    initial_mhd_parameters = ideal_mhd_parameters.boundary_to_ideal_mhd_parameters(
        boundary
    )
    pressure = initial_mhd_parameters.pressure
    boundary_toroidal_flux = initial_mhd_parameters.boundary_toroidal_flux

    if volume_averaged_beta == 0.0:
        # Instead of setting a null pressure profile, we set a pressure profile
        # with p(0) = 1.0 Pa.
        if pressure.coefficients[0] == 0.0:
            raise ValueError(
                "When setting volume_averaged_beta=0, then pressure.coefficients[0]"
                " can't be zero."
            )
        new_pressure = flux_power_series.FluxPowerSeriesProfile(
            coefficients=[c / pressure.coefficients[0] for c in pressure.coefficients]
        )
    else:
        volume_averaged_pressure = flux_power_series.evaluate_volume_average(pressure)
        if volume_averaged_pressure == 0.0:
            raise ValueError(
                "The provided pressure profile has a null volume-averaged pressure."
                " Please set a pressure profile with a non-null volume-averaged"
                " pressure."
            )
        minor_radius = float(
            surface_rz_fourier.to_simsopt(surface=boundary).minor_radius()
        )
        magnetic_field_strength_initial_guess = boundary_toroidal_flux / (
            np.pi * minor_radius**2
        )
        # <beta> ~ 2 mu_0 <p> / B**2
        # <p> = p0 / peaking_factor ~ <beta> B**2 / (2 mu_0)
        pressure_peaking_factor = (
            flux_power_series.evaluate_at_normalized_effective_radius(
                pressure, np.array([0.0])
            ).item()
            / volume_averaged_pressure
        )
        pressure_on_axis_initial_guess = (
            volume_averaged_beta
            * magnetic_field_strength_initial_guess**2
            / (2 * scipy_constants.mu_0)
            * pressure_peaking_factor
        )
        pressure_scale_factor_initial_guess = (
            pressure_on_axis_initial_guess
            / flux_power_series.evaluate_at_normalized_effective_radius(
                pressure, np.array([0.0])
            ).item()
        )
        pressure_initial_guess = flux_power_series.scale(
            profile=pressure,
            scale_factor=pressure_scale_factor_initial_guess,
        )
        ideal_mhd_parameters_initial_guess = ideal_mhd_parameters.IdealMHDParameters(
            pressure=pressure_initial_guess,
            toroidal_current=initial_mhd_parameters.toroidal_current,
            boundary_toroidal_flux=boundary_toroidal_flux,
        )
        vmec_settings = vmec_settings_module.vmec_settings_low_fidelity_fixed_boundary(
            boundary=boundary
        )
        vmecpp_wout_initial_guess = vmec_utils.run_vmec(
            boundary=boundary,
            mhd_parameters=ideal_mhd_parameters_initial_guess,
            vmec_settings=vmec_settings,
        )
        # Linear extrapolation of the pressure on axis.
        pressure_scale_factor = (
            volume_averaged_beta
            / vmecpp_wout_initial_guess.betatotal
            * pressure_scale_factor_initial_guess
        )
        new_pressure = flux_power_series.scale(
            profile=pressure,
            scale_factor=pressure_scale_factor,
        )

    return ideal_mhd_parameters.IdealMHDParameters(
        pressure=new_pressure,
        toroidal_current=initial_mhd_parameters.toroidal_current,
        boundary_toroidal_flux=initial_mhd_parameters.boundary_toroidal_flux,
    )

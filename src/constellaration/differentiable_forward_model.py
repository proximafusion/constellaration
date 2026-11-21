"""JAX path from a fixed-resolution design PyTree to equilibrium metrics."""

from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import mu_0
from vmecpp import autodiff_wout

from constellaration.boozer import jax_transform
from constellaration.forward_model import (
    ConstellarationMetrics,
    ConstellarationSettings,
)
from constellaration.geometry import surface_utils
from constellaration.mhd import (
    differentiable_vmec,
    ideal_mhd_parameters_utils,
    jax_elongation,
    jax_geometry,
    jax_gradient_length,
    jax_magnetics,
    jax_turbulent_transport,
    pressure_profile,
    vmec_settings,
    vmec_utils,
)
from constellaration.omnigeneity import jax_qi

if TYPE_CHECKING:
    from constellaration.problems import Design


def make_forward_model(
    reference: "Design",
    settings: ConstellarationSettings,
) -> Callable[["Design"], tuple[ConstellarationMetrics, dict[str, jax.Array]]]:
    """Freeze VMEC resolution and quadrature; return a JIT/grad-ready callable.

    Use float64 and pass designs with the same mode layout, field periods and
    pressure-profile degree as the reference. Non-converged VMEC solves raise;
    undefined metrics are not converted to artificial penalties.
    """
    boundary = reference.boundary
    nfp = boundary.n_field_periods
    resolution = settings.vmec_settings
    if resolution is None:
        resolution = vmec_settings.create_vmec_settings_from_preset(
            boundary, settings.vmec_preset_settings
        )
        mpol = max(resolution.n_poloidal_modes, boundary.n_poloidal_modes)
        ntor = max(resolution.max_toroidal_mode, boundary.max_toroidal_mode)
        (
            ntheta,
            nphi,
        ) = surface_utils.n_poloidal_toroidal_points_to_satisfy_nyquist_criterion(
            mpol, ntor
        )
        final = resolution.multigrid_steps[-1]
        resolution = resolution.model_copy(
            update=dict(
                n_poloidal_modes=mpol,
                max_toroidal_mode=ntor,
                n_poloidal_grid_points=max(ntheta, resolution.n_poloidal_grid_points),
                n_toroidal_grid_points=max(nphi, resolution.n_toroidal_grid_points),
                multigrid_steps=[
                    *resolution.multigrid_steps[:-1],
                    final.model_copy(
                        update={"force_tolerance": min(1e-18, final.force_tolerance)}
                    ),
                ],
            )
        )
    if (
        resolution.n_poloidal_modes < boundary.n_poloidal_modes
        or resolution.max_toroidal_mode < boundary.max_toroidal_mode
    ):
        raise ValueError("VMEC resolution would truncate design modes")
    template = vmec_utils.build_vmecpp_indata(
        ideal_mhd_parameters_utils.design_to_ideal_mhd_parameters(reference),
        boundary,
        resolution,
    )
    solve = differentiable_vmec.make_equilibrium_solver(template)
    ns = int(template.ns_array[-1])
    if ns < 5:
        raise ValueError("Differentiable metrics need at least five VMEC radial points")
    modes = {
        name: jnp.asarray(value)
        for name, value in autodiff_wout.static_fields(template).items()
        if name in ("xm", "xn", "xm_nyq", "xn_nyq")
    }
    (
        ntheta,
        nphi,
    ) = surface_utils.n_poloidal_toroidal_points_to_satisfy_nyquist_criterion(
        resolution.n_poloidal_modes, resolution.max_toroidal_mode
    )
    theta, phi = jnp.moveaxis(
        jnp.asarray(
            surface_utils.make_theta_phi_grid(
                ntheta, nphi, np.pi / nfp, include_endpoints=True
            )
        ),
        -1,
        0,
    )
    transform = None
    qi_settings = settings.qi_settings
    if settings.boozer_preset_settings is not None and qi_settings is not None:
        transform = jax_transform.make_transform(
            modes,
            nfp=nfp,
            ns=ns,
            mboz=max(6 * resolution.n_poloidal_modes, 2),
            nboz=max(2 * resolution.max_toroidal_mode - 1, 0),
            surfaces=settings.boozer_preset_settings.normalized_toroidal_flux,
            magnetic_only=True,
        )
    transport = settings.turbulent_settings
    surfaces = (
        None if transport is None else jnp.asarray(transport.normalized_toroidal_flux)
    )
    basis = (
        None
        if reference.pressure is None
        else jnp.asarray(
            pressure_profile.power_series_basis(len(reference.pressure.gradient_logits))
        )
    )
    vacuum_pressure = jnp.array([1.0, -1.0])
    temperature = settings.mercier_softmin_temperature

    def model(design):
        boundary = jnp.stack((design.boundary.r_cos, design.boundary.z_sin))
        area = jax_geometry.mean_cross_sectional_area(boundary, nfp)
        if basis is None:
            pressure = vacuum_pressure
            flux = area
        else:
            p0 = design.pressure.axis_pressure
            weights = jax.nn.softmax(jnp.stack((*design.pressure.gradient_logits, 0.0)))
            pressure = p0 * (basis @ weights)
            mean_pressure = jnp.sum(pressure / jnp.arange(1, pressure.size + 1))
            flux = area * (1 + mu_0 * (p0 - mean_pressure))
        wout = solve(boundary, pressure, flux)
        axis_mirror, edge_mirror = jax_magnetics.mirror_ratios(wout, modes, theta, phi)
        q = None
        if transform is not None:
            spectrum, _ = transform(wout)
            q = jax_qi.residual(spectrum, nfp, qi_settings, wout["bmnc"][1, 1])
        compression = None
        if transport is not None:
            compression = jax_turbulent_transport.flux_compression(
                wout,
                modes,
                surfaces,
                nfp,
                transport.n_field_lines,
                transport.n_toroidal_points,
            ).mean()
        metrics = ConstellarationMetrics.model_construct(
            aspect_ratio=wout["aspect"],
            aspect_ratio_over_edge_rotational_transform=wout["aspect"]
            / wout["iotaf"][-1],
            max_elongation=jax_elongation.max_elongation(
                wout, modes["xm"], modes["xn"], nfp, ntheta, nphi
            ),
            average_triangularity=jax_geometry.average_triangularity(boundary, nfp),
            axis_rotational_transform_over_n_field_periods=wout["iotaf"][0] / nfp,
            edge_rotational_transform_over_n_field_periods=wout["iotaf"][-1] / nfp,
            axis_magnetic_mirror_ratio=axis_mirror,
            edge_magnetic_mirror_ratio=edge_mirror,
            vacuum_well=jax_magnetics.vacuum_well(wout),
            minimum_normalized_magnetic_gradient_scale_length=nfp
            * jnp.min(
                jax_gradient_length.normalized_scale_length(wout, modes, theta, phi)
            ),
            minimum_normalized_mercier_criterion=jax_magnetics.mercier_softmin(
                wout, temperature
            ),
            magnetic_mirror_ratio_well=2
            * jnp.abs(edge_mirror - axis_mirror)
            / (edge_mirror + axis_mirror),
            volume_averaged_beta=wout["betatotal"],
            beta_peaking_factor=wout["betaxis"] / wout["betatotal"],
            qi=q,
            flux_compression_in_regions_of_bad_curvature=compression,
        )
        return metrics, wout

    return model

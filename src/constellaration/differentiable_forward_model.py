"""JAX path from a fixed-resolution design PyTree to equilibrium metrics."""

from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from vmecpp import autodiff_wout

from constellaration import forward_model
from constellaration.geometry import surface_utils
from constellaration.mhd import ideal_mhd_parameters_utils, vmec_settings, vmec_utils

if TYPE_CHECKING:
    from constellaration.problems import Design


def make_forward_model(
    reference: "Design",
    settings: forward_model.ConstellarationSettings,
) -> Callable[
    ["Design"], tuple[forward_model.ConstellarationMetrics, dict[str, jax.Array]]
]:
    """Freeze VMEC resolution and quadrature; return a JIT/grad-ready callable.

    Use float64 and pass designs with the same mode layout, field periods and
    pressure-profile degree as the reference. Non-converged VMEC solves raise;
    undefined metrics are not converted to artificial penalties.
    """
    boundary = reference.boundary
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
    solve = vmec_utils.make_equilibrium_solver(
        template, verbose=resolution.verbose, max_threads=resolution.max_threads
    )
    ns = int(template.ns_array[-1])
    if ns < 5:
        raise ValueError("Differentiable metrics need at least five VMEC radial points")
    static = autodiff_wout.static_fields(template)
    transform = forward_model._prepare_boozer_transform(
        vmec_utils.VmecppWOut.model_construct(**static), settings
    )

    def model(design):
        boundary = jnp.stack((design.boundary.r_cos, design.boundary.z_sin))
        pressure, flux = ideal_mhd_parameters_utils.design_pressure_and_flux(design)
        wout = solve(boundary, pressure, flux)
        equilibrium = vmec_utils.VmecppWOut.model_construct(**{**static, **wout})
        metrics = forward_model._metrics_from_equilibrium(
            equilibrium, settings, transform
        )
        return metrics, wout

    return model

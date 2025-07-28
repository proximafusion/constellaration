import concurrent.futures as futures
import functools
import multiprocessing
import os
import sys
from typing import Callable

import jaxtyping as jt
import nevergrad
import numpy as np
from jax import numpy as jnp
from simsopt import geo
from vmecpp import _pydantic_numpy as pydantic_numpy

from constellaration.boozer import boozer as boozer_module
from constellaration.data_generation import (
    rotational_transform_scaling,
    vmec_optimization_settings,
)
from constellaration.geometry import surface_rz_fourier, surface_utils
from constellaration.mhd import geometry_utils
from constellaration.mhd import ideal_mhd_parameters as ideal_mhd_parameters_module
from constellaration.mhd import vmec_settings as vmec_settings_module
from constellaration.mhd import vmec_utils
from constellaration.omnigeneity import omnigenity_field, omnigenity_field_sampling
from constellaration.utils import pytree


class OmnigenousFieldVmecOptimizationMetrics(pydantic_numpy.BaseModelWithNumpy):
    aspect_ratio: float
    """The aspect ratio of the configuration."""

    edge_iota: float
    """The edge iota of the configuration."""

    max_elongation: float
    """The maximum elongation of the configuration."""

    omnigenous_field_residuals: jt.Float[np.ndarray, " n_collocation_points"]
    """The residuals between the configuration magnetic field strength and the target
    omnigenous field."""

    modB_maxima_residuals: jt.Float[np.ndarray, " n_collocation_points"]
    """The residuals between the maxima of the magnetic field strength along a field
    line and its value at phi=0."""


pytree.register_pydantic_data(
    surface_rz_fourier.SurfaceRZFourier,
    meta_fields=["n_field_periods", "is_stellarator_symmetric"],
)


def optimize_boundary_omnigenity_vmec(
    targets: omnigenity_field_sampling.OmnigenousFieldAndTargets,
    settings: vmec_optimization_settings.OmnigenousFieldVmecOptimizationSettings,
) -> surface_rz_fourier.SurfaceRZFourier:
    boundary = _generate_initial_guess(
        targets=targets,
    )
    boundary = _refine_sample(
        boundary=boundary,
        targets=targets,
        settings=settings,
    )
    return boundary


def _generate_initial_guess(
    targets: omnigenity_field_sampling.OmnigenousFieldAndTargets,
) -> surface_rz_fourier.SurfaceRZFourier:
    simsopt_surface = geo.SurfaceRZFourier(
        nfp=targets.omnigenous_field.n_field_periods,
        stellsym=True,
        mpol=1,
        ntor=1,
    )
    assert targets.rotational_transform is not None
    assert targets.aspect_ratio is not None
    assert targets.max_elongation is not None
    torsion = rotational_transform_scaling.get_torsion_at_rotational_transform_over_n_field_periods(  # noqa: E501
        rotational_transform_over_n_field_periods=targets.rotational_transform
        / targets.omnigenous_field.n_field_periods,
        aspect_ratio=targets.aspect_ratio,
        elongation=targets.max_elongation,
    )
    simsopt_surface.make_rotating_ellipse(
        major_radius=1.0,
        minor_radius=1.0 / targets.aspect_ratio,
        elongation=targets.max_elongation,
        torsion=torsion,  # type: ignore
    )
    return surface_rz_fourier.from_simsopt(surface=simsopt_surface)


def _refine_sample(
    boundary: surface_rz_fourier.SurfaceRZFourier,
    targets: omnigenity_field_sampling.OmnigenousFieldAndTargets,
    settings: vmec_optimization_settings.OmnigenousFieldVmecOptimizationSettings,
) -> surface_rz_fourier.SurfaceRZFourier:
    boundary = surface_rz_fourier.set_max_mode_numbers(
        surface=boundary,
        max_poloidal_mode=settings.max_poloidal_mode,
        max_toroidal_mode=settings.max_toroidal_mode,
    )

    mask = surface_rz_fourier.build_mask(
        surface=boundary,
        max_poloidal_mode=settings.max_poloidal_mode,
        max_toroidal_mode=settings.max_toroidal_mode,
    )
    initial_guess, unravel_and_unmask_fn = pytree.mask_and_ravel(
        pytree=boundary,
        mask=mask,
    )

    scale = surface_rz_fourier.compute_infinity_norm_spectrum_scaling_fun(
        poloidal_modes=boundary.poloidal_modes.flatten(),
        toroidal_modes=boundary.toroidal_modes.flatten(),
        alpha=settings.infinity_norm_spectrum_scaling,
    ).reshape(boundary.poloidal_modes.shape)
    scale = np.concatenate([scale[mask.r_cos], scale[mask.z_sin]])

    x = np.copy(initial_guess) / scale

    func = functools.partial(
        _objective_function,
        scale=scale,
        unravel_and_unmask_fn=unravel_and_unmask_fn,
        targets=targets,
    )
    func = functools.partial(
        func,
        n_residuals=func(x).size,
    )

    for _ in range(settings.n_inner_optimizations):
        x = _nevergrad_minimize(
            fun=func,
            x=x,
            hypercube_bounds=settings.gradient_free_optimization_hypercube_bounds,
            budget_per_design_variable=settings.gradient_free_budget_per_design_variable,
            max_time=settings.gradient_free_max_time,
            verbose=settings.verbose,
        )

    return unravel_and_unmask_fn(jnp.asarray(x * scale))


def _objective_function(
    x: np.ndarray,
    scale: np.ndarray,
    unravel_and_unmask_fn: Callable[[np.ndarray], surface_rz_fourier.SurfaceRZFourier],
    targets: omnigenity_field_sampling.OmnigenousFieldAndTargets,
    n_residuals: int | None = None,
) -> jt.Float[np.ndarray, " n_residuals"]:
    boundary = unravel_and_unmask_fn(np.asarray(x * scale))
    try:
        metrics = _forward_model(
            boundary=boundary,
            omnigenous_field=targets.omnigenous_field,
        )
        assert targets.rotational_transform is not None
        assert targets.aspect_ratio is not None
        assert targets.max_elongation is not None
        return np.concatenate(
            [
                np.atleast_1d(
                    [
                        (metrics.aspect_ratio - targets.aspect_ratio)
                        / targets.aspect_ratio
                    ]
                ),
                np.atleast_1d(
                    [
                        (metrics.edge_iota - targets.rotational_transform)
                        / targets.rotational_transform
                    ]
                ),
                np.atleast_1d(
                    [
                        np.maximum(
                            metrics.max_elongation - targets.max_elongation,
                            0,
                        )
                        / targets.max_elongation
                    ]
                ),
                np.atleast_1d(metrics.omnigenous_field_residuals.flatten())
                / np.sqrt(metrics.omnigenous_field_residuals.size),
                np.atleast_1d(metrics.modB_maxima_residuals.flatten())
                / np.sqrt(metrics.modB_maxima_residuals.size),
            ]
        )
    except Exception as _:
        assert n_residuals is not None
        # If the forward model fails, we return a large residual.
        return np.sqrt(2 * 1e6 / n_residuals) * np.ones(n_residuals)


def _forward_model(
    boundary: surface_rz_fourier.SurfaceRZFourier,
    omnigenous_field: omnigenity_field.OmnigenousField,
) -> OmnigenousFieldVmecOptimizationMetrics:
    vmec_preset_settings = vmec_settings_module.VmecPresetSettings(
        fidelity="low_fidelity",
    )
    vmec_settings = vmec_settings_module.create_vmec_settings_from_preset(
        boundary,
        settings=vmec_preset_settings,
    )
    ideal_mhd_parameters = ideal_mhd_parameters_module.boundary_to_ideal_mhd_parameters(
        boundary
    )
    equilibrium = vmec_utils.run_vmec(
        boundary=boundary,
        mhd_parameters=ideal_mhd_parameters,
        vmec_settings=vmec_settings,
    )
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
    boozer_preset_settings = boozer_module.BoozerPresetSettings(
        normalized_toroidal_flux=[1.0],
    )
    boozer_settings = boozer_module.create_boozer_settings_from_equilibrium_resolution(
        mhd_equilibrium=equilibrium, settings=boozer_preset_settings
    )
    boozer = boozer_module.run_boozer(
        equilibrium=equilibrium,
        settings=boozer_settings,
    )
    theta, phi = omnigenity_field.get_theta_and_phi_boozer(
        field=omnigenous_field,
        n_alpha=n_poloidal_points,
        n_eta=n_toroidal_points,
        iota=boozer.iota[0],
    )
    target_modB = omnigenity_field.get_modb_boozer(
        field=omnigenous_field,
        rho=1.0,
        n_alpha=n_poloidal_points,
        n_eta=n_toroidal_points,
    )
    modB = _compute_modB(boozer, np.asarray(theta), np.asarray(phi))
    omnigenous_field_residuals = _compute_omnigenous_field_residuals(
        modB=modB,
        target_modB=np.asarray(target_modB),
    )
    modB_maxima_residuals = _compute_modB_maxima_reisudals(
        modB=modB,
        n_alpha=n_poloidal_points,
    )
    return OmnigenousFieldVmecOptimizationMetrics(
        aspect_ratio=equilibrium.aspect,
        edge_iota=float(equilibrium.iotaf[-1]),
        max_elongation=max_elongation,
        omnigenous_field_residuals=np.asarray(omnigenous_field_residuals),
        modB_maxima_residuals=np.asarray(modB_maxima_residuals),
    )


def _compute_omnigenous_field_residuals(
    modB: jt.Float[np.ndarray, " n_collocation_points"],
    target_modB: jt.Float[np.ndarray, " n_collocation_points"],
) -> jt.Float[np.ndarray, " n_collocation_points"]:
    # Scale both magnetic field strength such that the average is ~1T.
    scaled_modB = modB / np.mean(modB)
    scaled_target_modB = target_modB / np.mean(target_modB)
    return (scaled_modB - scaled_target_modB) / scaled_target_modB


def _compute_modB_maxima_reisudals(
    modB: jt.Float[np.ndarray, " n_collocation_points"],
    n_alpha: int,
) -> jt.Float[np.ndarray, " n_collocation_points"]:
    modB = modB.reshape(
        n_alpha,
        -1,
    )
    return np.max(modB, axis=1) - modB[:, 0]


def _compute_modB(
    boozer: boozer_module.BoozerOutput,
    theta: jt.Float[np.ndarray, " n_collocation_points"],
    phi: jt.Float[np.ndarray, " n_collocation_points"],
) -> jt.Float[np.ndarray, " n_collocation_points"]:
    angle = boozer.xm_b[:, None] * theta[None, :] - boozer.xn_b[:, None] * phi[None, :]
    return np.sum(boozer.bmnc_b[:, 0][:, None] * np.cos(angle), axis=0)


def _get_n_logical_cores() -> int:
    """Return the number of logical cores on the machine."""
    # See: https://askubuntu.com/questions/1292702/how-to-get-number-of-phy-logical-cores
    if "linux" in sys.platform:
        return int(
            os.popen("egrep '^core id' /proc/cpuinfo | sort -u | wc -l").read().strip()
        )
    return os.cpu_count() or 1


def _nevergrad_minimize(
    fun: Callable[
        [jt.Float[np.ndarray, " n_design_variables"]],
        jt.Float[np.ndarray, " n_residuals"],
    ],
    x: jt.Float[np.ndarray, " n_design_variables"],
    hypercube_bounds: float,
    budget_per_design_variable: int,
    max_time: float | None,
    verbose: bool,
) -> jt.Float[np.ndarray, " n_design_variables"]:
    """Minimize a vector valued function using Nevergrad.

    Args:
        fun: The vector valued function to minimize.
        x: The initial guess for the optimization.
        hypercube_bounds: The bounds of the unit hypercube for the optimization.
        budget_per_design_variable: The budget for the optimization per design variable.
        max_time: The maximum time to run the optimization for.
        verbose: Whether to print the optimization progress to the console.
    """
    lower_bounds = x - hypercube_bounds
    upper_bounds = x + hypercube_bounds
    parametrization = nevergrad.p.Array(
        init=np.asarray(x),
        lower=np.asarray(lower_bounds),
        upper=np.asarray(upper_bounds),
    )
    # Set random state from numpy.
    rng = np.random.RandomState()
    state = rng.get_state()
    parametrization.random_state.set_state(state)
    optimizer = nevergrad.optimizers.NGOpt(
        parametrization=parametrization,
        budget=budget_per_design_variable * len(x),
        num_workers=_get_n_logical_cores(),
    )
    optimizer.suggest(np.asarray(x))
    scalarized_func = functools.partial(
        _scalarize,
        func=fun,
    )
    mp_context = multiprocessing.get_context("forkserver")
    with futures.ProcessPoolExecutor(
        max_workers=optimizer.num_workers, mp_context=mp_context
    ) as executor:
        recommendation = optimizer.minimize(
            objective_function=scalarized_func,
            executor=executor,  # type: ignore
            verbosity=1 if verbose else 0,
            max_time=max_time,
        )
    return np.copy(recommendation.value)


def _scalarize(
    x: jt.Float[np.ndarray, " n_design_variables"],
    func: Callable[
        [jt.Float[np.ndarray, " n_design_variables"]],
        jt.Float[np.ndarray, " n_residuals"],
    ],
) -> float:
    """Scalarize the function value by taking the log of the sum of squares."""
    return np.log(0.5 * float(np.sum(func(x) ** 2)))

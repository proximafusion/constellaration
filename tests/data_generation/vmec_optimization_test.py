import numpy as np

from constellaration.data_generation import (
    vmec_optimization,
    vmec_optimization_settings,
)
from constellaration.omnigeneity import omnigenity_field_sampling

np.random.Generator(np.random.PCG64(0))


def test_nevergrad_minimize_simple_quadratic() -> None:
    x0 = np.array([0.0, 0.0])
    result = vmec_optimization._nevergrad_minimize(
        fun=_test_residual_fun,
        x=x0,
        hypercube_bounds=2.0,
        budget_per_design_variable=200,
        max_time=20.0,
        verbose=False,
    )
    # The minimum is at [2, -2]
    np.testing.assert_allclose(result, [2.0, -2.0], atol=0.0, rtol=1e-1)


def _test_residual_fun(x: np.ndarray) -> np.ndarray:
    return np.array([(x[0] - 2) ** 2 + (x[1] + 2) ** 2])


def test_optimize_boundary_omnigenity_vmec_with_sampled_targets() -> None:
    settings_sampler = (
        omnigenity_field_sampling.SampleOmnigenousFieldAndTargetsSettings(n_samples=1)
    )
    targets = settings_sampler.sample_omnigenous_fields_and_targets()[0]
    targets.omnigenous_field.n_field_periods = 4

    settings = vmec_optimization_settings.OmnigenousFieldVmecOptimizationSettings(
        infinity_norm_spectrum_scaling=1.0,
        max_poloidal_mode=1,
        max_toroidal_mode=2,
        n_inner_optimizations=1,
        gradient_free_optimization_hypercube_bounds=0.1,
        gradient_free_budget_per_design_variable=1,
        gradient_free_max_time=30,  # The MacOS 13 runners are a bit slow...
        verbose=False,
    )

    surface = vmec_optimization.optimize_boundary_omnigenity_vmec(
        targets=targets,
        settings=settings,
    )

    # Check that the surface is not a rotating ellipse
    assert surface.r_cos[0, 5] != 0.0
    assert surface.z_sin[0, 5] != 0.0

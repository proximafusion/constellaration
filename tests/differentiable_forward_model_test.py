import jax
import jax.numpy as jnp
import numpy as np
import vmecpp

from constellaration import differentiable_forward_model, problems
from constellaration.mhd import (
    ideal_mhd_parameters_utils,
    pressure_profile,
    vmec_settings,
)
from tests.equilibrium_reference_test import reference_inputs
from tests.utils.vmec_utils_test import assert_directional_derivative


def case():
    boundary, _, settings = reference_inputs("finite_beta")
    design = problems.Design(
        boundary=boundary,
        pressure=pressure_profile.IntegratedBernsteinProfile(
            axis_pressure=3000,
            gradient_logits=(0.35, -0.4),
        ),
    )
    settings.vmec_settings = vmec_settings.VmecSettings(
        n_poloidal_modes=4,
        max_toroidal_mode=3,
        n_poloidal_grid_points=20,
        n_toroidal_grid_points=18,
        multigrid_steps=[
            vmec_settings.VmecMultiGridStep(
                n_flux_surfaces=ns,
                force_tolerance=1e-21,
                n_max_iterations=18000,
            )
            for ns in (15, 29)
        ],
        time_step=0.5,
    )
    assert settings.qi_settings is not None
    assert settings.turbulent_settings is not None
    settings.qi_settings = settings.qi_settings.model_copy(
        update=dict(
            n_field_lines=19,
            n_toroidal_points=161,
            n_field_contours=101,
            n_output_toroidal_points=191,
        )
    )
    settings.turbulent_settings = settings.turbulent_settings.model_copy(
        update=dict(
            n_field_lines=29,
            n_toroidal_points=32,
        )
    )
    return design, settings


def test_design_inputs_match_vmec_pressure_and_flux():
    reference, settings = case()
    mhd = ideal_mhd_parameters_utils.design_to_ideal_mhd_parameters(reference)
    with vmecpp.enable_x64(True):
        model = jax.jit(
            differentiable_forward_model.make_forward_model(reference, settings)
        )
        metrics, wout = model(reference)
        assert wout["phi"].dtype == jnp.float64
        np.testing.assert_allclose(
            wout["phi"][-1], mhd.boundary_toroidal_flux, rtol=1e-12
        )
        s = (np.arange(wout["pres"].size - 1) + 0.5) / (wout["pres"].size - 1)
        pressure = np.polynomial.polynomial.polyval(s, mhd.pressure.coefficients)
        np.testing.assert_allclose(wout["pres"][1:], pressure, rtol=1e-12)
        assert metrics.qi is not None
        assert metrics.flux_compression_in_regions_of_bad_curvature is not None


def test_all_alm_objectives_and_constraints_are_differentiable():
    reference, settings = case()
    formulations = [
        problems.GeometricalProblem(),
        problems.SimpleToBuildQIStellarator(),
        problems.MHDStableQIStellarator(),
        problems.MHDStableQIStellaratorMedium(),
        problems.MHDStableQIStellaratorLoose(),
        problems.MaximumBetaQIStellarator(),
        problems.SimpleToBuildFiniteBetaQIStellarator(),
    ]
    profile = reference.pressure
    assert profile is not None
    with vmecpp.enable_x64(True):
        model = differentiable_forward_model.make_forward_model(reference, settings)
        r = jnp.asarray(reference.boundary.r_cos)

        def evaluate(t):
            boundary = reference.boundary.model_copy(
                update={"r_cos": r.at[1, 1].add(0.06 * t)}
            )
            pressure = profile.model_copy(
                update={
                    "axis_pressure": profile.axis_pressure * jnp.exp(0.20 * t),
                    "gradient_logits": (0.35 + 0.1 * t, -0.4),
                }
            )
            metrics, _ = model(
                problems.Design.model_construct(boundary=boundary, pressure=pressure)
            )
            values = [
                problem.alm_objective_constraints(metrics) for problem in formulations
            ]
            return jnp.concatenate(
                [jnp.concatenate((jnp.atleast_1d(obj), cons)) for obj, cons in values]
            )

        assert_directional_derivative(evaluate, steps=(2e-5, 1e-5))

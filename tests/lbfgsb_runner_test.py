import jax
import jax.numpy as jnp
import numpy as np
import pytest
import vmecpp

from constellaration import problems
from constellaration.optimization import augmented_lagrangian as al
from constellaration.optimization import augmented_lagrangian_runner as runner
from constellaration.optimization import settings
from tests.differentiable_forward_model_test import case


def test_finite_beta_lbfgsb_moves_pressure_and_boundary():
    design, forward_settings = case()
    forward_settings.turbulent_settings = None
    problem = problems.MaximumBetaQIStellarator()
    optimizer = settings.AugmentedLagrangianMethodSettings(
        maxit=1,
        penalty_parameters_initial=1e-16,
        bounds_initial=5e-4,
        augmented_lagrangian_settings=al.AugmentedLagrangianSettings(),
        oracle_settings=settings.LBFGSBSettings(maxiter=1, maxfun=4, gtol=1e-10),
    )
    config = settings.OptimizationSettings(
        optimizer_settings=optimizer,
        forward_model_settings=forward_settings,
        max_poloidal_mode=1,
        max_toroidal_mode=1,
        infinity_norm_spectrum_scaling=1.5,
        log_axis_pressure_scaling=2.0,
        pressure_gradient_logits_scaling=0.5,
    )
    initial, initial_metrics = runner.run(
        design,
        config.model_copy(
            update={"optimizer_settings": optimizer.model_copy(update={"maxit": 0})}
        ),
        problem,
    )
    final, final_metrics = runner.run(design, config, problem)
    assert initial_metrics is not None
    assert final_metrics is not None
    assert initial.pressure is not None
    assert final.pressure is not None
    assert final.pressure.axis_pressure > initial.pressure.axis_pressure
    assert final.pressure.gradient_logits != initial.pressure.gradient_logits
    assert not np.array_equal(final.boundary.r_cos, initial.boundary.r_cos)
    assert initial_metrics.volume_averaged_beta is not None
    assert final_metrics.volume_averaged_beta is not None
    assert final_metrics.volume_averaged_beta > initial_metrics.volume_averaged_beta
    assert (
        problems.Design.model_validate_json(final.model_dump_json()).pressure
        == final.pressure
    )


def test_nonfinite_inner_gradient_is_an_error():
    with vmecpp.enable_x64(True):
        state = al.AugmentedLagrangianState(
            x=jnp.zeros(1),
            multipliers=jnp.zeros(1),
            penalty_parameters=jnp.ones(1),
            objective=jnp.zeros(()),
            constraints=jnp.zeros(1),
            bounds=jnp.ones(1),
        )

        def nondifferentiable(x, state):
            # Finite value but singular derivative at x=0.
            value = jnp.sqrt(x[0])
            return value, (value, state.constraints)

        with pytest.raises(FloatingPointError, match="Non-finite"):
            runner._lbfgsb_step(
                state,
                settings.LBFGSBSettings(),
                jax.value_and_grad(nondifferentiable, has_aux=True),
            )

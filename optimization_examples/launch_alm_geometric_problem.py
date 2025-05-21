import constellaration.forward_model as forward_model
import constellaration.initial_guess as init
import constellaration.optimization.augmented_lagrangian as al
import constellaration.optimization.augmented_lagrangian_runner as runner
import constellaration.optimization.settings as optimization_settings
import constellaration.problems as problems
from constellaration.mhd import vmec_settings as vmec_settings_module
from constellaration.utils import seed_util

OPTIMIZATION_SETTINGS = optimization_settings.OptimizationSettings(
    max_poloidal_mode=4,
    max_toroidal_mode=4,
    infinity_norm_spectrum_scaling=1.5,
    optimizer_settings=optimization_settings.AugmentedLagrangianMethodSettings(
        maxit=40,
        oracle_settings=optimization_settings.NevergradSettings(
            num_workers=48,
            budget_initial=1500,
            budget_max=20000,
            budget_increment=260,
            max_time=None,
        ),
        penalty_parameters_initial=10,
        bounds_initial=0.5,
        augmented_lagrangian_settings=al.AugmentedLagrangianSettings(
            constraint_violation_tolerance_reduction_factor=0.8,
            penalty_parameters_increase_factor=5,
            bounds_reduction_factor=0.9,
            penalty_parameters_max=1e8,
            bounds_min=0.05,
        ),
    ),
    forward_model_settings=forward_model.ConstellarationSettings(
        qi_settings=None,
        turbulent_settings=None,
        vmec_preset_settings=vmec_settings_module.VmecPresetSettings(
            fidelity="low_fidelity",
        ),
    ),
)


def main():
    seed_util.seed_everything(123)
    problem = problems.GeometricalProblem()
    boundary = init.generate_rotating_ellipse(
        aspect_ratio=4,
        elongation=3.0,
        rotational_transform=0.3 * 3,
        n_field_periods=3,
    )

    runner.run(boundary, OPTIMIZATION_SETTINGS, problem)


if __name__ == "__main__":
    main()

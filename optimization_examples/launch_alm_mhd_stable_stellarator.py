import constellaration.forward_model as forward_model
import constellaration.initial_guess as init
import constellaration.optimization.augmented_lagrangian as al
import constellaration.optimization.augmented_lagrangian_runner as runner
import constellaration.optimization.settings as optimization_settings
import constellaration.problems as problems
import constellaration.utils.seed_util as seed_util
from constellaration.mhd import vmec_settings as vmec_settings_module

OPTIMIZATION_SETTINGS = optimization_settings.OptimizationSettings(
    max_poloidal_mode=4,
    max_toroidal_mode=4,
    infinity_norm_spectrum_scaling=1.5,
    optimizer_settings=optimization_settings.AugmentedLagrangianMethodSettings(
        maxit=50,
        oracle_settings=optimization_settings.NevergradSettings(
            num_workers=48,
            budget_initial=1500,
            budget_max=20000,
            budget_increment=300,
            max_time=None,
        ),
        penalty_parameters_initial=10,
        bounds_initial=0.33,
        augmented_lagrangian_settings=al.AugmentedLagrangianSettings(
            constraint_violation_tolerance_reduction_factor=0.8,
            penalty_parameters_increase_factor=5,
            bounds_reduction_factor=0.98,
            penalty_parameters_max=1e8,
            bounds_min=0.05,
        ),
    ),
    forward_model_settings=forward_model.ConstellarationSettings(
        vmec_preset_settings=vmec_settings_module.VmecPresetSettings(
            fidelity="very_low_fidelity",
        )
    ),
)


ASPECT_RATIO_UPPER_BOUNDS = [6, 8.0, 10.0, 12.0]


def main():
    for aspect_ratio_upper_bound in ASPECT_RATIO_UPPER_BOUNDS:
        seed_util.seed_everything(123)

        problem = problems.MHDStableQIStellarator()
        boundary = init.generate_nae(
            aspect_ratio=20.0 * 0.25 * 3,
            max_elongation=4.0,
            rotational_transform=0.25 * 3,
            mirror_ratio=0.25,
            n_field_periods=3,
            max_poloidal_mode=1,
            max_toroidal_mode=1,
        )

        runner.run(boundary, OPTIMIZATION_SETTINGS, problem, aspect_ratio_upper_bound)


if __name__ == "__main__":
    main()

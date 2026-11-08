"""Optimize a maximum-beta QI stellarator with the augmented Lagrangian method."""

from constellaration import forward_model, initial_guess, problems
from constellaration.mhd import pressure_profile
from constellaration.mhd import vmec_settings as vmec_settings_module
from constellaration.optimization import augmented_lagrangian as al
from constellaration.optimization import augmented_lagrangian_runner as runner
from constellaration.optimization import settings as optimization_settings
from constellaration.utils import seed_util

OPTIMIZATION_SETTINGS = optimization_settings.OptimizationSettings(
    max_poloidal_mode=4,
    max_toroidal_mode=4,
    infinity_norm_spectrum_scaling=1.5,
    log_axis_pressure_scaling=10.0,
    pressure_gradient_logits_scaling=1.0,
    optimizer_settings=optimization_settings.AugmentedLagrangianMethodSettings(
        maxit=50,
        oracle_settings=optimization_settings.NevergradSettings(
            num_workers=8,
            budget_initial=20_000,
            budget_max=20_000,
            budget_increment=0,
            max_time=None,
            batch_mode=False,
        ),
        penalty_parameters_initial=10,
        bounds_initial=0.33,
        augmented_lagrangian_settings=al.AugmentedLagrangianSettings(),
    ),
    forward_model_settings=forward_model.ConstellarationSettings(
        vmec_preset_settings=vmec_settings_module.VmecPresetSettings(
            fidelity="very_low_fidelity",
        ),
        turbulent_settings=None,
    ),
)


def main() -> None:
    seed_util.seed_everything(123)
    problem = problems.MaximumBetaQIStellarator()
    boundary = initial_guess.generate_nae(
        aspect_ratio=12,
        max_elongation=7,
        rotational_transform=1.0,
        mirror_ratio=0.2,
        n_field_periods=5,
        # A smoother seed avoids a bad VMEC Jacobian; optimization still uses (4, 4).
        max_poloidal_mode=2,
        max_toroidal_mode=2,
    )
    design = problems.Design(
        boundary=boundary,
        pressure=pressure_profile.IntegratedBernsteinProfile(
            # Calibrated at the configured VMEC fidelity: beta ~5%, beta peaking ~3.
            axis_pressure=84_100,
            gradient_logits=(1.4435, 0, -1),
        ),
    )
    runner.run(design, OPTIMIZATION_SETTINGS, problem)


if __name__ == "__main__":
    main()

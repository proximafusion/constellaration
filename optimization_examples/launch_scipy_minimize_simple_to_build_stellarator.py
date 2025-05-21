import constellaration.forward_model as forward_model
import constellaration.initial_guess as init
import constellaration.optimization.scipy_minimize_runner as runner
import constellaration.optimization.settings as optimization_settings
import constellaration.problems as problems
import constellaration.utils.seed_util as seed_util
from constellaration.mhd import vmec_settings as vmec_settings_module

OPTIMIZATION_SETTINGS = optimization_settings.OptimizationSettings(
    max_poloidal_mode=4,
    max_toroidal_mode=4,
    infinity_norm_spectrum_scaling=1.5,
    optimizer_settings=optimization_settings.ScipyMinimizeSettings(
        method="COBYQA",
        # method="trust-constr",
        options={
            "maxiter": 10000,  # "gtol": 1e-15, "xtol": 1e-15, "barrier_tol": 1e-15
        },
    ),
    forward_model_settings=forward_model.ConstellarationSettings(
        vmec_preset_settings=vmec_settings_module.VmecPresetSettings(
            fidelity="low_fidelity",
        )
    ),
)


def main():
    seed_util.seed_everything(123)
    problem = problems.SimpleToBuildQIStellarator()
    boundary = init.generate_nae(
        aspect_ratio=10,
        max_elongation=5.0,
        rotational_transform=0.25 * 3,
        mirror_ratio=0.25,
        n_field_periods=3,
        max_poloidal_mode=1,
        max_toroidal_mode=1,
    )
    runner.run(boundary, OPTIMIZATION_SETTINGS, problem)


if __name__ == "__main__":
    main()

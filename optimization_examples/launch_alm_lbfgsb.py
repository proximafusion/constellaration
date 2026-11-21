"""Fixed-boundary VMEC++ optimization with an exact equilibrium adjoint.

Run: python -m optimization_examples.launch_alm_lbfgsb
"""

import numpy as np
import vmecpp

from constellaration import (
    differentiable_forward_model,
    forward_model,
    initial_guess,
    problems,
)
from constellaration.mhd import vmec_settings
from constellaration.optimization import augmented_lagrangian as al
from constellaration.optimization import augmented_lagrangian_runner, settings


def main():
    design = problems.Design(
        boundary=initial_guess.generate_nae(
            aspect_ratio=10,
            max_elongation=5,
            rotational_transform=0.75,
            mirror_ratio=0.25,
            n_field_periods=3,
            max_poloidal_mode=1,
            max_toroidal_mode=1,
        )
    )
    problem = problems.GeometricalProblem()
    forward_settings = forward_model.ConstellarationSettings(
        vmec_settings=vmec_settings.VmecSettings(
            n_poloidal_modes=4,
            max_toroidal_mode=3,
            n_poloidal_grid_points=20,
            n_toroidal_grid_points=18,
            multigrid_steps=[
                vmec_settings.VmecMultiGridStep(
                    n_flux_surfaces=ns,
                    force_tolerance=1e-18,
                    n_max_iterations=18000,
                )
                for ns in (15, 29)
            ],
            time_step=0.5,
        ),
        boozer_preset_settings=None,
        qi_settings=None,
        turbulent_settings=None,
    )
    options = settings.OptimizationSettings(
        max_poloidal_mode=1,
        max_toroidal_mode=1,
        infinity_norm_spectrum_scaling=1.5,
        optimizer_settings=settings.AugmentedLagrangianMethodSettings(
            maxit=2,
            penalty_parameters_initial=0.02,
            bounds_initial=0.004,
            augmented_lagrangian_settings=al.AugmentedLagrangianSettings(
                bounds_min=0.002
            ),
            oracle_settings=settings.LBFGSBSettings(maxiter=4, maxfun=12),
        ),
        forward_model_settings=forward_settings,
    )
    with vmecpp.enable_x64(True):
        model = differentiable_forward_model.make_forward_model(
            design, forward_settings
        )
        start, initial_constraints = problem.alm_objective_constraints(model(design)[0])
        _, metrics = augmented_lagrangian_runner.run(design, options, problem)
        assert metrics is not None
        end, final_constraints = problem.alm_objective_constraints(metrics)
    print(f"Maximum elongation: {start:.6f} -> {end:.6f}")
    print("Positive constraint violations (aspect ratio, triangularity, edge iota):")
    print(f"  initial: {np.maximum(0, initial_constraints)}")
    print(f"  final:   {np.maximum(0, final_constraints)}")
    if not end < start:
        raise RuntimeError("The example did not improve the objective")


if __name__ == "__main__":
    main()

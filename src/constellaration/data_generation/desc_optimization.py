from desc import equilibrium as desc_equilibrium
from desc import geometry as desc_geometry
from desc import objectives as desc_objectives
from desc import optimize as desc_optimize

from constellaration import initial_guess as initial_guess_module
from constellaration.data_generation import (
    desc_optimization_settings as settings_module,
)
from constellaration.geometry import surface_rz_fourier, surface_utils_desc
from constellaration.omnigeneity import (
    omnigenity_field,
    omnigenity_field_desc,
    omnigenity_field_sampling,
)


def optimize_boundary_omnigenity_desc(
    targets: omnigenity_field_sampling.OmnigenousFieldAndTargets,
    settings: settings_module.DescOmnigenousFieldOptimizationSettings,
) -> surface_rz_fourier.SurfaceRZFourier:
    """Optimize a boundary for a target omnigenous field."""

    equilibrium = _optimize_equilibrium_for_omnigenity_desc(
        targets=targets,
        settings=settings,
    )

    assert isinstance(equilibrium.surface, desc_geometry.FourierRZToroidalSurface)
    return surface_utils_desc.from_desc_fourier_rz_toroidal_surface(equilibrium.surface)


def generate_qp_initialization_from_targets(
    targets: omnigenity_field_sampling.OmnigenousFieldAndTargets,
    settings: settings_module.QPInitialGuessSettings,
) -> surface_rz_fourier.SurfaceRZFourier:
    assert targets.rotational_transform is not None
    new_settings = settings.model_copy(
        update=dict(
            n_field_periods=targets.omnigenous_field.n_field_periods,
            aspect_ratio=targets.aspect_ratio or settings.aspect_ratio,
            mirror_ratio=_get_mirror_ratio_from_field(targets.omnigenous_field),
            is_iota_positive=targets.rotational_transform < 0.0,
        )
    )
    return get_qp_initial_guess(settings=new_settings)


def generate_nae_initialization_from_targets(
    targets: omnigenity_field_sampling.OmnigenousFieldAndTargets,
    settings: settings_module.NearAxisExpansionInitialGuessSettings,
) -> surface_rz_fourier.SurfaceRZFourier:
    new_settings = settings.model_copy(
        update=dict(
            n_field_periods=targets.omnigenous_field.n_field_periods,
            aspect_ratio=targets.aspect_ratio or settings.aspect_ratio,
            max_elongation=targets.max_elongation or settings.max_elongation,
            rotational_transform=(
                targets.rotational_transform or settings.rotational_transform
            ),
            mirror_ratio=_get_mirror_ratio_from_field(targets.omnigenous_field),
        ),
    )
    return get_near_axis_initial_guess(settings=new_settings)


def _optimize_equilibrium_for_omnigenity_desc(
    targets: omnigenity_field_sampling.OmnigenousFieldAndTargets,
    settings: settings_module.DescOmnigenousFieldOptimizationSettings,
) -> desc_equilibrium.Equilibrium:
    """Optimize a boundary for a target omnigenous field."""

    if isinstance(
        settings.initial_guess_settings, settings_module.QPInitialGuessSettings
    ):
        initial_guess_surface = generate_qp_initialization_from_targets(
            settings=settings.initial_guess_settings, targets=targets
        )
    elif isinstance(
        settings.initial_guess_settings,
        settings_module.NearAxisExpansionInitialGuessSettings,
    ):
        initial_guess_surface = generate_nae_initialization_from_targets(
            settings=settings.initial_guess_settings, targets=targets
        )
    else:
        raise ValueError(
            f"Unknown initial guess settings type: "
            f"{type(settings.initial_guess_settings)}"
        )

    # Create equilibrium object
    equilibrium = create_desc_equilibrium_object(
        surface=initial_guess_surface,
        settings=settings.equilibrium_settings,
    )

    # Solve the equilibrium
    equilibrium = solve_desc_equilibrium(
        desc_equilibrium_object=equilibrium,
    )

    # Objectives and constraints
    objective = create_desc_objective_function(
        equilibrium=equilibrium,
        targets=targets,
        settings=settings.objective_settings,
    )
    constraints = create_desc_constraints(
        equilibrium_object=equilibrium,
    )

    # Optimizer
    optimizer = create_desc_optimizer(
        settings=settings.optimizer_settings,
    )

    # Run the optimization
    (equilibrium,), _ = optimizer.optimize(
        things=equilibrium,
        objective=objective,
        constraints=constraints,
        maxiter=settings.optimizer_settings.maxiter,
        verbose=settings.optimizer_settings.verbose,
    )
    # Solve the equilibrium again
    equilibrium = solve_desc_equilibrium(
        desc_equilibrium_object=equilibrium,
    )

    return equilibrium


def get_qp_initial_guess(
    settings: settings_module.QPInitialGuessSettings,
) -> surface_rz_fourier.SurfaceRZFourier:
    """Get an initial guess boundary from settings."""
    return surface_utils_desc.from_qp_model(
        major_radius=settings.major_radius,
        aspect_ratio=settings.aspect_ratio,
        elongation=settings.elongation,
        mirror_ratio=settings.mirror_ratio,
        torsion=settings.torsion,
        n_field_periods=settings.n_field_periods,
        is_stellarator_symmetric=settings.is_stellarator_symmetric,
        is_iota_positive=settings.is_iota_positive,
    )


def get_near_axis_initial_guess(
    settings: settings_module.NearAxisExpansionInitialGuessSettings,
) -> surface_rz_fourier.SurfaceRZFourier:
    return initial_guess_module.generate_nae(
        aspect_ratio=settings.aspect_ratio,
        max_elongation=settings.max_elongation,
        rotational_transform=settings.rotational_transform,
        mirror_ratio=settings.mirror_ratio,
        n_field_periods=settings.n_field_periods,
        max_poloidal_mode=settings.max_poloidal_mode,
        max_toroidal_mode=settings.max_toroidal_mode,
    )


def create_desc_equilibrium_object(
    surface: surface_rz_fourier.SurfaceRZFourier,
    settings: settings_module.DescEquilibriumSettings,
) -> desc_equilibrium.Equilibrium:
    """Get the DESC equilibrium object from the surface and settings."""
    desc_surface = surface_utils_desc.to_desc_fourier_rz_toroidal_surface(surface)
    return desc_equilibrium.Equilibrium(
        Psi=settings.psi,
        surface=desc_surface,
        M=settings.max_poloidal_mode,
        N=settings.max_toroidal_mode,
        check_orientation=settings.check_orientation,
    )


def solve_desc_equilibrium(
    desc_equilibrium_object: desc_equilibrium.Equilibrium,
) -> desc_equilibrium.Equilibrium:
    """Solve the DESC equilibrium object."""

    assert hasattr(desc_equilibrium_object, "solve")
    eq, _ = desc_equilibrium_object.solve(objective="force", verbose=3)
    return eq


def create_desc_objective_function(
    equilibrium: desc_equilibrium.Equilibrium,
    targets: omnigenity_field_sampling.OmnigenousFieldAndTargets,
    settings: settings_module.DescObjectiveFunctionSettings,
) -> desc_objectives.ObjectiveFunction:
    """Create the objective function for the DESC equilibrium object."""
    desc_field = omnigenity_field_desc.omnigenous_field_to_desc(
        targets.omnigenous_field
    )
    return settings.get_objective(
        equilibrium=equilibrium,
        field=desc_field,
        aspect_ratio=targets.aspect_ratio,
        elongation=targets.max_elongation,
        rotational_transform=targets.rotational_transform,
    )


def create_desc_constraints(
    equilibrium_object: desc_equilibrium.Equilibrium,
) -> tuple[desc_objectives.objective_funs._Objective, ...]:
    assert isinstance(
        equilibrium_object.surface, desc_geometry.FourierRZToroidalSurface
    )
    idx_r00 = equilibrium_object.surface.R_basis.get_idx(M=0, N=0)  # Fix major radius
    r_modes_to_fix = equilibrium_object.surface.R_basis.modes[idx_r00]
    return (
        desc_objectives.FixBoundaryR(eq=equilibrium_object, modes=r_modes_to_fix),
        desc_objectives.CurrentDensity(
            eq=equilibrium_object
        ),  # vacuum equilibrium force balance
        desc_objectives.FixPressure(
            eq=equilibrium_object
        ),  # fix vacuum pressure profile
        desc_objectives.FixCurrent(eq=equilibrium_object),  # fix vacuum current profile
        desc_objectives.FixPsi(
            eq=equilibrium_object
        ),  # fix total toroidal magnetic flux
    )


def create_desc_optimizer(
    settings: settings_module.DescOptimizerSettings,
) -> desc_optimize.Optimizer:
    """Create the DESC optimizer."""
    return desc_optimize.Optimizer(settings.name)


def _get_mirror_ratio_from_field(field: omnigenity_field.OmnigenousField):
    min_ = field.modB_spline_knot_coefficients[0, 0]
    max_ = field.modB_spline_knot_coefficients[0, -1]
    return (max_ - min_) / (min_ + max_)

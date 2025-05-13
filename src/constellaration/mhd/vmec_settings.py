from typing import Literal

import pydantic
from constellaration.geometry import surface_rz_fourier, surface_utils


class VmecMultiGridStep(pydantic.BaseModel):
    n_flux_surfaces: int
    force_tolerance: float
    n_max_iterations: int


class VmecSettings(pydantic.BaseModel):
    n_poloidal_modes: int
    """Corresponds to INDATA's mpol.

    VMEC will use Fourier coefficients with modes in the inclusive range
    range [0, n_poloidal_modes - 1]
    """
    max_toroidal_mode: int
    """Corresponds to INDATA's ntor.

    VMEC will use Fourier coefficients with modes in the inclusive range
    [-max_toroidal_mode_number, max_toroidal_mode_number].
    """
    n_poloidal_grid_points: int
    """Corresponds to VMEC's 'ntheta'."""
    n_toroidal_grid_points: int
    """Corresponds to VMEC's 'nzeta'."""
    multigrid_steps: list[VmecMultiGridStep]
    time_step: float

    full_vacuum_field_update_every: int = 1
    """Interval for performing a full update of the free-boundary computation.

    Values greater than 1 mean that the matrix in the Nestor module is re-used for the
    next `nvacskip` iterations after the one that computed it.

    Corresponds to VMEC's `nvacskip`.
    """
    verbose: bool = False
    """Whether to print VMEC's output to stdout."""

    max_threads: int = 1
    """Number of threads to use for running VMEC.

    Note that multithreaded runs are no longer perfectly reproducible, because
    truncation errors accumulate differently based on the completion order of threads!
    Keep this at 1 for reliable gradient estimation with finite differences.
    """


# TODO(mariap): remove this class
class VmecPresetSettings(pydantic.BaseModel):
    """Presets to derive the parameters of VmecSettings from a boundary."""

    fidelity: Literal[
        "from_boundary_resolution",
        "high_fidelity",
        "low_fidelity",
        "ultra_low_fidelity",
    ] = "from_boundary_resolution"
    """Available presets:

     * `from_boundary_resolution`: Matches VMEC++'s internal resolution to the
            boundary's Fourier resolution. Optimizes for VMEC convergence rate over
            runtime and high fidelity.
     * `high_fidelity`: Optimizes for correctness/fidelity over runtime and VMEC
            convergence rate.
     * `low_fidelity`: Optimizes for runtime over correctness/fidelity and VMEC
            convergence rate.
    * `ultra_low_fidelity`: Optimizes for runtime over correctness/fidelity and
            VMEC convergence rate, but with a very low resolution. This is meant to be
            used for very fast convergence in optimization tasks where.

    For details on the presets meaning, please refer to the respective factory functions
    in `mhd/vmec_settings_utils.py`.

    """
    verbose: bool = False
    """Whether to print VMEC++'s output to stdout."""

    multithreaded: bool = False
    """Whether to run VMEC++ in multithreaded mode, in which case it will spawn as many
    threads as the number of logical CPU cores.

    Note that if CPU cores are overcommitted (i.e. the cores are shared with other
    running processes), performance could degrade significantly even with respect to
    single-threaded runs: only turn this on if VMEC++ will be the only process running
    on the machine.

    Also note that multithreaded runs are not perfectly reproducible because of
    concurrency and floating point arithmetic.
    """


def create_vmec_settings_from_preset(
    boundary: surface_rz_fourier.SurfaceRZFourier,
    settings: VmecPresetSettings,
) -> VmecSettings:
    """Derives VmecSettings from the boundary of a PlasmaConfiguration according to the
    given fidelity preset.

    This task provides a standardized API to derive the VMEC resolution parameters from
    a boundary. Multiple fidelity presets are available that provide different tradeoffs
    between correctness, execution time, and convergence rate.

    The MakegridOutput is required if and only if the  settings for free boundary are
    requested.
    """

    if settings.fidelity == "high_fidelity":
        vmec_settings = vmec_settings_high_fidelity_fixed_boundary(
            boundary=boundary,
        )
    elif settings.fidelity == "low_fidelity":
        vmec_settings = vmec_settings_low_fidelity_fixed_boundary(
            boundary=boundary,
        )
    elif settings.fidelity == "from_boundary_resolution":
        vmec_settings = vmec_settings_from_boundary_resolution(
            boundary=boundary,
        )
    elif settings.fidelity == "ultra_low_fidelity":
        vmec_settings = vmec_settings_ultra_low_fidelity_fixed_boundary(
            boundary=boundary,
        )
    else:
        raise ValueError(f"Unknown settings fidelity: {settings.fidelity}")

    vmec_settings.verbose = settings.verbose

    if settings.multithreaded:
        # this tells VMEC++ to use as many threads as there are logical CPU cores
        # VMEC++ handles max_threads>num_cores gracefully,
        # and 16 is a sensible default to avoid over-subscription.
        vmec_settings.max_threads = 16
    else:
        vmec_settings.max_threads = 1

    return vmec_settings


def vmec_settings_high_fidelity_fixed_boundary(
    boundary: surface_rz_fourier.SurfaceRZFourier,
) -> VmecSettings:
    """Sensible high-fidelity VMEC settings for fixed-boundary runs.

    Optimizes for correctness/fidelity over runtime and VMEC convergence rate.

    This is meant to be used to guarantee metrics are computed in a standardized manner
    or as a good (high-fidelity) default in case no better choice is available.
    """
    (
        largest_non_zero_poloidal_mode,
        largest_non_zero_toroidal_mode,
    ) = surface_rz_fourier.get_largest_non_zero_modes(surface=boundary)

    n_poloidal_modes = max(10, largest_non_zero_poloidal_mode)
    max_toroidal_mode = max(10, largest_non_zero_toroidal_mode)

    (
        n_poloidal_points,
        n_toroidal_points,
    ) = surface_utils.n_poloidal_toroidal_points_to_satisfy_nyquist_criterion(
        n_poloidal_modes=n_poloidal_modes,
        max_toroidal_mode=max_toroidal_mode,
    )

    return VmecSettings(
        n_poloidal_modes=n_poloidal_modes,
        max_toroidal_mode=max_toroidal_mode,
        n_poloidal_grid_points=n_poloidal_points,
        n_toroidal_grid_points=n_toroidal_points,
        multigrid_steps=[
            VmecMultiGridStep(
                n_flux_surfaces=25, force_tolerance=1e-17, n_max_iterations=2000
            ),
            VmecMultiGridStep(
                n_flux_surfaces=51, force_tolerance=1e-17, n_max_iterations=2000
            ),
            VmecMultiGridStep(
                n_flux_surfaces=99, force_tolerance=1e-15, n_max_iterations=40000
            ),
        ],
        time_step=0.5,
    )


def vmec_settings_low_fidelity_fixed_boundary(
    boundary: surface_rz_fourier.SurfaceRZFourier,
) -> VmecSettings:
    """A sensible low-fidelity VMEC configuration that provides defaults for the given
    boundary by matching the boundary's Fourier resolution to VMEC++'s internal
    resolution.

    Optimization tasks might default to these settings to speed up convergence.
    """
    # The criterion we use below to set n_toroidal_grid_points breaks down for ntor == 0
    assert boundary.max_toroidal_mode != 0, "axisymmetric boundaries are not supported"

    (
        largest_non_zero_poloidal_mode,
        largest_non_zero_toroidal_mode,
    ) = surface_rz_fourier.get_largest_non_zero_modes(surface=boundary)

    # Based on past VMEC experience, this is a solution that should allow good
    # convergence rates while still providing reasonable resolution
    # until we have a more data-driven suggestion.
    n_poloidal_modes = largest_non_zero_poloidal_mode + 2
    max_toroidal_mode = largest_non_zero_toroidal_mode + 2

    (
        n_poloidal_points,
        n_toroidal_points,
    ) = surface_utils.n_poloidal_toroidal_points_to_satisfy_nyquist_criterion(
        n_poloidal_modes=n_poloidal_modes,
        max_toroidal_mode=max_toroidal_mode,
    )

    return VmecSettings(
        n_poloidal_modes=n_poloidal_modes,
        max_toroidal_mode=max_toroidal_mode,
        n_poloidal_grid_points=n_poloidal_points,
        n_toroidal_grid_points=n_toroidal_points,
        # Reduce the number of flux surfaces and the force tolerance to speed up
        # convergence.
        multigrid_steps=[
            VmecMultiGridStep(
                n_flux_surfaces=25, force_tolerance=1e-17, n_max_iterations=2000
            ),
            VmecMultiGridStep(
                n_flux_surfaces=71, force_tolerance=1e-13, n_max_iterations=20000
            ),
        ],
        time_step=0.7,
    )


def vmec_settings_ultra_low_fidelity_fixed_boundary(
    boundary: surface_rz_fourier.SurfaceRZFourier,
) -> VmecSettings:
    """A very low-fidelity VMEC configuration that provides defaults for the given
    boundary by matching the boundary's Fourier resolution to VMEC++'s internal
    resolution.

    Used for very fast convergence in optimization tasks.
    """
    # The criterion we use below to set n_toroidal_grid_points breaks down for ntor == 0
    assert boundary.max_toroidal_mode != 0, "axisymmetric boundaries are not supported"

    (
        largest_non_zero_poloidal_mode,
        largest_non_zero_toroidal_mode,
    ) = surface_rz_fourier.get_largest_non_zero_modes(surface=boundary)

    # Based on past VMEC experience, this is a solution that should allow good
    # convergence rates while still providing reasonable resolution
    # until we have a more data-driven suggestion.
    n_poloidal_modes = largest_non_zero_poloidal_mode + 2
    max_toroidal_mode = largest_non_zero_toroidal_mode + 2

    (
        n_poloidal_points,
        n_toroidal_points,
    ) = surface_utils.n_poloidal_toroidal_points_to_satisfy_nyquist_criterion(
        n_poloidal_modes=n_poloidal_modes,
        max_toroidal_mode=max_toroidal_mode,
    )

    return VmecSettings(
        n_poloidal_modes=n_poloidal_modes,
        max_toroidal_mode=max_toroidal_mode,
        n_poloidal_grid_points=n_poloidal_points,
        n_toroidal_grid_points=n_toroidal_points,
        multigrid_steps=[
            VmecMultiGridStep(
                n_flux_surfaces=25, force_tolerance=1e-17, n_max_iterations=2000
            ),
            VmecMultiGridStep(
                n_flux_surfaces=51, force_tolerance=1e-9, n_max_iterations=20000
            ),
        ],
        time_step=0.9,
    )


def vmec_settings_from_boundary_resolution(
    boundary: surface_rz_fourier.SurfaceRZFourier,
) -> VmecSettings:
    """A sensible VMEC configuration that provides defaults for the given associated
    boundary by matching the boundary's Fourier resolution to VMEC++'s internal
    resolution.

    Optimizes for VMEC convergence rate over runtime and high fidelity.

    This is meant to be used to guarantee metrics are computed in a standardized manner
    or as a good default in case no better VmecSettings choice is available.
    """
    # The criterion we use below to set n_toroidal_grid_points breaks down for ntor == 0
    assert boundary.max_toroidal_mode != 0, "axisymmetric boundaries are not supported"

    (
        largest_non_zero_poloidal_mode,
        largest_non_zero_toroidal_mode,
    ) = surface_rz_fourier.get_largest_non_zero_modes(surface=boundary)

    # Based on past VMEC experience, this is a solution that should allow good
    # convergence rates while still providing reasonable resolution
    # until we have a more data-driven suggestion.
    n_poloidal_modes = largest_non_zero_poloidal_mode + 2
    max_toroidal_mode = largest_non_zero_toroidal_mode + 2

    (
        n_poloidal_points,
        n_toroidal_points,
    ) = surface_utils.n_poloidal_toroidal_points_to_satisfy_nyquist_criterion(
        n_poloidal_modes=n_poloidal_modes,
        max_toroidal_mode=max_toroidal_mode,
    )

    return VmecSettings(
        n_poloidal_modes=n_poloidal_modes,
        max_toroidal_mode=max_toroidal_mode,
        n_poloidal_grid_points=n_poloidal_points,
        n_toroidal_grid_points=n_toroidal_points,
        multigrid_steps=[
            VmecMultiGridStep(
                n_flux_surfaces=25, force_tolerance=1e-17, n_max_iterations=2000
            ),
            VmecMultiGridStep(
                n_flux_surfaces=51, force_tolerance=1e-17, n_max_iterations=2000
            ),
            VmecMultiGridStep(
                n_flux_surfaces=99, force_tolerance=1e-15, n_max_iterations=40000
            ),
        ],
        time_step=0.5,
    )

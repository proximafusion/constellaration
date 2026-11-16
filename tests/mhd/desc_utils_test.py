from pathlib import Path

import numpy as np
import pytest

from constellaration.mhd import desc_utils, vmec_utils


@pytest.fixture(scope="module")
def vmecpp_wout() -> vmec_utils.VmecppWOut:
    with Path("tests/utils/huggingface_dataset_wout.json").open("r") as f:
        return vmec_utils.VmecppWOut.model_validate_json(f.read())


@pytest.fixture(scope="module")
def initial_equilibrium(
    vmecpp_wout: vmec_utils.VmecppWOut,
) -> desc_utils.DescEquilibrium:
    return desc_utils.desc_equilibrium_from_vmecpp_wout(vmecpp_wout)


def test_desc_equilibrium_from_vmecpp_wout(
    vmecpp_wout: vmec_utils.VmecppWOut,
    initial_equilibrium: desc_utils.DescEquilibrium,
) -> None:
    assert initial_equilibrium.n_field_periods == vmecpp_wout.nfp
    assert initial_equilibrium.poloidal_resolution == vmecpp_wout.mpol - 1
    assert initial_equilibrium.toroidal_resolution == vmecpp_wout.ntor
    assert initial_equilibrium.iota_profile is None
    np.testing.assert_allclose(
        initial_equilibrium.total_toroidal_flux, vmecpp_wout.phi[-1], rtol=1e-12
    )
    # DESC samples VMEC's pressure on VMEC's radial grid, s = rho^2.
    pressure = initial_equilibrium.pressure_profile
    np.testing.assert_allclose(
        pressure.knots**2, np.linspace(0.0, 1.0, vmecpp_wout.ns), atol=1e-12
    )
    np.testing.assert_allclose(pressure.params, vmecpp_wout.presf, atol=1e-12)


def test_round_trip_to_and_from_desc(
    initial_equilibrium: desc_utils.DescEquilibrium,
) -> None:
    desc_object = desc_utils.to_desc_equilibrium(initial_equilibrium)

    round_trip = desc_utils.from_desc_equilibrium(desc_object)

    assert round_trip.model_dump_json() == initial_equilibrium.model_dump_json()


def test_round_trip_json(initial_equilibrium: desc_utils.DescEquilibrium) -> None:
    json_string = initial_equilibrium.model_dump_json()

    round_trip = desc_utils.DescEquilibrium.model_validate_json(json_string)

    assert round_trip.model_dump_json() == json_string


def test_solve_desc_equilibrium(
    initial_equilibrium: desc_utils.DescEquilibrium,
) -> None:
    # Solve at reduced resolution to keep the test fast.
    desc_object = desc_utils.to_desc_equilibrium(initial_equilibrium)
    desc_object.change_resolution(L=4, M=4, N=2)
    equilibrium = desc_utils.from_desc_equilibrium(desc_object)

    solution = desc_utils.solve_desc_equilibrium(equilibrium)

    assert solution.normalized_mean_absolute_force_error < 1e-2
    assert (
        solution.normalized_max_absolute_force_error
        >= solution.normalized_mean_absolute_force_error
    )
    # Fixed-boundary solve: the boundary, flux and profiles are constraints.
    solved = solution.equilibrium
    for field in ("r_boundary_lmn", "z_boundary_lmn"):
        np.testing.assert_allclose(
            getattr(solved, field), getattr(equilibrium, field), rtol=0.0, atol=1e-10
        )
    np.testing.assert_allclose(
        solved.total_toroidal_flux, equilibrium.total_toroidal_flux, rtol=1e-12
    )
    # Profiles are only preserved up to the round-off of DESC's constraint projection.
    np.testing.assert_allclose(
        solved.pressure_profile.params,
        equilibrium.pressure_profile.params,
        rtol=0.0,
        atol=1e-10,
    )
    assert solved.current_profile is not None
    assert equilibrium.current_profile is not None
    np.testing.assert_allclose(
        solved.current_profile.params,
        equilibrium.current_profile.params,
        rtol=0.0,
        atol=1e-10,
    )

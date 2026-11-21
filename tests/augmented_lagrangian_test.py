import jax.numpy as jnp
import numpy as np
import pytest

from constellaration.optimization import augmented_lagrangian as al


@pytest.fixture
def state():
    return al.AugmentedLagrangianState(
        x=jnp.zeros(3),
        multipliers=jnp.zeros(1),
        penalty_parameters=jnp.ones(1),
        objective=jnp.array(0.0),
        constraints=jnp.zeros(1),
        bounds=jnp.array([1.0, 0.1, 0.02]),
    )


@pytest.mark.parametrize(
    ("factor", "expected"),
    [(0.5, [0.5, 0.05, 0.05]), (0.95, [0.95, 0.095, 0.05]), (1.0, [1.0, 0.1, 0.05])],
)
def test_bounds_reduction(state, factor, expected):
    settings = al.AugmentedLagrangianSettings(
        bounds_reduction_factor=factor, bounds_min=0.05
    )
    updated = al.update_augmented_lagrangian_state(
        state.x, state.objective, state.constraints, state, settings
    )
    np.testing.assert_allclose(updated.bounds, expected)
    np.testing.assert_allclose(state.bounds, [1.0, 0.1, 0.02])


@pytest.mark.parametrize(
    ("updates", "expected"), [(1, 0.5), (2, 0.25), (5, 0.05), (8, 0.05)]
)
def test_repeated_bounds_reduction_stops_at_floor(state, updates, expected):
    settings = al.AugmentedLagrangianSettings(
        bounds_reduction_factor=0.5, bounds_min=0.05
    )
    for _ in range(updates):
        state = al.update_augmented_lagrangian_state(
            state.x, state.objective, state.constraints, state, settings
        )
    np.testing.assert_allclose(state.bounds, [expected, 0.05, 0.05])


def test_explicit_bounds_override_is_unchanged(state):
    settings = al.AugmentedLagrangianSettings(
        bounds_reduction_factor=0.5, bounds_min=0.05
    )
    override = jnp.array([0.01, 0.3, 2.0])
    updated = al.update_augmented_lagrangian_state(
        state.x, state.objective, state.constraints, state, settings, bounds=override
    )
    np.testing.assert_array_equal(updated.bounds, override)

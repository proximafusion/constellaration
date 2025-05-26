import multiprocessing
import time
from concurrent import futures

import jax.numpy as jnp
import nevergrad
import numpy as np
from nevergrad.parametrization import parameter as param

import constellaration.optimization.augmented_lagrangian as al
import constellaration.optimization.settings as settings_module
import constellaration.utils.seed_util as seed_util


def objective_constraints(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    objective = jnp.array(100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2)
    constraints = jnp.array(
        [
            -1 * x[0] * jnp.sin(x[0] * 3) + 3 * x[1] + 1.6,
            (x[0] - 1.5) ** 2 * jnp.cos(2 * x[0]) + (x[1] + 1) ** 2 - 1.5,
            (x[0] - 0.5) ** 2 * jnp.sin(2 * x[0]) + x[1] ** 2 - 2.25,
            2.25 - ((x[0] - 0.5) ** 2 * jnp.sin(2 * x[0]) + x[1] ** 2),
        ]
    )

    return (objective, constraints)


def run(x0: jnp.ndarray, settings: settings_module.AugmentedLagrangianMethodSettings):
    objective, constraints = objective_constraints(x0)

    budget = settings.oracle_settings.budget_initial

    state = al.AugmentedLagrangianState(
        x=jnp.copy(x0),
        multipliers=jnp.zeros_like(constraints),
        penalty_parameters=settings.penalty_parameters_initial
        * jnp.ones_like(constraints),
        objective=objective,
        constraints=constraints,
        bounds=jnp.array(settings.bounds_initial),
    )

    mp_context = multiprocessing.get_context("forkserver")
    for k in range(settings.maxit):
        parametrization = nevergrad.p.Array(
            init=np.array(state.x),
            lower=np.array(state.x - state.bounds),
            upper=np.array(state.x + state.bounds),
        )
        random_state = np.random.get_state()  # noqa: NPY002
        parametrization.random_state.set_state(random_state)
        oracle = nevergrad.optimizers.NGOpt(
            parametrization=parametrization,
            budget=budget,
            num_workers=settings.oracle_settings.num_workers,
        )
        oracle.suggest(np.array(state.x))

        t0 = time.time()
        with futures.ProcessPoolExecutor(
            max_workers=settings.oracle_settings.num_workers, mp_context=mp_context
        ) as executor:
            running_evaluations: list[tuple[futures.Future, param.Parameter]] = []
            rest_budget = budget

            while (rest_budget or running_evaluations) and (
                settings.oracle_settings.max_time is None
                or time.time() < t0 + settings.oracle_settings.max_time
            ):
                while len(running_evaluations) < min(
                    settings.oracle_settings.num_workers, rest_budget
                ):
                    candidate = oracle.ask()

                    future = executor.submit(
                        objective_constraints, x=jnp.array(candidate.value)
                    )
                    running_evaluations.append((future, candidate))
                    rest_budget -= 1

                # Wait on just the futures
                completed, _ = futures.wait(
                    [fut for fut, _ in running_evaluations],
                    return_when=(
                        futures.ALL_COMPLETED
                        if settings.oracle_settings.batch_mode
                        else futures.FIRST_COMPLETED
                    ),
                )

                # Find completed futures and process them
                for future, candidate in running_evaluations:
                    if future in completed:
                        objective, constraints = future.result()

                        if rest_budget % 1 == 0:
                            recommendation = oracle.provide_recommendation()
                            print(
                                f"    {al.augmented_lagrangian_function(objective, constraints, state)} {rest_budget} {recommendation.value}"  # noqa: E501
                            )

                        oracle.tell(
                            candidate,
                            al.augmented_lagrangian_function(
                                objective, constraints, state
                            ).item(),
                        )

                # Remove completed futures from running_evaluations
                running_evaluations = [
                    (fut, cand)
                    for fut, cand in running_evaluations
                    if fut not in completed
                ]

            recommendation = oracle.provide_recommendation()

        objective, constraints = objective_constraints(recommendation.value)

        print(
            f"k={k} objective={objective} feas={jnp.sum(jnp.maximum(0., constraints))} constraints={constraints} penalty_params={state.penalty_parameters} multipliers={state.multipliers}"  # noqa: E501
        )

        state = al.update_augmented_lagrangian_state(
            x=jnp.copy(recommendation.value),
            objective=objective,
            constraints=constraints,
            state=state,
            settings=settings.augmented_lagrangian_settings,
        )

        budget = int(
            jnp.minimum(
                settings.oracle_settings.budget_max,
                budget + settings.oracle_settings.budget_increment,
            )
        )


if __name__ == "__main__":
    x = jnp.array([2.0, 2.0])

    seed_util.seed_everything(123)

    settings = settings_module.AugmentedLagrangianMethodSettings(
        maxit=25,
        oracle_settings=settings_module.NevergradSettings(
            num_workers=20,
            budget_initial=100,
            budget_max=200,
            budget_increment=26,
            max_time=None,
            batch_mode=False,
        ),
        penalty_parameters_initial=1,
        bounds_initial=2,
        augmented_lagrangian_settings=al.AugmentedLagrangianSettings(),
    )
    run(x, settings)

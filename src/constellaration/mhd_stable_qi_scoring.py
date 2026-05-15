"""Scoring utilities for the multi-objective MHD-stable QI benchmark.

This module bundles the post-evaluation logic used by the contractor scoring
pipeline of the multi-objective MHD-stable QI problem: enumerating successive
Pareto levels in the ``(-lgradB, aspect_ratio)`` objective space and selecting
the top levels whose hypervolume (HIV) stays above a fraction of the L0
front's HIV. The functions accept generic 2D minimization objective arrays so
they are also reusable for other multi-objective post-analyses.
"""

from __future__ import annotations

import jaxtyping as jt
import numpy as np
from pymoo.indicators import hv


def pareto_levels(
    objectives: jt.Float[np.ndarray, "n_points n_objectives"],
) -> list[np.ndarray]:
    """Enumerate successive non-dominated Pareto levels (minimization).

    A point ``i`` dominates ``j`` if ``objectives[i] <= objectives[j]``
    componentwise and ``objectives[i] != objectives[j]``. Level 0 is the
    non-dominated front of the input; level 1 is the non-dominated front
    after removing level 0; and so on, until every point has been assigned.

    Args:
        objectives: ``(n_points, n_objectives)`` array; every column is
            interpreted as a minimization objective.

    Returns:
        List of integer arrays of original indices, one per level, starting
        with level 0.
    """
    if objectives.ndim != 2:
        raise ValueError(
            f"objectives must be 2D (n_points, n_objectives), got shape "
            f"{objectives.shape}"
        )
    n_points = objectives.shape[0]
    if n_points == 0:
        return []

    remaining = np.arange(n_points)
    levels: list[np.ndarray] = []
    while remaining.size:
        front_mask = _non_dominated_mask(objectives[remaining])
        levels.append(remaining[front_mask])
        remaining = remaining[~front_mask]
    return levels


def _non_dominated_mask(
    objectives: jt.Float[np.ndarray, "n_points n_objectives"],
) -> np.ndarray:
    """Boolean mask of non-dominated points (minimization)."""
    n_points = objectives.shape[0]
    is_non_dominated = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        if not is_non_dominated[i]:
            continue
        dominates_i = np.all(objectives <= objectives[i], axis=1) & np.any(
            objectives < objectives[i], axis=1
        )
        if np.any(dominates_i):
            is_non_dominated[i] = False
    return is_non_dominated


def hypervolume(
    objectives: jt.Float[np.ndarray, "n_points n_objectives"],
    reference_point: jt.Float[np.ndarray, " n_objectives"],
) -> float:
    """Hypervolume dominated by ``objectives`` relative to ``reference_point``.

    All objectives are minimized. Returns ``0.0`` for an empty input.
    """
    if objectives.shape[0] == 0:
        return 0.0
    indicator = hv.Hypervolume(ref_point=np.asarray(reference_point))
    output = indicator(np.asarray(objectives))
    assert output is not None
    return float(output)


def select_top_pareto_levels_by_hiv_fraction(
    objectives: jt.Float[np.ndarray, "n_points n_objectives"],
    reference_point: jt.Float[np.ndarray, " n_objectives"],
    fraction: float = 0.9,
) -> np.ndarray:
    """Indices of points from the top Pareto levels worth keeping for diversity.

    Mirrors the contractor SoW filter: enumerate Pareto levels of the
    minimization-objective array, then keep levels ``0`` through ``K - 1``
    where ``K`` is the smallest level such that ``HIV(levels[K:]) <
    fraction * HIV(level 0)``. If no level satisfies the condition, every
    level is kept. Level 0 is always included.

    Args:
        objectives: ``(n_points, n_objectives)`` minimization objective array.
        reference_point: Reference point passed to the HIV indicator.
        fraction: Inclusion threshold, default 0.9 (the value used by the
            SoW baselines).

    Returns:
        Sorted array of selected point indices (into ``objectives``).
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    levels = pareto_levels(objectives)
    if not levels:
        return np.empty(0, dtype=int)

    hiv_l0 = hypervolume(objectives[levels[0]], reference_point)
    if hiv_l0 <= 0.0:
        # L0 front does not dominate the reference point; nothing meaningful
        # to compare against, so we just return L0.
        return np.sort(levels[0])

    k_threshold: int | None = None
    for k in range(len(levels)):
        remaining_indices = np.concatenate(levels[k:])
        hiv_k = hypervolume(objectives[remaining_indices], reference_point)
        if hiv_k < fraction * hiv_l0:
            k_threshold = k
            break

    if k_threshold is None:
        selected = np.concatenate(levels)
    else:
        selected = np.concatenate(levels[:k_threshold])
    return np.sort(selected)

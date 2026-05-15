"""Scoring utilities for the multi-objective MHD-stable QI benchmark.

This module bundles the post-evaluation logic used by the contractor scoring
pipeline of the multi-objective MHD-stable QI problem: enumerating successive
Pareto levels in the ``(-lgradB, aspect_ratio)`` objective space and selecting
the top levels whose hypervolume (HIV) stays above a fraction of the L0
front's HIV. The functions accept generic 2D minimization objective arrays so
they are also reusable for other multi-objective post-analyses.
"""

from __future__ import annotations

import itertools
from typing import Literal

import jaxtyping as jt
import numpy as np
from pymoo.indicators import hv

from constellaration import forward_model, problems
from constellaration.geometry import surface_rz_fourier

TightnessLevel = Literal["tight", "medium", "loose"]

DEFAULT_AR_MIN = 6.0
DEFAULT_AR_MAX = 12.0
DEFAULT_N_BINS = 10
DEFAULT_MAX_PER_BIN = 15
DEFAULT_N_POLOIDAL_POINTS = 32
DEFAULT_N_TOROIDAL_POINTS = 32

REFERENCE_POINT_LGRADB_AR = np.array([1.0, 20.0])
"""HIV reference point in ``(-lgradB, aspect_ratio)`` minimization space."""

DEFAULT_INCLUSION_HIV_FRACTION = 0.9
"""Fraction of L0 HIV at which to stop including sub-fronts for diversity."""


def problem_for_tightness_level(
    tightness_level: TightnessLevel,
) -> problems.MHDStableQIStellarator:
    """Return the MHD-stable QI problem instance for the given tightness level."""
    if tightness_level == "tight":
        return problems.MHDStableQIStellarator()
    if tightness_level == "medium":
        return problems.MHDStableQIStellaratorMedium()
    if tightness_level == "loose":
        return problems.MHDStableQIStellaratorLoose()
    raise ValueError(
        f"tightness_level must be 'tight', 'medium' or 'loose', got "
        f"{tightness_level!r}"
    )


def _compute_metrics(
    boundaries: list[surface_rz_fourier.SurfaceRZFourier],
) -> list[forward_model.ConstellarationMetrics]:
    settings = forward_model.ConstellarationSettings.default_high_fidelity()
    out: list[forward_model.ConstellarationMetrics] = []
    for boundary in boundaries:
        metrics, _ = forward_model.forward_model(boundary=boundary, settings=settings)
        out.append(metrics)
    return out


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


def binned_diversity_score(
    boundaries: list[surface_rz_fourier.SurfaceRZFourier],
    aspect_ratios: jt.Float[np.ndarray, " n_points"],
    lgradB_values: jt.Float[np.ndarray, " n_points"],
    *,
    ar_min: float = DEFAULT_AR_MIN,
    ar_max: float = DEFAULT_AR_MAX,
    n_bins: int = DEFAULT_N_BINS,
    max_per_bin: int = DEFAULT_MAX_PER_BIN,
    n_poloidal_points: int = DEFAULT_N_POLOIDAL_POINTS,
    n_toroidal_points: int = DEFAULT_N_TOROIDAL_POINTS,
) -> float:
    """Binned geometric diversity score across the aspect-ratio range.

    Bins ``aspect_ratios`` into ``n_bins`` equal-width bins spanning
    ``[ar_min, ar_max]``. For each bin with at least two configurations, the
    top ``max_per_bin`` boundaries by ``lgradB_values`` are normalized to the
    bin center via :func:`surface_rz_fourier.scale_to_aspect_ratio`, and the
    mean pairwise symmetrized RMS normal displacement distance is computed.
    Bins with fewer than two configurations score zero. The overall score is
    the arithmetic mean across **all** ``n_bins`` bins -- empty bins penalize
    the score, rewarding coverage of the AR range.

    Args:
        boundaries: Plasma boundaries (one per submitted configuration).
        aspect_ratios: Aspect ratio of each boundary (typically from the
            forward-model metrics).
        lgradB_values: ``min_normalized_magnetic_gradient_scale_length``
            (already multiplied by ``n_field_periods``) for each boundary.
            Used to pick the top ``max_per_bin`` per bin.
        ar_min: Lower edge of the binning range.
        ar_max: Upper edge of the binning range.
        n_bins: Number of equal-width bins.
        max_per_bin: Per-bin cap on the number of boundaries used to
            compute pairwise distances.
        n_poloidal_points: Poloidal grid resolution for the distance metric.
        n_toroidal_points: Toroidal grid resolution for the distance metric.

    Returns:
        Mean diversity score across the ``n_bins`` bins.
    """
    n = len(boundaries)
    if len(aspect_ratios) != n or len(lgradB_values) != n:
        raise ValueError(
            "boundaries, aspect_ratios and lgradB_values must have the same length"
        )
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    if ar_max <= ar_min:
        raise ValueError(f"ar_max must exceed ar_min, got [{ar_min}, {ar_max}]")
    if max_per_bin < 2:
        raise ValueError(f"max_per_bin must be >= 2, got {max_per_bin}")

    bin_width = (ar_max - ar_min) / n_bins
    bin_centers = ar_min + (np.arange(n_bins) + 0.5) * bin_width

    aspect_ratios = np.asarray(aspect_ratios)
    lgradB_values = np.asarray(lgradB_values)

    bin_scores = np.zeros(n_bins)
    for bin_idx in range(n_bins):
        bin_low = ar_min + bin_idx * bin_width
        bin_high = bin_low + bin_width
        if bin_idx == n_bins - 1:
            mask = (aspect_ratios >= bin_low) & (aspect_ratios <= bin_high)
        else:
            mask = (aspect_ratios >= bin_low) & (aspect_ratios < bin_high)
        member_indices = np.flatnonzero(mask)
        if member_indices.size < 2:
            continue

        # Keep the top `max_per_bin` by lgradB (higher is better).
        sorted_idx = member_indices[np.argsort(-lgradB_values[member_indices])]
        kept = sorted_idx[:max_per_bin]

        normalized = [
            surface_rz_fourier.scale_to_aspect_ratio(
                boundaries[int(i)], target_aspect_ratio=float(bin_centers[bin_idx])
            )
            for i in kept
        ]
        bin_scores[bin_idx] = _mean_pairwise_rms_distance(
            normalized,
            n_poloidal_points=n_poloidal_points,
            n_toroidal_points=n_toroidal_points,
        )

    return float(np.mean(bin_scores))


def _mean_pairwise_rms_distance(
    boundaries: list[surface_rz_fourier.SurfaceRZFourier],
    n_poloidal_points: int,
    n_toroidal_points: int,
) -> float:
    """Mean over all unordered pairs of the symmetrized RMS normal distance."""
    distances: list[float] = []
    for a, b in itertools.combinations(boundaries, 2):
        distances.append(
            surface_rz_fourier.compute_rms_normal_displacement_distance(
                a,
                b,
                n_poloidal_points=n_poloidal_points,
                n_toroidal_points=n_toroidal_points,
            )
        )
    return float(np.mean(distances))


def score_boundaries_performance(
    boundaries: list[surface_rz_fourier.SurfaceRZFourier],
    tightness_level: TightnessLevel,
    *,
    metrics: list[forward_model.ConstellarationMetrics] | None = None,
) -> float:
    """Hypervolume (HIV) of the feasible submitted boundaries at a tightness level.

    Runs the constellaration forward model at high fidelity on every boundary
    (unless precomputed ``metrics`` are passed), filters the configurations
    feasible at ``tightness_level``, and returns the hypervolume of their
    non-dominated front in ``(-lgradB, aspect_ratio)`` space relative to the
    reference point ``(1.0, 20.0)``. Higher is better. Empty / all-infeasible
    submissions score 0.

    Args:
        boundaries: Submitted plasma boundaries.
        tightness_level: One of ``"tight"``, ``"medium"``, ``"loose"``.
        metrics: Optional pre-computed forward-model metrics aligned with
            ``boundaries``. Useful for repeated scoring without re-running
            VMEC. If omitted, the forward model is invoked.

    Returns:
        Hypervolume score (a non-negative float).
    """
    problem = problem_for_tightness_level(tightness_level)
    if metrics is None:
        metrics = _compute_metrics(boundaries)
    if len(metrics) != len(boundaries):
        raise ValueError(
            "metrics must have the same length as boundaries when provided"
        )

    feasible_metrics = [m for m in metrics if problem.is_feasible(m)]
    if not feasible_metrics:
        return 0.0
    objectives = np.array(
        [
            (
                -1.0 * m.minimum_normalized_magnetic_gradient_scale_length,
                1.0 * m.aspect_ratio,
            )
            for m in feasible_metrics
        ]
    )
    return hypervolume(objectives, REFERENCE_POINT_LGRADB_AR)


def score_boundaries_diversity(
    boundaries: list[surface_rz_fourier.SurfaceRZFourier],
    tightness_level: TightnessLevel,
    *,
    metrics: list[forward_model.ConstellarationMetrics] | None = None,
    inclusion_hiv_fraction: float = DEFAULT_INCLUSION_HIV_FRACTION,
    ar_min: float = DEFAULT_AR_MIN,
    ar_max: float = DEFAULT_AR_MAX,
    n_bins: int = DEFAULT_N_BINS,
    max_per_bin: int = DEFAULT_MAX_PER_BIN,
    n_poloidal_points: int = DEFAULT_N_POLOIDAL_POINTS,
    n_toroidal_points: int = DEFAULT_N_TOROIDAL_POINTS,
) -> float:
    """Binned geometric diversity score for boundaries near the Pareto front.

    Pipeline:
      1. Evaluate the forward model on each boundary at high fidelity
         (unless ``metrics`` are passed).
      2. Keep only configurations feasible at ``tightness_level``.
      3. Compute successive Pareto levels in
         ``(-lgradB, aspect_ratio)``-minimization space and keep the top
         levels whose cumulative sub-front HIV stays at or above
         ``inclusion_hiv_fraction * HIV(L0)`` (default ``0.9 * HIV(L0)``).
      4. Score those near-optimal boundaries via
         :func:`binned_diversity_score`: 10 AR bins from 6 to 12, top-15 by
         lgradB per bin, mean pairwise symmetrized RMS normal-displacement
         distance averaged across all bins (empty bins score 0).

    Args:
        boundaries: Submitted plasma boundaries.
        tightness_level: One of ``"tight"``, ``"medium"``, ``"loose"``.
        metrics: Optional pre-computed forward-model metrics aligned with
            ``boundaries``.
        inclusion_hiv_fraction: HIV-fraction-of-L0 cutoff for Pareto-level
            inclusion. SoW default 0.9.
        ar_min, ar_max, n_bins, max_per_bin, n_poloidal_points,
            n_toroidal_points: Forwarded to :func:`binned_diversity_score`.

    Returns:
        Diversity score (a non-negative float).
    """
    problem = problem_for_tightness_level(tightness_level)
    if metrics is None:
        metrics = _compute_metrics(boundaries)
    if len(metrics) != len(boundaries):
        raise ValueError(
            "metrics must have the same length as boundaries when provided"
        )

    feasible: list[tuple[surface_rz_fourier.SurfaceRZFourier, float, float]] = []
    for boundary, m in zip(boundaries, metrics):
        if problem.is_feasible(m):
            feasible.append(
                (
                    boundary,
                    m.minimum_normalized_magnetic_gradient_scale_length,
                    m.aspect_ratio,
                )
            )
    if len(feasible) < 2:
        return 0.0

    feasible_boundaries = [item[0] for item in feasible]
    lgradB_arr = np.array([item[1] for item in feasible])
    ar_arr = np.array([item[2] for item in feasible])
    objectives = np.column_stack([-lgradB_arr, ar_arr])

    selected = select_top_pareto_levels_by_hiv_fraction(
        objectives,
        reference_point=REFERENCE_POINT_LGRADB_AR,
        fraction=inclusion_hiv_fraction,
    )
    if selected.size < 2:
        return 0.0

    selected_boundaries = [feasible_boundaries[int(i)] for i in selected]
    return binned_diversity_score(
        boundaries=selected_boundaries,
        aspect_ratios=ar_arr[selected],
        lgradB_values=lgradB_arr[selected],
        ar_min=ar_min,
        ar_max=ar_max,
        n_bins=n_bins,
        max_per_bin=max_per_bin,
        n_poloidal_points=n_poloidal_points,
        n_toroidal_points=n_toroidal_points,
    )

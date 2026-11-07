import abc

import jaxtyping as jt
import numpy as np
import pydantic
from pymoo.indicators import hv

from constellaration import forward_model
from constellaration.geometry import surface_rz_fourier
from constellaration.mhd import ideal_mhd_parameters_utils, pressure_profile

_DEFAULT_RELATIVE_TOLERANCE = 1e-2


class _Problem(abc.ABC):
    _does_it_require_qi: bool

    def is_feasible(self, metrics: forward_model.ConstellarationMetrics) -> bool:
        """Checks if the design is feasible based on the constraints."""
        normalized_constraint_violations = self._normalized_constraint_violations(
            metrics
        )
        return bool(
            np.all(normalized_constraint_violations <= _DEFAULT_RELATIVE_TOLERANCE)
        )

    def compute_feasibility(
        self, metrics: forward_model.ConstellarationMetrics
    ) -> float:
        """Computes the feasibility of the design."""
        normalized_constraint_violations = self._normalized_constraint_violations(
            metrics
        )
        return float(np.max(np.maximum(normalized_constraint_violations, 0.0)))

    @abc.abstractmethod
    def _normalized_constraint_violations(
        self, metrics: forward_model.ConstellarationMetrics
    ) -> np.ndarray:
        pass


class EvaluationSingleObjective(pydantic.BaseModel):
    objective: float
    """The objective value of the solution."""
    minimize_objective: bool
    """Whether the objective is to be minimized (True) or maximized (False)."""
    feasibility: float
    """The infinity norm of the normalized constraint violations."""
    score: float
    """The score of the solution (0: bad, 1: good)."""


class SingleObjectiveProblem(_Problem):
    def evaluate(
        self, boundary: surface_rz_fourier.SurfaceRZFourier
    ) -> EvaluationSingleObjective:
        """Evaluate a single boundary and return objective, feasibility,
        and normalized score.
        """
        if self._does_it_require_qi:
            settings = forward_model.ConstellarationSettings.default_high_fidelity()
        else:
            settings = (
                forward_model.ConstellarationSettings.default_high_fidelity_skip_qi()
            )
        metrics, _ = forward_model.forward_model(boundary, settings=settings)
        score = 0.0
        if self.is_feasible(metrics):
            score = self._score(metrics)
        objective, minimize_objective = self.get_objective(metrics)
        return EvaluationSingleObjective(
            objective=objective,
            minimize_objective=minimize_objective,
            feasibility=self.compute_feasibility(metrics),
            score=score,
        )

    @abc.abstractmethod
    def get_objective(
        self, metrics: forward_model.ConstellarationMetrics
    ) -> tuple[float, bool]:
        """Returns objective value and whether the objective should be minimized (True)
        or maximized (False)."""
        pass

    @abc.abstractmethod
    def _score(self, metrics: forward_model.ConstellarationMetrics) -> float:
        pass


class EvaluationMultiObjective(pydantic.BaseModel):
    objectives: list[list[tuple[float, bool]]]
    """A list of objectives for each solution, where each objective is a tuple
    of (objective_value, minimize_objective)."""
    feasibility: list[float]
    """A list of infinity norms of the normalized constraint violations for each
    solution."""
    score: float
    """The hypervolume of the Pareto front of the solutions."""


class MultiObjectiveProblem(_Problem):
    @abc.abstractmethod
    def evaluate(
        self, boundaries: list[surface_rz_fourier.SurfaceRZFourier]
    ) -> EvaluationMultiObjective:
        """Evaluate a list of boundaries and return a score for the set
        (e.g., hypervolume).
        """
        pass


class GeometricalProblem(SingleObjectiveProblem, pydantic.BaseModel):
    """A simple geometrical problem.

    Feasibility constraints:
        1. Aspect ratio <= aspect_ratio_upper_bound. Aspect ratio compares the boundary's
            “width” to its “height.”
        2. Average triangularity <= average_triangularity_upper_bound. Triangularity
            measures how “D-shaped” the boundary is; negative values mean more indentation.
        3. Edge rotational transform >= edge_rotational_transform_over_n_field_periods_lower_bound.
            Rotational transform describes the winding of magnetic field lines around
            the plasma edge per field period.

    Scoring:
        * Computes a normalized score based on the maximum elongation of the boundary.
        It is scaled from 1 (circular) to 10 (very elongated). Higher scores reward
        shapes closer to circular flux surfaces.

    Attributes:
        aspect_ratio_upper_bound: Max allowed aspect ratio.
        average_triangularity_upper_bound: Max allowed average triangularity.
        edge_rotational_transform_over_n_field_periods_lower_bound: Minimum
            required edge rotational transform per field period.
    """  # noqa: E501

    _aspect_ratio_upper_bound: pydantic.PositiveFloat = 4.0

    _average_triangularity_upper_bound: float = -0.5

    _edge_rotational_transform_over_n_field_periods_lower_bound: (
        pydantic.PositiveFloat
    ) = 0.3

    _does_it_require_qi: bool = False

    def get_objective(
        self, metrics: forward_model.ConstellarationMetrics
    ) -> tuple[float, bool]:
        return (metrics.max_elongation, True)

    def _score(self, metrics: forward_model.ConstellarationMetrics) -> float:
        return 1.0 - _normalize_between_bounds(
            value=metrics.max_elongation,
            lower_bound=1.0,
            upper_bound=10.0,
        )

    def _normalized_constraint_violations(
        self, metrics: forward_model.ConstellarationMetrics
    ) -> np.ndarray:
        constraint_targets = np.array(
            [
                self._aspect_ratio_upper_bound,
                self._average_triangularity_upper_bound,
                self._edge_rotational_transform_over_n_field_periods_lower_bound,
            ]
        )
        constraint_violations = np.array(
            [
                metrics.aspect_ratio - self._aspect_ratio_upper_bound,
                metrics.average_triangularity - self._average_triangularity_upper_bound,
                self._edge_rotational_transform_over_n_field_periods_lower_bound
                - np.abs(metrics.edge_rotational_transform_over_n_field_periods),
            ]
        )
        return constraint_violations / np.abs(constraint_targets)


class SimpleToBuildQIStellarator(SingleObjectiveProblem, pydantic.BaseModel):
    """A problem to evaluate stellarator designs for ease of construction and deviation
    from a quasi-isodynamic (QI) field.

    Feasibility constraints:
        1. Aspect ratio <= aspect_ratio_upper_bound. Aspect ratio compares the
            boundary's “width” to its “height.”
        2. Edge rotational transform >= edge_rotational_transform_over_n_field_periods_lower_bound.
            Rotational transform describes the winding of magnetic field lines around
            the plasma edge per field period.
        3. log10(qi residual) <= log10_qi_upper_bound. QI ensures good properties in
            terms of confinement of fusion-born energetic particles, neoclassical
            transport,
            and reduction of bootstrap current.
        4. Edge magnetic mirror ratio <= edge_magnetic_mirror_ratio_upper_bound.
            Magnetic mirror ratio controls the variation in field strength at the
            plasma boundary.
        5. Max elongation <= max_elongation_upper_bound. Elongation measures the
            vertical stretching of the plasma boundary.

    Scoring:
        * Computes a normalized score based on the minimum normalized magnetic gradient
          scale length at the edge. It is scaled linearly from 0 (poor) to 1 (optimal).
          Higher scores reward fields that are easier to realize with coils.

    Attributes:
        aspect_ratio_upper_bound: Max allowed aspect ratio.
        edge_rotational_transform_over_n_field_periods_lower_bound: Minimum
            required edge rotational transform per field period.
        log10_qi_upper_bound: Max allowed log10 of the QI residual.
        edge_magnetic_mirror_ratio_upper_bound: Max allowed edge magnetic mirror ratio.
        max_elongation_upper_bound: Max allowed elongation.
    """  # noqa: E501

    _aspect_ratio_upper_bound: pydantic.PositiveFloat = 10.0

    _edge_rotational_transform_over_n_field_periods_lower_bound: (
        pydantic.PositiveFloat
    ) = 0.25

    _log10_qi_upper_bound: pydantic.NegativeFloat = -4.0

    _edge_magnetic_mirror_ratio_upper_bound: pydantic.PositiveFloat = 0.2

    _max_elongation_upper_bound: pydantic.PositiveFloat = 5.0

    _does_it_require_qi: bool = True

    def get_objective(
        self, metrics: forward_model.ConstellarationMetrics
    ) -> tuple[float, bool]:
        return (metrics.minimum_normalized_magnetic_gradient_scale_length, False)

    def _score(self, metrics: forward_model.ConstellarationMetrics) -> float:
        return _normalize_between_bounds(
            value=metrics.minimum_normalized_magnetic_gradient_scale_length,
            lower_bound=0.0,
            upper_bound=20.0,
        )

    def _normalized_constraint_violations(
        self, metrics: forward_model.ConstellarationMetrics
    ) -> np.ndarray:
        assert metrics.qi is not None
        constraint_violations = np.array(
            [
                metrics.aspect_ratio - self._aspect_ratio_upper_bound,
                self._edge_rotational_transform_over_n_field_periods_lower_bound
                - np.abs(metrics.edge_rotational_transform_over_n_field_periods),
                np.log10(metrics.qi) - self._log10_qi_upper_bound,
                metrics.edge_magnetic_mirror_ratio
                - self._edge_magnetic_mirror_ratio_upper_bound,
                metrics.max_elongation - self._max_elongation_upper_bound,
            ]
        )
        constraint_targets = np.array(
            [
                self._aspect_ratio_upper_bound,
                self._edge_rotational_transform_over_n_field_periods_lower_bound,
                self._log10_qi_upper_bound,
                self._edge_magnetic_mirror_ratio_upper_bound,
                self._max_elongation_upper_bound,
            ]
        )
        return constraint_violations / np.abs(constraint_targets)


class Design(pydantic.BaseModel):
    """Boundary and optional pressure profile for equilibrium optimization.

    With pressure=None, equilibria use the fixed profile p(s) = 1 Pa * (1 - s),
    where s is normalized toroidal flux, zero toroidal current, and edge toroidal
    flux pi * a**2 * 1 T, with a the boundary minor radius. This is a near-vacuum
    approximation, not exactly zero pressure. Finite-beta problems require an
    explicit pressure profile.
    """

    boundary: surface_rz_fourier.SurfaceRZFourier
    pressure: pressure_profile.IntegratedBernsteinProfile | None = None


class _FiniteBetaQIStellarator(SimpleToBuildQIStellarator):
    """Shared QI and sampled-stability constraints for joint candidates."""

    _max_elongation_upper_bound: pydantic.PositiveFloat = 7.0
    _magnetic_mirror_ratio_well_upper_bound: pydantic.PositiveFloat = 0.05
    _edge_rotational_transform_over_n_field_periods_lower_bound: (
        pydantic.PositiveFloat
    ) = 0.225

    def _common_constraint_violations(
        self, metrics: forward_model.ConstellarationMetrics
    ) -> tuple[np.ndarray, float, float] | None:
        """Return inherited QI constraints and the two stability violations."""
        required = (
            metrics.qi,
            metrics.minimum_normalized_mercier_criterion,
            metrics.magnetic_mirror_ratio_well,
            metrics.volume_averaged_beta,
        )
        if any(value is None or not np.isfinite(value) for value in required):
            return None
        if metrics.qi < 0:
            return None

        # log10(0) represents perfect QI, but keep optimizer outputs finite.
        legacy_metrics = metrics.model_copy(update={"qi": max(metrics.qi, 1e-300)})
        legacy = super()._normalized_constraint_violations(legacy_metrics)
        mercier = -metrics.minimum_normalized_mercier_criterion / 1e-3
        mirror_well = (
            metrics.magnetic_mirror_ratio_well
            - self._magnetic_mirror_ratio_well_upper_bound
        ) / self._magnetic_mirror_ratio_well_upper_bound
        return legacy, mercier, mirror_well

    def evaluate(self, design: Design) -> EvaluationSingleObjective:
        if design.pressure is None:
            raise ValueError("finite-beta problems require a design with pressure")
        metrics, _ = forward_model.forward_model(
            design.boundary,
            ideal_mhd_parameters=ideal_mhd_parameters_utils.design_to_ideal_mhd_parameters(
                design
            ),
            settings=forward_model.ConstellarationSettings.default_high_fidelity(),
        )
        objective, minimize = self.get_objective(metrics)
        return EvaluationSingleObjective(
            objective=objective,
            minimize_objective=minimize,
            feasibility=self.compute_feasibility(metrics),
            score=self._score(metrics) if self.is_feasible(metrics) else 0.0,
        )


class MaximumBetaQIStellarator(_FiniteBetaQIStellarator):
    """A problem to maximize volume-averaged beta for a QI stellarator.

    Feasibility constraints:
        1. Aspect ratio < 12.
        2. Absolute edge rotational transform per field period >= 0.225.
        3. QI residual <= 1e-4.
        4. Edge magnetic mirror ratio <= 0.2.
        5. Maximum elongation <= 7.
        6. Conservative soft minimum of the normalized Mercier criterion > 0.
        7. Magnetic mirror-ratio well < 0.05.

    There is no minimum beta or beta-peaking constraint.
    All bounds are enforced without a relative feasibility tolerance. Mercier
    is evaluated on interior VMEC radial samples; mirror-ratio well is a max-J
    proxy, not proof of max-J. Toroidal current is zero, and edge flux is
    estimated for an approximately 1 T axis field.

    Scoring:
        * Maximizes VMEC's volume-averaged beta, expressed as a fraction.
          The score is beta clipped to [0, 1] for feasible candidates and zero
          for infeasible candidates.

    Attributes:
        aspect_ratio_upper_bound: Strict upper bound on aspect ratio.
        edge_rotational_transform_over_n_field_periods_lower_bound: Minimum
            absolute edge rotational transform per field period.
        log10_qi_upper_bound: Maximum log10 of the QI residual.
        edge_magnetic_mirror_ratio_upper_bound: Maximum edge mirror ratio.
        max_elongation_upper_bound: Maximum elongation.
        magnetic_mirror_ratio_well_upper_bound: Strict upper bound on mirror well.
    """

    _aspect_ratio_upper_bound: pydantic.PositiveFloat = 12.0

    @property
    def n_constraints(self) -> int:
        return 7

    def _normalized_constraint_violations(
        self, metrics: forward_model.ConstellarationMetrics
    ) -> np.ndarray:
        common = self._common_constraint_violations(metrics)
        if common is None:
            return np.full(self.n_constraints, np.inf)
        legacy, mercier, mirror_well = common
        violations = np.concatenate((legacy, [mercier, mirror_well]))
        return np.where(np.isfinite(violations), violations, np.inf)

    def is_feasible(self, metrics: forward_model.ConstellarationMetrics) -> bool:
        violations = self._normalized_constraint_violations(metrics)
        aspect, _, _, _, _, mercier, mirror_well = violations
        return bool(
            np.all(violations <= 0)
            and aspect < 0
            and mercier < 0
            and mirror_well < 0
        )

    def get_objective(
        self, metrics: forward_model.ConstellarationMetrics
    ) -> tuple[float, bool]:
        beta = metrics.volume_averaged_beta
        return (float(beta) if beta is not None else float("nan"), False)

    def _score(self, metrics: forward_model.ConstellarationMetrics) -> float:
        assert metrics.volume_averaged_beta is not None
        return _normalize_between_bounds(
            value=metrics.volume_averaged_beta, lower_bound=0.0, upper_bound=1.0
        )


class SimpleToBuildFiniteBetaQIStellarator(_FiniteBetaQIStellarator):
    """A problem to evaluate finite-beta QI stellarators for ease of construction.

    Feasibility constraints:
        1. Aspect ratio <= 5.
        2. Absolute edge rotational transform per field period >= 0.225.
        3. QI residual <= 1e-4.
        4. Edge magnetic mirror ratio <= 0.2.
        5. Maximum elongation <= 7.
        6. Conservative soft minimum of the normalized Mercier criterion > 0.
        7. Magnetic mirror-ratio well < 0.05.
        8. VMEC volume-averaged beta >= 0.03.
        9. Beta peaking in [2.9, 3.1], inclusive. Beta peaking is axis beta
           divided by VMEC's volume-averaged beta.

    All bounds are enforced without a relative feasibility tolerance. Mercier
    is evaluated on interior VMEC radial samples; mirror-ratio well is a max-J
    proxy, not proof of max-J. Toroidal current is zero, and edge flux is
    estimated for an approximately 1 T axis field.

    Scoring:
        * Maximizes the minimum normalized magnetic gradient scale length at the
          edge, as in SimpleToBuildQIStellarator. The score is the scale length
          divided by 20 and clipped to [0, 1] for feasible candidates, and zero
          for infeasible candidates.

    Attributes:
        aspect_ratio_upper_bound: Maximum aspect ratio.
        edge_rotational_transform_over_n_field_periods_lower_bound: Minimum
            absolute edge rotational transform per field period.
        log10_qi_upper_bound: Maximum log10 of the QI residual.
        edge_magnetic_mirror_ratio_upper_bound: Maximum edge mirror ratio.
        max_elongation_upper_bound: Maximum elongation.
        magnetic_mirror_ratio_well_upper_bound: Strict upper bound on mirror well.
    """

    _aspect_ratio_upper_bound: pydantic.PositiveFloat = 5.0

    @property
    def n_constraints(self) -> int:
        return 10

    def _normalized_constraint_violations(
        self, metrics: forward_model.ConstellarationMetrics
    ) -> np.ndarray:
        common = self._common_constraint_violations(metrics)
        peaking = metrics.beta_peaking_factor
        if common is None or peaking is None or not np.isfinite(peaking):
            return np.full(self.n_constraints, np.inf)
        legacy, mercier, mirror_well = common
        beta = metrics.volume_averaged_beta
        assert beta is not None
        violations = np.concatenate(
            (
                legacy,
                [
                    mercier,
                    mirror_well,
                    (0.03 - beta) / 0.03,
                    (2.9 - peaking) / 3.0,
                    (peaking - 3.1) / 3.0,
                ],
            )
        )
        return np.where(np.isfinite(violations), violations, np.inf)

    def is_feasible(self, metrics: forward_model.ConstellarationMetrics) -> bool:
        violations = self._normalized_constraint_violations(metrics)
        _, _, _, _, _, mercier, mirror_well, _, _, _ = violations
        return bool(np.all(violations <= 0) and mercier < 0 and mirror_well < 0)


class MHDStableQIStellarator(MultiObjectiveProblem, pydantic.BaseModel):
    """A multi-objective problem to evaluate the trade-off between compactness and
    simple coils for ideal-MHD stable QI stellarator designs.

    Feasibility constraints:
        1. Edge rotational transform >=
            edge_rotational_transform_over_n_field_periods_lower_bound.
            The rotational transform describes the winding of magnetic field lines
            around the plasma edge per field period.
        2. log10(qi residual) <= log10_qi_upper_bound. QI ensures good properties in
            terms of confinement of fusion-born energetic particles, neoclassical
            transport, and reduction of bootstrap current.
        3. Edge magnetic mirror ratio <= edge_magnetic_mirror_ratio_upper_bound.
            Magnetic mirror ratio controls the variation in field strength at the
            plasma boundary.
        4. Flux compression in regions of bad curvature <=
            flux_compression_in_regions_of_bad_curvature_upper_bound. The flux
            compression in regions of bad curvature is a geometrical proxy for
            turbulent transport.
        5. Vacuum well >= vacuum_well_lower_bound. The vacuum well is a measure of the
            stability of the plasma against ideal-MHD instabilities.

    Scoring:
        * Computes the hypervolume of the feasible designs in the 2D space defined by
          the minimum normalized magnetic gradient scale length and aspect ratio.
          Higher scores reward sets of designs with larger hypervolume.

    Attributes:
        edge_rotational_transform_over_n_field_periods_lower_bound: Minimum
            required edge rotational transform per field period.
        log10_qi_upper_bound: Max allowed log10 of the QI residual.
        edge_magnetic_mirror_ratio_upper_bound: Max allowed edge magnetic mirror ratio.
        flux_compression_in_regions_of_bad_curvature_upper_bound: Max allowed
            flux compression in regions of bad curvature.
        vacuum_well_lower_bound: Minimum required vacuum well.
    """

    _edge_rotational_transform_over_n_field_periods_lower_bound: (
        pydantic.PositiveFloat
    ) = 0.25

    _log10_qi_upper_bound: pydantic.NegativeFloat = -3.5

    _edge_magnetic_mirror_ratio_upper_bound: pydantic.PositiveFloat = 0.25

    _flux_compression_in_regions_of_bad_curvature_upper_bound: (
        pydantic.PositiveFloat
    ) = 0.9

    _vacuum_well_lower_bound: pydantic.NonNegativeFloat = 0.0

    _does_it_require_qi: bool = True

    def evaluate(
        self, boundaries: list[surface_rz_fourier.SurfaceRZFourier]
    ) -> EvaluationMultiObjective:
        metrics: list[forward_model.ConstellarationMetrics] = []
        for boundary in boundaries:
            setting = forward_model.ConstellarationSettings.default_high_fidelity()
            m, _ = forward_model.forward_model(
                boundary=boundary,
                settings=setting,
            )
            metrics.append(m)
        objectives = [self.get_objectives(m) for m in metrics]
        feasibility = [self.compute_feasibility(m) for m in metrics]
        return EvaluationMultiObjective(
            objectives=objectives,
            feasibility=feasibility,
            score=self._score(metrics),
        )

    def get_objectives(
        self, metrics: forward_model.ConstellarationMetrics
    ) -> list[tuple[float, bool]]:
        """Returns a list of (objective_value, minimize) tuples.
        Each tuple contains:
        - The objective value
        - A boolean indicating whether the objective should be minimized (True) or
        maximized (False)
        Returns:
            List of tuples with:
            - minimum normalized magnetic gradient scale length (maximize)
            - aspect ratio (minimize)
        """
        return [
            (metrics.minimum_normalized_magnetic_gradient_scale_length, False),
            (metrics.aspect_ratio, True),
        ]

    def _score(self, metrics: list[forward_model.ConstellarationMetrics]) -> float:
        feasible_metrics = [m for m in metrics if self.is_feasible(m)]
        if not feasible_metrics:
            return 0.0
        X = np.array(
            [
                (
                    -1.0 * m.minimum_normalized_magnetic_gradient_scale_length,
                    1.0 * m.aspect_ratio,
                )
                for m in feasible_metrics
            ]
        )
        reference_point = np.array([1.0, 20.0])
        return _hypervolume(
            X=X,
            reference_point=reference_point,
        )

    def _normalized_constraint_violations(
        self, metrics: forward_model.ConstellarationMetrics
    ) -> np.ndarray:
        assert metrics.qi is not None
        assert metrics.flux_compression_in_regions_of_bad_curvature is not None
        constraint_violations = np.array(
            [
                self._edge_rotational_transform_over_n_field_periods_lower_bound
                - np.abs(metrics.edge_rotational_transform_over_n_field_periods),
                np.log10(metrics.qi) - self._log10_qi_upper_bound,
                metrics.edge_magnetic_mirror_ratio
                - self._edge_magnetic_mirror_ratio_upper_bound,
                metrics.flux_compression_in_regions_of_bad_curvature
                - self._flux_compression_in_regions_of_bad_curvature_upper_bound,
                self._vacuum_well_lower_bound - metrics.vacuum_well,
            ]
        )
        constraint_targets = np.array(
            [
                self._edge_rotational_transform_over_n_field_periods_lower_bound,
                self._log10_qi_upper_bound,
                self._edge_magnetic_mirror_ratio_upper_bound,
                self._flux_compression_in_regions_of_bad_curvature_upper_bound,
                np.maximum(1e-1, self._vacuum_well_lower_bound),
            ]
        )
        return constraint_violations / np.abs(constraint_targets)


class MHDStableQIStellaratorMedium(MHDStableQIStellarator):
    """Medium-tightness variant of :class:`MHDStableQIStellarator`.

    Constraint bounds (1% tolerance applied as in the parent class):
        * Edge rotational transform / NFP >= 0.20
        * log10(QI residuals) <= -3.5
        * Edge magnetic mirror ratio <= 0.25
        * Flux compression in regions of bad curvature <= 0.9
        * Vacuum well >= -0.025
    """

    _edge_rotational_transform_over_n_field_periods_lower_bound: float = 0.20

    _log10_qi_upper_bound: float = -3.5

    _edge_magnetic_mirror_ratio_upper_bound: float = 0.25

    _flux_compression_in_regions_of_bad_curvature_upper_bound: float = 0.9

    _vacuum_well_lower_bound: float = -0.025


class MHDStableQIStellaratorLoose(MHDStableQIStellarator):
    """Loose-tightness variant of :class:`MHDStableQIStellarator`.

    Constraint bounds (1% tolerance applied as in the parent class):
        * Edge rotational transform / NFP >= 0.15
        * log10(QI residuals) <= -3.0
        * Edge magnetic mirror ratio <= 0.30
        * Flux compression in regions of bad curvature <= 1.1
        * Vacuum well >= -0.05
    """

    _edge_rotational_transform_over_n_field_periods_lower_bound: float = 0.15

    _log10_qi_upper_bound: float = -3.0

    _edge_magnetic_mirror_ratio_upper_bound: float = 0.30

    _flux_compression_in_regions_of_bad_curvature_upper_bound: float = 1.1

    _vacuum_well_lower_bound: float = -0.05


def _hypervolume(
    X: jt.Float[np.ndarray, "n_points n_metrics"],
    reference_point: jt.Float[np.ndarray, " n_metrics"],
) -> float:
    """Computes the hypervolume of X with respect to the reference point."""
    indicator = hv.Hypervolume(ref_point=reference_point)
    output = indicator(X)
    assert output is not None
    return output


def _normalize_between_bounds(
    value: float,
    lower_bound: float,
    upper_bound: float,
) -> float:
    """Normalizes a value between 0 and 1 based on the given bounds."""
    assert lower_bound < upper_bound
    normalized_value = (value - lower_bound) / (upper_bound - lower_bound)
    return np.clip(normalized_value, 0.0, 1.0)

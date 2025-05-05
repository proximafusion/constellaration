import abc

import jaxtyping as jt
import numpy as np
import pydantic
from pymoo.indicators import hv
from constellaration import forward_model
from constellaration.geometry import surface_rz_fourier


class SingleObjectiveProblem(abc.ABC):
    def score(self, boundary: surface_rz_fourier.SurfaceRZFourier) -> float:
        settings = forward_model.ConstellarationSettings.default_high_fidelity()
        metrics, _ = forward_model.forward_model(boundary, settings=settings)
        if not self.is_feasible(metrics):
            return float("nan")
        return self._score(metrics)

    @abc.abstractmethod
    def is_feasible(self, metrics: forward_model.ConstellarationMetrics) -> bool:
        pass

    @abc.abstractmethod
    def _score(self, metrics: forward_model.ConstellarationMetrics) -> float:
        pass


class MultiObjectiveProblem(abc.ABC):
    @abc.abstractmethod
    def score(self, boundaries: list[surface_rz_fourier.SurfaceRZFourier]) -> float:
        pass


class GeometricalProblem(SingleObjectiveProblem, pydantic.BaseModel):
    _aspect_ratio_upper_bound: pydantic.PositiveFloat = 5.0

    _average_triangularity_upper_bound: float = -0.5

    _edge_rotational_transform_over_n_field_periods_lower_bound: (
        pydantic.PositiveFloat
    ) = 0.2

    def _score(self, metrics: forward_model.ConstellarationMetrics) -> float:
        return 1.0 - _normalize_between_bounds(
            value=metrics.max_elongation,
            lower_bound=1.0,
            upper_bound=10.0,
        )

    def is_feasible(self, metrics: forward_model.ConstellarationMetrics) -> bool:
        if (
            _is_constraint_violated(
                value=metrics.aspect_ratio,
                upper_bound=self._aspect_ratio_upper_bound,
            )
            or _is_constraint_violated(
                value=metrics.average_triangularity,
                upper_bound=self._average_triangularity_upper_bound,
            )
            or _is_constraint_violated(
                value=metrics.edge_rotational_transform_over_n_field_periods,
                lower_bound=self._edge_rotational_transform_over_n_field_periods_lower_bound,
            )
        ):
            return False
        return True


class SimpleToBuildQIStellarator(SingleObjectiveProblem, pydantic.BaseModel):
    _aspect_ratio_upper_bound: pydantic.PositiveFloat = 10.0

    _edge_rotational_transform_over_n_field_periods_lower_bound: (
        pydantic.PositiveFloat
    ) = 0.25

    _log10_qi_upper_bound: pydantic.NegativeFloat = -4.0

    _edge_magnetic_mirror_ratio_upper_bound: pydantic.PositiveFloat = 0.2

    _max_elongation_upper_bound: pydantic.PositiveFloat = 5.0

    def _score(self, metrics: forward_model.ConstellarationMetrics) -> float:
        return _normalize_between_bounds(
            value=metrics.minimum_normalized_magnetic_gradient_scale_length,
            lower_bound=0.0,
            upper_bound=10.0,
        )

    def is_feasible(self, metrics: forward_model.ConstellarationMetrics) -> bool:
        if metrics.qi is None:
            return False
        if (
            _is_constraint_violated(
                value=metrics.aspect_ratio,
                upper_bound=self._aspect_ratio_upper_bound,
            )
            or _is_constraint_violated(
                value=metrics.edge_rotational_transform_over_n_field_periods,
                lower_bound=self._edge_rotational_transform_over_n_field_periods_lower_bound,
            )
            or _is_constraint_violated(
                value=np.log10(metrics.qi),
                upper_bound=self._log10_qi_upper_bound,
            )
            or _is_constraint_violated(
                value=metrics.edge_magnetic_mirror_ratio,
                upper_bound=self._edge_magnetic_mirror_ratio_upper_bound,
            )
            or _is_constraint_violated(
                value=metrics.max_elongation,
                upper_bound=self._max_elongation_upper_bound,
            )
        ):
            return False
        return True


class MHDStableQIStellarator(MultiObjectiveProblem, pydantic.BaseModel):
    _aspect_ratio_over_edge_rotational_transform_upper_bound: pydantic.PositiveFloat = (
        10.0
    )

    _edge_rotational_transform_over_n_field_periods_lower_bound: (
        pydantic.PositiveFloat
    ) = 0.25

    _log10_qi_upper_bound: pydantic.NegativeFloat = -3.5

    _edge_magnetic_mirror_ratio_upper_bound: pydantic.PositiveFloat = 0.25

    _flux_compression_in_regions_of_bad_curvature_upper_bound: (
        pydantic.PositiveFloat
    ) = 0.9

    _vacuum_well_lower_bound: pydantic.NonNegativeFloat = 0.0

    def score(self, boundaries: list[surface_rz_fourier.SurfaceRZFourier]) -> float:
        feasible_metrics: list[forward_model.ConstellarationMetrics] = []
        for boundary in boundaries:
            setting = forward_model.ConstellarationSettings.default_high_fidelity()
            metrics, _ = forward_model.forward_model(
                boundary=boundary,
                settings=setting,
            )
            if self.is_feasible(metrics=metrics):
                feasible_metrics.append(metrics)
        X = np.array(
            [
                (
                    -1.0 * m.minimum_normalized_magnetic_gradient_scale_length,
                    1.0 * m.aspect_ratio,
                )
                for m in feasible_metrics
            ]
        )
        reference_point = np.array([-5.0, 1.0])
        return _hypervolume(
            X=X,
            reference_point=reference_point,
        )

    def is_feasible(self, metrics: forward_model.ConstellarationMetrics) -> bool:
        assert metrics.qi is not None
        assert metrics.flux_compression_in_regions_of_bad_curvature is not None
        if (
            _is_constraint_violated(
                value=metrics.aspect_ratio_over_edge_rotational_transform,
                upper_bound=self._aspect_ratio_over_edge_rotational_transform_upper_bound,
            )
            or _is_constraint_violated(
                value=metrics.edge_rotational_transform_over_n_field_periods,
                lower_bound=self._edge_rotational_transform_over_n_field_periods_lower_bound,
            )
            or _is_constraint_violated(
                value=np.log10(metrics.qi),
                upper_bound=self._log10_qi_upper_bound,
            )
            or _is_constraint_violated(
                value=metrics.edge_magnetic_mirror_ratio,
                upper_bound=self._edge_magnetic_mirror_ratio_upper_bound,
            )
            or _is_constraint_violated(
                value=metrics.flux_compression_in_regions_of_bad_curvature,
                upper_bound=self._flux_compression_in_regions_of_bad_curvature_upper_bound,
            )
            or _is_constraint_violated(
                value=metrics.vacuum_well,
                lower_bound=self._vacuum_well_lower_bound,
            )
        ):
            return False
        return True


def _hypervolume(
    X: jt.Float[np.ndarray, "n_points n_metrics"],
    reference_point: jt.Float[np.ndarray, " n_metrics"],
) -> float:
    indicator = hv.Hypervolume(ref_point=reference_point)
    output = indicator(X)
    assert output is not None
    return output


def _normalize_between_bounds(
    value: float,
    lower_bound: float,
    upper_bound: float,
) -> float:
    assert lower_bound < upper_bound
    normalized_value = (value - lower_bound) / (upper_bound - lower_bound)
    return np.clip(normalized_value, 0.0, 1.0)


def _is_constraint_violated(
    value: float,
    lower_bound: float = -np.inf,
    upper_bound: float = np.inf,
    relative_epsilon: float = 1e-2,
) -> bool:
    """Check if a value violates the constraint defined by the lower and upper bounds.

    Args:
        value: The value to check.
        lower_bound: The lower bound of the constraint. Defaults to -np.inf.
        upper_bound: The upper bound of the constraint. Defaults to np.inf.
        relative_epsilon: A small value to avoid floating-point precision issues.
            Defaults to 1e-2.

    Returns:
        True if the value violates the constraint, False otherwise.
    """
    soft_lower_bound = (
        -relative_epsilon if lower_bound == 0 else lower_bound * (1 - relative_epsilon)
    )
    soft_upper_bound = (
        relative_epsilon if upper_bound == 0 else upper_bound * (1 + relative_epsilon)
    )
    return value < soft_lower_bound or value > soft_upper_bound

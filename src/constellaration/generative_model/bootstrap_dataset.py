import datasets
import matplotlib.pyplot as plt
import numpy as np
import orjson
import pandas as pd
from matplotlib import axes
from sklearn import calibration, ensemble, mixture, model_selection

from constellaration import problems
from constellaration.geometry import surface_rz_fourier, surface_utils

DATASET_MAX_TOROIDAL_MODE = 4
MAX_POLOIDAL_MODE = 4
MAX_TOROIDAL_MODE = 4
N_FIELD_PERIODS = 3
SEED = 24

ANY_PROBLEM = problems.GeometricalProblem | problems.SimpleToBuildQIStellarator


def load_source_datasets_with_no_errors() -> pd.DataFrame:
    """Load and concatenate source datasets from HugginFace dataset.

    Includes all ids, unfolded targets and unfolded metrics but not the contents of the
    mhd  equilibria.
    """
    dframe = datasets.load_dataset("proxima-fusion/constellaration", "default")[  # type: ignore
        "train"
    ].to_pandas()  # type: ignore

    assert isinstance(dframe, pd.DataFrame)
    errors_dframe = dframe[
        [
            "misc.has_optimize_boundary_omnigenity_vmec_error",
            "misc.has_optimize_boundary_omnigenity_desc_error",
            "misc.has_generate_qp_initialization_from_targets_error",
            "misc.has_generate_nae_initialization_from_targets_error",
            "misc.has_neurips_2025_forward_model_error",
        ]
    ].fillna(False)

    dframe = dframe[~errors_dframe.any(axis=1)]

    # Make a column with the method name for convenience
    dframe["method"] = dframe.apply(_create_method_name_column, axis=1)

    return dframe  # type: ignore


def find_feasible_candidate_in_the_data(
    X: np.ndarray,
    Y_obj: np.ndarray,
    Y_cons: np.ndarray,
) -> np.ndarray:
    feasible_indices = np.where(np.all(Y_cons <= 0, axis=1))[0]
    if len(feasible_indices) == 0:
        raise RuntimeError("No feasible candidate found.")
    print(f"Feasible candidates in the data: {len(feasible_indices)}")
    sorted_feasible_indices = feasible_indices[np.argsort(Y_obj[feasible_indices])]
    return X[sorted_feasible_indices[0]]


def _create_method_name_column(row: pd.Series) -> str:
    method_id_cols = [
        "desc_omnigenous_field_optimization_settings.id",
        "qp_init_omnigenous_field_optimization_settings.id",
        "nae_init_omnigenous_field_optimization_settings.id",
        "vmec_omnigenous_field_optimization_settings.id",
    ]
    for col in method_id_cols:
        if pd.notna(row[col]):  # type: ignore
            return col.replace("_omnigenous_field_optimization_settings.id", "")  # type: ignore
    return "unknown"


def _unflatten_metrics_and_concatenate(
    dframe: pd.DataFrame,
) -> pd.DataFrame:
    """Unflattens the metrics and concatenates them with the original DataFrame."""
    metrics_dframe = pd.DataFrame(orjson.loads(f"[{','.join(dframe['metrics.json'])}]"))
    return pd.concat(
        [
            dframe.drop(columns=["metrics.json"]).reset_index(drop=True),
            metrics_dframe,
        ],
        axis=1,
    )


def _unserialize_surface(
    dframe: pd.DataFrame,
) -> pd.DataFrame:
    """Unserializes the surface data in the DataFrame."""

    def _unserialize_surface_row(
        row: pd.Series,
    ) -> pd.Series:
        """Unserializes a single row of the DataFrame."""
        surface = surface_rz_fourier.SurfaceRZFourier.model_validate_json(
            row["boundary.json"]  # type: ignore
        )  # type: ignore
        row["boundary"] = surface
        return row

    return dframe.apply(_unserialize_surface_row, axis=1)  # type: ignore


def _augment_dataset(
    dframe: pd.DataFrame,
) -> pd.DataFrame:
    """Performs data augmentation on the dataset."""

    def _flip_z_sin_if_negative(
        row: pd.Series,
    ) -> pd.Series:
        """Flip theta sign."""
        z_sin = row["boundary"].z_sin  # type: ignore
        if z_sin[0, DATASET_MAX_TOROIDAL_MODE + 1] < 0:
            row["boundary"] = row["boundary"].model_copy(update=dict(z_sin=-1 * z_sin))  # type: ignore
        return row  # type: ignore

    def _flip_modes(
        row: pd.Series,
        toroidal_mode: int | None = None,
        poloidal_mode: int | None = None,
    ) -> pd.Series:
        """Flip toroidal mode number with abs(n) == toroidal_mode."""
        boundary = row["boundary"]
        r_cos = boundary.r_cos  # type: ignore
        z_sin = boundary.z_sin  # type: ignore
        if toroidal_mode is not None:
            mask = np.ones(2 * DATASET_MAX_TOROIDAL_MODE + 1)
            mask[DATASET_MAX_TOROIDAL_MODE - toroidal_mode] = -1
            mask[DATASET_MAX_TOROIDAL_MODE + toroidal_mode] = -1
            r_cos = r_cos * mask[None, :]
            z_sin = z_sin * mask[None, :]
        if poloidal_mode is not None:
            mask = np.ones(DATASET_MAX_TOROIDAL_MODE + 1)
            mask[poloidal_mode] = -1
            r_cos = r_cos * mask[:, None]
            z_sin = z_sin * mask[:, None]
        row["boundary"] = boundary.model_copy(update=dict(r_cos=r_cos, z_sin=z_sin))  # type: ignore
        return row

    dframe = dframe.apply(_flip_z_sin_if_negative, axis=1)  # type: ignore

    return pd.concat(
        [
            dframe,
        ],
        axis=0,
    ).reset_index(drop=True)


def _x_to_surface(
    x: np.ndarray,
    max_poloidal_mode: int,
    max_toroidal_mode: int,
    n_field_periods: int,
) -> surface_rz_fourier.SurfaceRZFourier:
    shape = (max_poloidal_mode + 1, 2 * max_toroidal_mode + 1)
    r_cos = np.zeros(shape).ravel()
    z_sin = np.zeros(shape).ravel()
    r_cos[max_toroidal_mode] = 1.0
    x_to_r_cos, x_to_z_sin = np.split(x, 2)
    r_cos[max_toroidal_mode + 1 :] = x_to_r_cos
    z_sin[max_toroidal_mode + 1 :] = x_to_z_sin
    return surface_rz_fourier.SurfaceRZFourier(
        r_cos=r_cos.reshape(shape),
        z_sin=z_sin.reshape(shape),
        n_field_periods=n_field_periods,
    )


def _to_X(
    dframe: pd.DataFrame,
    max_poloidal_mode: int,
    max_toroidal_mode: int,
) -> np.ndarray:
    Xs: list[np.ndarray] = []
    for _, row in dframe.iterrows():
        # TODO(amerlo): remove this since it is not needed anymore
        surface = surface_rz_fourier.set_max_mode_numbers(
            surface=row["boundary"],  # type: ignore
            max_poloidal_mode=max_poloidal_mode,
            max_toroidal_mode=max_toroidal_mode,
        )
        x = np.concatenate(
            [
                surface.r_cos.ravel()[surface.max_toroidal_mode + 1 :],
                surface.z_sin.ravel()[surface.max_toroidal_mode + 1 :],
            ]
        )
        Xs.append(x)
    return np.array(Xs)


def _compute_x_scale(surface: surface_rz_fourier.SurfaceRZFourier) -> np.ndarray:
    mask = surface_rz_fourier.build_surface_rz_fourier_mask(
        surface=surface,  # type: ignore
        max_poloidal_mode=surface.max_poloidal_mode,
        max_toroidal_mode=surface.max_toroidal_mode,
    )
    scale = surface_utils.energy_spectrum_scaling(
        poloidal_modes=surface.poloidal_modes,
        toroidal_modes=surface.toroidal_modes,
        energy_scale=1.5,
    )
    return np.concatenate([scale[mask.r_cos], scale[mask.z_sin]])


def _to_Y_constraints(
    dframe: pd.DataFrame,
    problem: ANY_PROBLEM,
) -> np.ndarray:
    """Extracts the constraint metrics from the DataFrame from the given problem."""
    if isinstance(problem, problems.GeometricalProblem):
        targets = np.array(
            [
                problem._average_triangularity_upper_bound,
                problem._edge_rotational_transform_over_n_field_periods_lower_bound,
                problem._aspect_ratio_upper_bound,
            ]
        )
        values = dframe[
            [
                "average_triangularity",
                "edge_rotational_transform_over_n_field_periods",
                "aspect_ratio",
            ]
        ].to_numpy()
        mask = np.array([1, -1, 1])
        return (values - targets) * mask
    elif isinstance(problem, problems.SimpleToBuildQIStellarator):
        targets = np.array(
            [
                problem._aspect_ratio_upper_bound,
                problem._edge_rotational_transform_over_n_field_periods_lower_bound,
                problem._log10_qi_upper_bound,
                problem._edge_magnetic_mirror_ratio_upper_bound,
                problem._max_elongation_upper_bound,
            ]
        )
        values = dframe[
            [
                "aspect_ratio",
                "edge_rotational_transform_over_n_field_periods",
                "qi",
                "edge_magnetic_mirror_ratio",
                "max_elongation",
            ]
        ].to_numpy()
        # applying log10 to the qi constraint
        values[:, 2] = np.log10(values[:, 2])
        mask = np.array([1, -1, 1, 1, 1])
        return (values - targets) * mask
    else:
        raise NotImplementedError(
            f"Constraints extraction for problem {problem} is not implemented."
        )


def _to_Y_objective(
    dframe: pd.DataFrame,
    problem: ANY_PROBLEM,
) -> np.ndarray:
    """Extracts the objective metrics from the DataFrame from the given problem."""
    if isinstance(problem, problems.GeometricalProblem):
        return dframe["max_elongation"].to_numpy()
    elif isinstance(problem, problems.SimpleToBuildQIStellarator):
        return dframe["minimum_normalized_magnetic_gradient_scale_length"].to_numpy()
    else:
        raise NotImplementedError(
            f"Objective extraction for problem {problem} is not implemented."
        )


def _n_components_that_minimizes_bic(
    X: np.ndarray,
    seed: int,
) -> int:
    """Finds the number of components that minimizes BIC for a GMM."""
    n_components_to_try = np.arange(2, 50)
    gmm_candidates = [
        mixture.GaussianMixture(int(n), random_state=seed) for n in n_components_to_try
    ]
    gmm_candidates_bics = []
    for i, model in enumerate(gmm_candidates):
        model.fit(X)
        gmm_candidates_bics.append(model.bic(X))
        print(
            f"n_components={n_components_to_try[i]}, "
            f"BIC={model.bic(X):.2f}, "
            f"converged={model.converged_}"
        )
    return n_components_to_try[np.argmin(gmm_candidates_bics)]


def _plot_data_and_samples(
    Z: np.ndarray,
    Z_samples: np.ndarray,
) -> axes.Axes:
    _, ax = plt.subplots(1, 1, tight_layout=True)
    ax.scatter(
        Z[:, 0],
        Z[:, 1],
        c="black",
        label="Original data",
    )
    ax.scatter(
        Z_samples[:, 0],
        Z_samples[:, 1],
        c="red",
        alpha=0.33,
        label="GMM samples",
    )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend()
    return ax


def _plot_regressor_oob(
    regressor: ensemble.RandomForestRegressor,
    y: np.ndarray,
) -> axes.Axes:
    oob_prediction = regressor.oob_prediction_
    _, ax = plt.subplots(1, 1, tight_layout=True)
    ax.scatter(
        oob_prediction,
        y,
    )
    ax.plot(
        [oob_prediction.min(), oob_prediction.max()],
        [y.min(), y.max()],
        color="red",
        linestyle="--",
    )
    ax.set_xlabel("OOB prediction")
    ax.set_ylabel("True value")
    return ax


def _calibrate_std_with_nll(
    regressor: ensemble.RandomForestRegressor,
    Z: np.ndarray,
    y: np.ndarray,
) -> float:
    if not getattr(regressor, "oob_prediction_", None) is not None:
        raise ValueError("RandomForest must be fit with oob_score=True")

    y_oob = regressor.oob_prediction_

    tree_predictionss = np.vstack([t.predict(Z) for t in regressor.estimators_])
    sigma = np.clip(tree_predictionss.std(axis=0), a_min=1e-6, a_max=None)

    ratios = (y - y_oob) / sigma
    return np.sqrt(np.mean(ratios**2))


def _fit_calibrated_classifier(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
    n_estimators: int,
) -> calibration.CalibratedClassifierCV:
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    classifier = ensemble.RandomForestClassifier(
        n_estimators=n_estimators,
        oob_score=True,
        n_jobs=-1,
        random_state=random_state,
    )
    classifier.fit(X_train, y_train)
    print(f"Constraint OOB accuracy: {classifier.oob_score_:.4f}")
    calibrated_classifier = calibration.CalibratedClassifierCV(
        estimator=classifier,
        cv="prefit",
    )
    calibrated_classifier.fit(X_test, y_test)
    return calibrated_classifier

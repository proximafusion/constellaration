import numpy as np
import pydantic
from scipy import stats as scipy_stats
from vmecpp import _pydantic_numpy as pydantic_numpy

from constellaration.omnigeneity import omnigenity_field


class OmnigenousFieldAndTargets(pydantic_numpy.BaseModelWithNumpy):
    """Omnigenous field and targets."""

    omnigenous_field: omnigenity_field.OmnigenousField
    """Omnigenous field."""
    rotational_transform: float | None = None
    """Targets for the rotational transform at the edge."""
    max_elongation: float | None = None
    """Target max elongation for the optimization."""
    aspect_ratio: float | None = None
    """Target max aspect ratio for the optimization."""
    major_radius: float | None = 1.0
    """Major radius to be fixed during the optimization."""


class SampleMagneticWellBetaCDFSettings(pydantic.BaseModel):
    """Settings for samplign the  magnetic well with the cdf of a beta distribution."""

    beta_min: pydantic.PositiveFloat = 2.0
    """Minimum beta parameter of the beta distribution."""

    alpha_min: pydantic.PositiveFloat = 2.0
    """Minimum alpha parameter of the beta distribution."""

    beta_max: pydantic.PositiveFloat = 6.0
    """Maximum beta parameter of the beta distribution."""

    alpha_max: pydantic.PositiveFloat = 6.0
    """Minimum alpha parameter of the beta distribution."""

    mean_modb: float = 1.0  # Default at 1T
    """Mean value of the magnetic well."""

    mirror_ratio_min: float = 0.1
    """Minimum mirror ratio of the magnetic well.

    Defined as (max - min) / (max + min).
    """

    mirror_ratio_max: float = 0.4
    """Maximum mirror ratio of the magnetic well.

    Defined as (max - min) / (max + min).
    """

    n_points_per_curve: int = 51
    """Number of points per curve to generate at equidistant intervals including the
    endpoints."""

    def sample_curves(
        self,
        seed: int | None = 42,
        n_samples: int = 10,
        betas: list[float] | None = None,
        alphas: list[float] | None = None,
        mirror_ratios: list[float] | None = None,
    ) -> list[list[float]]:
        """Samples the magnetic well using the beta distribution.

        Returns:
            List of sampled magnetic wells.
        """
        rng = np.random.default_rng(seed)

        # Check if we need to sample the parameters or if they are provided
        betas = betas or rng.uniform(self.beta_min, self.beta_max, n_samples).tolist()
        alphas = (
            alphas or rng.uniform(self.alpha_min, self.alpha_max, n_samples).tolist()
        )
        mirror_ratios = (
            mirror_ratios
            or rng.uniform(
                self.mirror_ratio_min, self.mirror_ratio_max, n_samples
            ).tolist()
        )

        assert betas is not None
        assert alphas is not None
        assert mirror_ratios is not None

        assert len(betas) == len(alphas) == len(mirror_ratios) == n_samples, (
            f"betas, alphas and mirror_ratios must have the same length as n_samples. "
            f"Got {len(betas)}, {len(alphas)} and {len(mirror_ratios)}."
        )

        x = np.linspace(0, 1, self.n_points_per_curve, endpoint=True)

        def _sample_curve(
            alpha: float, beta: float, mirror_ratio: float
        ) -> list[float]:
            """Gets the CDF of the beta distribution and scales it to a range with a
            target mean and mirror ratio."""
            cdf = scipy_stats.beta.cdf(x, alpha, beta)
            f_min = np.min(cdf)
            f_max = np.max(cdf)
            f_mean = np.mean(cdf)

            denom = (f_max - f_min) - mirror_ratio * ((f_max + f_min) - 2 * f_mean)
            scale = 2 * mirror_ratio * self.mean_modb / denom
            bias = self.mean_modb - scale * f_mean

            modb = scale * cdf + bias

            return modb.tolist()

        return [
            _sample_curve(alpha, beta, mirror_ratio)
            for alpha, beta, mirror_ratio in zip(alphas, betas, mirror_ratios)
        ]


class SampleStellaratorSymmetricOmnigenousFieldSetting(pydantic.BaseModel):
    """Settings for sampling omnigenous fields."""

    n_field_periods_min: int = 1
    """Minimum number of field periods to sample from a uniform distribution."""

    n_field_periods_max: int = 5
    """Maximum number of field periods to sample from a uniform distribution."""

    max_x_eta_coefficients: pydantic.NonNegativeInt = 2
    """Maximum number of coefficients for the x_eta coefficients."""

    max_x_alpha_coefficients: pydantic.NonNegativeInt = 2
    """Maximum number of coefficients for the x_alpha coefficients."""

    non_zero_xlmn_std_alpha: float = 0.4
    """The standard deviation of the non zero x_lmn coefficients when m or n is 0.

    $\\sigma = \\alpha \\exp(-\\beta \\max(|m|, |n|))$
    """

    non_zero_xlmn_std_beta: float = 0.4
    """How quickly the std decays with increasing m (in eta) or abs(n) (in alpha) modes.

    $\\sigma = \\alpha \\exp(-\\beta \\max(|m|, |n|))$
    """

    non_zero_xlmn_mean: float = 0.0
    """Mean of the non-zero x_lmn coefficients.

    Non zero coefficients are entries of the
    x_lmn array form the OmnigenousField class (See Eq 7 Dudt et al 2024) that allow for
    stellarator symmetry: x_lmn[:, :, n_x_alpha_coefficients // 2 :] = 0  and
    x_lmn[:, ::2, :] = 0
    """

    non_zero_xlmn_abs_cutoff: float = 1.2
    """Cutoff for the absolute value of the non-zero x_lmn coefficients.

    If a sample surpasses this value, it is set to non_zero_xlmn_abs_cutoff.
    """

    magnetic_well_settings: SampleMagneticWellBetaCDFSettings = (
        SampleMagneticWellBetaCDFSettings()
    )
    """Settings for sampling the magnetic well."""

    @property
    def n_x_alpha_coefficients(self) -> int:
        return 2 * self.max_x_alpha_coefficients + 1

    def sample_omnigenous_fields(
        self,
        seed: int | None = 42,
        n_samples: int = 10,
        n_field_periods: list[int] | None = None,
        alphas: list[float] | None = None,
        betas: list[float] | None = None,
        mirror_ratios: list[float] | None = None,
    ) -> list[omnigenity_field.OmnigenousField]:
        """Samples omnigenous fields.

        Returns:
            List of sampled omnigenous fields.
        """

        rng = np.random.default_rng(seed)

        # Check if we need to sample the parameters or if they are provided
        n_field_periods = (
            n_field_periods
            or rng.integers(
                self.n_field_periods_min, self.n_field_periods_max + 1, n_samples
            ).tolist()
        )

        assert n_field_periods is not None
        assert len(n_field_periods) == n_samples, (
            f"n_field_periods must have the same length as n_samples. "
            f"Got {len(n_field_periods)}."
        )

        # Sample the magnetic well
        modb_wells = self.magnetic_well_settings.sample_curves(
            seed=seed,
            n_samples=n_samples,
            alphas=alphas,
            betas=betas,
            mirror_ratios=mirror_ratios,
        )

        fields = []
        for modb_well, nfp in zip(modb_wells, n_field_periods):
            # Indices arrays for m (eta) and n (alpha):
            m_indices = np.arange(self.max_x_eta_coefficients + 1).reshape(
                (self.max_x_eta_coefficients + 1, 1)
            )
            n_indices = np.arange(
                -self.max_x_alpha_coefficients, self.max_x_alpha_coefficients + 1
            ).reshape((1, self.n_x_alpha_coefficients))

            # Scale law for the standard deviation of the x_lmn coefficients
            sigma = self.non_zero_xlmn_std_alpha * np.exp(
                -self.non_zero_xlmn_std_beta
                * np.maximum(np.abs(m_indices), np.abs(n_indices))
            )
            sigma = sigma[None, ...]

            # Sample x_lmn coefficients
            x_lmn = rng.normal(loc=self.non_zero_xlmn_mean, scale=sigma)

            # Clip values to the cutoff
            x_lmn = np.clip(
                x_lmn,
                -self.non_zero_xlmn_abs_cutoff,
                self.non_zero_xlmn_abs_cutoff,
            )

            # Enforce stellarator symmetry
            x_lmn[:, :, self.n_x_alpha_coefficients // 2 :] = 0
            x_lmn[:, ::2, :] = 0

            # Build modB_spline_knot_coefficients
            modb_spline_knot_coefficients = np.array(
                [modb_well, np.zeros(len(modb_well)).tolist()]
            )

            fields.append(
                omnigenity_field.OmnigenousField(
                    n_field_periods=nfp,
                    poloidal_winding=0,
                    torodial_winding=nfp,  # Only OP supported
                    x_lmn=x_lmn,
                    modB_spline_knot_coefficients=modb_spline_knot_coefficients,
                )
            )
        return fields


class SampleOmnigenousFieldAndTargetsSettings(pydantic.BaseModel):
    """Settings for sampling omnigenous fields and targets."""

    omnigenous_field_settings: SampleStellaratorSymmetricOmnigenousFieldSetting = (
        SampleStellaratorSymmetricOmnigenousFieldSetting()
    )
    """Settings for sampling the omnigenous field."""

    n_field_periods: int | None = None
    """Number of field periods.

    If not specified it will be sampled wiht settings defined in
    omnigenous_field_settings.
    """

    mirror_ratio: float | None = None
    """Mirror ratio of the omnigenous field.

    If not specified it will be sampled with settings defined in
    omnigenous_field_settings.
    """

    edge_rotational_transform_over_n_field_periods: float | None = None
    """Edge rotational transform over the number of field periods."""

    edge_rotational_transform_over_n_field_periods_min: float = 0.1
    """Minimum ratio between the edge rotational transform and the number of field
    periods to sample from a uniform distribution.

    This will be ignored if edge_rotational_transform_over_n_field_periods is not None.
    """

    edge_rotational_transform_over_n_field_periods_max: float = 0.3
    """Maximum ratio between the edge rotational transform and the number of field
    periods to sample from a uniform distribution.This will be ignored if
    edge_rotational_transform_over_n_field_periods is not None."""

    aspect_ratio: float | None = None
    """Aspect ratio of the omnigenous field."""

    aspect_ratio_min: float = 4.0
    """Minimum aspect ratio to sample from a uniform distribution.

    This will be ignored if aspect_ratio is not None.
    """

    aspect_ratio_max: float = 12.0
    """Maximum aspect ratio to sample from a uniform distribution.

    This will be ignored if aspect_ratio is not None.
    """

    max_elongation: float | None = None
    """Maximum elongation to allow."""

    max_elongation_min: float = 4.0
    """Minimum maximum elongation to sample from a uniform distribution.

    This will be ignored if max_elongation is not None.
    """
    max_elongation_max: float = 7.0
    """Maximum maximum elongation to sample from a uniform distribution.

    This will be ignored if max_elongation is not None.
    """

    major_radius: float = 1.0
    """Major radius."""

    seed: int | None = 42
    """Seed for the random number generator."""

    n_samples: int = 1000
    """Number of samples to generate."""

    def sample_omnigenous_fields_and_targets(
        self,
    ) -> list[OmnigenousFieldAndTargets]:
        """Samples omnigenous fields and targets.

        Returns:
            List of sampled omnigenous fields and targets.
        """
        rng = np.random.default_rng(self.seed)

        n_field_periods = (
            None
            if self.n_field_periods is None
            else [self.n_field_periods] * self.n_samples
        )
        mirror_ratios = (
            None if self.mirror_ratio is None else [self.mirror_ratio] * self.n_samples
        )
        omnigenous_fields = self.omnigenous_field_settings.sample_omnigenous_fields(
            seed=self.seed,
            n_samples=self.n_samples,
            n_field_periods=n_field_periods,
            mirror_ratios=mirror_ratios,
        )
        edge_rotational_transform_over_n_field_periods = (
            [self.edge_rotational_transform_over_n_field_periods] * self.n_samples
            if self.edge_rotational_transform_over_n_field_periods is not None
            else rng.uniform(
                self.edge_rotational_transform_over_n_field_periods_min,
                self.edge_rotational_transform_over_n_field_periods_max,
                self.n_samples,
            )
        )
        rotational_transfomrs = [
            iota_over_nfp * omnigenous_field.n_field_periods
            for iota_over_nfp, omnigenous_field in zip(
                edge_rotational_transform_over_n_field_periods, omnigenous_fields
            )
        ]
        aspect_ratios = (
            [self.aspect_ratio] * self.n_samples
            if self.aspect_ratio is not None
            else rng.uniform(
                self.aspect_ratio_min, self.aspect_ratio_max, self.n_samples
            )
        )

        max_elongations = (
            [self.max_elongation] * self.n_samples
            if self.max_elongation is not None
            else rng.uniform(
                self.max_elongation_min, self.max_elongation_max, self.n_samples
            )
        )

        targets = []
        for omnigenous_field, rotational_transform, aspect_ratio, max_elongation in zip(
            omnigenous_fields,
            rotational_transfomrs,
            aspect_ratios,
            max_elongations,
        ):
            targets.append(
                OmnigenousFieldAndTargets(
                    omnigenous_field=omnigenous_field,
                    rotational_transform=rotational_transform,
                    aspect_ratio=aspect_ratio,
                    max_elongation=max_elongation,
                    major_radius=self.major_radius,
                )
            )

        return targets

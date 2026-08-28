"""Settings for the SPECTRE field-integrity metrics (``run_spectre``).

Mirrors ``vmec_settings.py``: one flat pydantic model, one docstring per field, and
factory functions rather than presets. Every default was measured on the commissioning
runs described in the walkthrough page linked from ``docs/spectre_field_integrity.md``.
"""

import math

import pydantic


class PoincareSettings(pydantic.BaseModel):
    """Field-line tracing for visualisation. Not needed for the metrics."""

    n_trajectories: int = 50
    """Field lines seeded over the full radius."""
    n_points_per_trajectory: int = 400
    """Punctures recorded per field line."""


class SpectreSettings(pydantic.BaseModel):
    """What ``run_spectre`` needs beyond the equilibrium.

    Two things are deliberately not settings and live inside ``run_spectre``: the
    magnetic axis is always pinned to VMEC's, and a field whose Beltrami residual is
    above 5 and no longer falling with resolution is always refused as diverged. The
    chain search has a fixed wall-clock budget of 1800 s and, if it runs out, one last
    attempt with 8x the time.
    """

    tolerance_pct: float = pydantic.Field(default=1.0, gt=0.0, le=20.0)
    """The tolerance tau on the severity M, in percent.

    The field is solved until M has converged to tau: the toroidal-resolution ladder
    stops at the first rung whose Beltrami residual is below ``2 * tau / 100``, the
    residual below which M no longer moves with resolution on the measured population.
    The worst case can be wider; it is reported as ``trust_pct``.
    """
    max_poloidal_order: int = pydantic.Field(default=40, ge=1)
    """The screen's horizon m_max.

    The lowest rational n/m (n a multiple of the field periods, m <= m_max) the
    rotational transform crosses is the chain that is solved for; a design that crosses
    none is served severity 0 without a solve. Sets which designs are solved at all,
    hence the cost of a dataset run.
    """
    n_volumes: int = pydantic.Field(default=1, ge=1)
    """Number of SPECTRE volumes. One volume is the vacuum field the metrics use."""
    poloidal_floor: int = pydantic.Field(default=14, ge=1)
    """Poloidal Fourier resolution floor.

    mpol = max(poloidal_floor, ceil(poloidal_per_order * m)), m the order of the chain
    the transform crosses.
    """
    poloidal_per_order: float = pydantic.Field(default=1.5, gt=0.0)
    """The factor in the mpol rule above."""
    toroidal_ladder: tuple[int, ...] = (14, 18, 22, 26)
    """The toroidal resolutions ntor solved in turn, until the residual meets the
    tolerance."""
    radial_resolution: int | None = None
    """Chebyshev degree of the single volume. ``None`` means ``mpol + 4``."""
    poincare: PoincareSettings | None = None
    """Trace field lines for visualisation; ``None`` traces nothing."""
    max_threads: int = 1
    """Number of threads for the SPECTRE solve.

    Multithreaded solves are not bit-reproducible, because truncation errors accumulate
    differently based on the completion order of threads. Keep this at 1 when the
    numbers must be reproducible.
    """

    @pydantic.field_validator("toroidal_ladder")
    @classmethod
    def _ladder_increasing(cls, ladder: tuple[int, ...]) -> tuple[int, ...]:
        if not ladder or any(b <= a for a, b in zip(ladder, ladder[1:])):
            raise ValueError("toroidal_ladder must be a non-empty increasing tuple")
        if ladder[0] < 1:
            raise ValueError("toroidal_ladder values must be >= 1")
        return ladder

    @property
    def stop_residual(self) -> float:
        """The Beltrami residual below which the ladder stops: ``2 * tau / 100``."""
        return 2.0 * self.tolerance_pct / 100.0

    def poloidal_modes(self, m: int) -> int:
        """mpol for a chain of order ``m``."""
        return max(self.poloidal_floor, math.ceil(self.poloidal_per_order * m))

    def radial_modes(self, m: int) -> int:
        """lrad for a chain of order ``m``: ``radial_resolution`` or ``mpol + 4``."""
        if self.radial_resolution is not None:
            return self.radial_resolution
        return self.poloidal_modes(m) + 4


def spectre_settings_metrics() -> SpectreSettings:
    """The settings the field-integrity metrics were commissioned with."""
    return SpectreSettings()

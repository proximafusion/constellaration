"""Settings for the SPECTRE field-integrity metrics."""

import math
from typing import Literal

import pydantic


class SpectreSettings(pydantic.BaseModel):
    """What ``run_spectre`` needs beyond the equilibrium.

    Two things are deliberately not settings: the magnetic axis is always pinned to
    VMEC's, and a field whose Beltrami residual is above
    ``spectre.RESIDUAL_DIVERGED`` and no longer falling with resolution is always
    refused as diverged.
    """

    tolerance_pct: float = pydantic.Field(default=1.0, gt=0.0, le=20.0)
    """The tolerance tau on the field integrity score M, in percent.

    The field is solved until M has converged to tau: the toroidal-resolution ladder
    stops at the first rung whose Beltrami residual is below ``2 * tau / 100``, the
    residual below which M no longer moves with resolution on the measured population.
    The residual reached is kept in the record (``beltrami_residual``).
    """
    max_poloidal_order: int = pydantic.Field(default=20, ge=1)
    """The screen's horizon m_max, inclusive: what counts as a low-order rational.

    A rational n/m (n a multiple of the field periods) the rotational transform
    crosses is solved for only when m <= m_max. This is an entry screen, not a width
    estimate: it rests on the premise that a chain of higher order cannot destroy a
    significant share of the flux. A design that crosses no rational up to m_max is
    given a field integrity score of 0 without a solve, so this sets which designs are
    solved at all.
    """
    max_rationals: int | None = pydantic.Field(default=3, ge=1)
    """How many of the crossed rationals are scored, lowest order first.

    The field integrity score is the sum over the chains of the ``max_rationals``
    lowest-order rationals the transform crosses. The lowest-order chain is not always
    the widest one, which is why the default is not 1; ``None`` scores every crossed
    rational up to ``max_poloidal_order``. The field is solved once, at the resolution
    the highest selected order asks for, so this sets the cost of the solve.
    """
    n_volumes: Literal[1] = 1
    """Number of SPECTRE volumes. One volume is the vacuum field the metrics use."""
    poloidal_floor: int = pydantic.Field(default=14, ge=1)
    """Poloidal Fourier resolution floor.

    mpol = max(poloidal_floor, ceil(poloidal_per_order * m)), m the highest order
    among the selected rationals.
    """
    poloidal_per_order: float = pydantic.Field(default=1.5, gt=0.0)
    """The factor in the mpol rule above."""
    toroidal_ladder: tuple[int, ...] = (14, 18, 22, 26)
    """The toroidal resolutions ntor solved in turn, until the residual meets the
    tolerance."""
    radial_resolution: int | None = None
    """Chebyshev degree of the single volume. ``None`` means ``mpol + 4``."""
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


def spectre_settings_metrics() -> SpectreSettings:
    """The settings the field-integrity metrics were commissioned with."""
    return SpectreSettings()

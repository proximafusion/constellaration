"""SPECTRE field-integrity metrics: the islands VMEC cannot see.

VMEC assumes nested flux surfaces, so its equilibrium cannot show the magnetic islands
its own boundary hosts. SPECTRE (https://gitlab.com/spectre-eq/spectre) re-solves the
boundary without that assumption. This module is the interface:

    output = run_spectre(wout, settings)               # screen, solve, chain search
    metrics = compute_field_integrity_metrics(output)  # the severity M, its receipts

The pure-Python parts -- the resonance screen, the resolution-ladder controller and the
reduction to metrics -- run without SPECTRE and are tested. The two steps that need a
field (the Beltrami solve and the fixed-point search) import ``spectre`` lazily and
raise ``SpectreNotAvailableError`` when it is not installed; SPECTRE is a Fortran source
build with no PyPI release (see ``docs/spectre_field_integrity.md``).

Definitions, with the constants that are part of the algorithm rather than settings:

* severity  M = N_fp (R_O R_X)^(1/4) / (m^2 iota'), summed over distinct chains;
  R_O, R_X the Greene residues of the O- and X-points, iota' the unperturbed shear at
  the resonance. Inside the pendulum domain (|R_O| <= 0.10, |R_X/R_O| in [0.85, 1.15])
  the flux the island destroys is Delta Phi / Phi_edge = kappa (4/pi) M, kappa = 0.648.
* trust_pct = 0.41 + 100 C(residual) residual, C in bands (1.97, 3.28, 9.73, 11.5):
  the most M could still move with higher resolution, in percent; None above 0.1.
* diverged: residual > 5 and not falling from the previous rung (a first-rung residual
  above 5 gets one more rung).
* chain search: wall-clock budget 1800 s, one 8x re-search when
  N_lib = pi m / sqrt(R_O) exceeds 1e5.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pydantic

from constellaration.mhd.spectre_settings import SpectreSettings

METRICS_VERSION = "2026-08-27-t35"
KAPPA = 0.648
RESIDUAL_DIVERGED = 5.0
TRUST_ENVELOPE_BANDS = ((1e-3, 1.97), (5e-3, 3.28), (1e-2, 9.73), (1e-1, 11.5))
TRUST_POLOIDAL_PCT = 0.41
PENDULUM_MAX_RESIDUE = 0.10
PENDULUM_RATIO_RANGE = (0.85, 1.15)
SEARCH_BUDGET_S = 1800.0
SEARCH_ESCALATION = 8.0
N_LIB_ESCALATE = 1e5


class SpectreNotAvailableError(ImportError):
    """A step needs SPECTRE and the ``spectre`` package cannot be imported."""


class RefusalClass(str, enum.Enum):
    """Why ``severity`` is 0 or None. A design that enters always gets a row."""

    NONE = "NONE"
    NO_RATIONAL = "NO_RATIONAL"
    """No rational n/m with m <= max_poloidal_order inside the transform: severity 0."""
    SIGN_INDEFINITE = "SIGN_INDEFINITE"
    """The transform passes through zero: both helicities in play, severity None."""
    FIELD_DIVERGED = "FIELD_DIVERGED"
    """Beltrami residual above 5 and not falling: the field is not a solution."""
    SEARCH_INCOMPLETE = "SEARCH_INCOMPLETE"
    """No O/X pair within the budget and its one 8x re-search."""
    NONTWIST_MERGED = "NONTWIST_MERGED"
    """Two crossings whose islands overlap: one reconnected system, the sum withheld."""


class Resonance(pydantic.BaseModel):
    """A rational the transform crosses, as the field's harmonics see it."""

    n: int
    m: int
    n_crossings: int
    shear: float
    """d iota / d psi_n at the crossing (the unperturbed shear, VMEC's profile)."""

    @property
    def iota(self) -> float:
        return self.n / self.m


class IslandChain(pydantic.BaseModel):
    """One chain the search found, at one crossing of n/m."""

    n: int
    m: int
    psi_n: float
    residue_o: float
    residue_x: float
    shear: float
    budget_exhausted: bool = False


class SpectreOutput(pydantic.BaseModel):
    """What ``run_spectre`` returns; the metrics are computed from this alone."""

    settings: SpectreSettings
    n_field_periods: int
    refusal_class: RefusalClass = RefusalClass.NONE
    resonance: Resonance | None = None
    field_h5: str | None = None
    """Path of the solved field, so the metrics can be recomputed without a re-solve."""
    n_poloidal_modes: int | None = None
    n_toroidal_modes: int | None = None
    radial_resolution: int | None = None
    beltrami_residual: float | None = None
    ladder_rungs_solved: int = 0
    axis_offset_mm: float | None = None
    """Distance between the VMEC axis and the axis SPECTRE was pinned to."""
    chains: list[IslandChain] = []
    solve_seconds: float = 0.0
    search_seconds: float = 0.0
    peak_rss_mib: float | None = None
    cores: int = 1
    spectre_version: str | None = None


class FieldIntegrityMetrics(pydantic.BaseModel):
    """The metrics row. Flat floats, like ``ConstellarationMetrics``."""

    severity: float | None
    severity_poloidal_mode: int | None
    n_chains_enumerated: int
    n_chains_found: int
    residue_o: float | None
    residue_x: float | None
    residue_ratio: float | None
    pendulum_domain: bool
    beltrami_residual: float | None
    trust_pct: float | None
    merged: bool
    refusal_class: RefusalClass
    metrics_version: str = METRICS_VERSION


# ------------------------------------------------------------------------------------
# step 1 -- the screen (pure Python)
# ------------------------------------------------------------------------------------


def admissible(n: int, m: int, n_field_periods: int) -> bool:
    """Is (n, m) the harmonic that drives the rational n/m in an N_fp-periodic field?

    The field carries toroidal harmonics that are multiples of N_fp only. A rational
    p/q in lowest terms resonates with (k p, k q), k = N_fp / gcd(p, N_fp), and the
    chain has m = k q islands. ``(2, 4)`` is admissible at N_fp = 2 (iota = 1/2 is a
    four-island chain there); ``(1, 2)`` and ``(4, 8)`` are not.
    """
    if n <= 0 or m <= 0 or n % n_field_periods != 0:
        return False
    g = math.gcd(n, m)
    p = n // g
    return g == n_field_periods // math.gcd(p, n_field_periods)


def lowest_order_resonance(
    iota: npt.ArrayLike,
    psi_n: npt.ArrayLike,
    n_field_periods: int,
    max_poloidal_order: int,
) -> Resonance | None:
    """The lowest-order admissible rational the transform crosses, with its shear.

    ``iota`` is the signed VMEC profile on the normalised-flux grid ``psi_n``; the
    screen runs on |iota| (the sign is a coordinate convention). Returns None when no
    admissible rational of order <= ``max_poloidal_order`` lies in the band.
    """
    io = np.abs(np.asarray(iota, dtype=float))
    s = np.asarray(psi_n, dtype=float)
    lo, hi = float(io.min()), float(io.max())
    best: tuple[int, int] | None = None
    nfp = n_field_periods
    for m in range(1, max_poloidal_order + 1):
        for n in range(nfp, math.ceil(hi * m) + nfp, nfp):
            if lo <= n / m <= hi and admissible(n, m, nfp):
                best = (n, m)
                break
        if best is not None:
            break
    if best is None:
        return None
    n, m = best
    target = n / m
    side = np.sign(io - target)
    for k in range(1, side.size):  # a grid point exactly on n/m keeps its previous side
        if side[k] == 0:
            side[k] = side[k - 1]
    crossings = np.flatnonzero(side[:-1] != side[1:])
    if crossings.size == 0:  # a tangency at a grid point
        crossings = np.array([int(np.argmin(np.abs(io - target)))])
    k = int(crossings[0])
    k0, k1 = max(k - 1, 0), min(k + 2, io.size - 1)
    shear = float((io[k1] - io[k0]) / (s[k1] - s[k0])) if s[k1] != s[k0] else 0.0
    return Resonance(n=n, m=m, n_crossings=int(crossings.size), shear=abs(shear))


def screen(
    iota: npt.ArrayLike,
    psi_n: npt.ArrayLike,
    n_field_periods: int,
    settings: SpectreSettings,
) -> tuple[RefusalClass, Resonance | None]:
    """Step 1: refuse at the screen, or name the chain to solve for."""
    io = np.asarray(iota, dtype=float)
    if io.min() < 0.0 < io.max():
        return RefusalClass.SIGN_INDEFINITE, None
    res = lowest_order_resonance(
        io, psi_n, n_field_periods, settings.max_poloidal_order
    )
    if res is None:
        return RefusalClass.NO_RATIONAL, None
    return RefusalClass.NONE, res


# ------------------------------------------------------------------------------------
# steps 2-3 -- resolution and the ntor ladder (pure controller)
# ------------------------------------------------------------------------------------


def ladder(
    solve: Callable[[int], float], settings: SpectreSettings
) -> tuple[list[float], bool]:
    """Run the toroidal-resolution ladder on a solver ``solve(ntor) -> residual``.

    Climb while the residual is above ``settings.stop_residual``; refuse as diverged
    when it is above ``RESIDUAL_DIVERGED`` and did not fall from the previous rung (or
    there is no rung left) -- a first-rung residual above 5 gets one more rung. Returns
    the residuals of the rungs solved and whether the field diverged.
    """
    residuals: list[float] = []
    for i, ntor in enumerate(settings.toroidal_ladder):
        r = float(solve(ntor))
        residuals.append(r)
        last = i == len(settings.toroidal_ladder) - 1
        if r > RESIDUAL_DIVERGED:
            falling = i == 0 or r < residuals[-2]
            if not falling or last:
                return residuals, True
            continue
        if r <= settings.stop_residual:
            return residuals, False
    return residuals, False


# ------------------------------------------------------------------------------------
# step 4 -- run_spectre
# ------------------------------------------------------------------------------------


def _require_spectre() -> Any:
    try:
        import spectre  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SpectreNotAvailableError(
            "run_spectre needs the SPECTRE package"
            " (https://gitlab.com/spectre-eq/spectre), a Fortran source build with no"
            " PyPI release; see docs/spectre_field_integrity.md for how to install it."
        ) from exc
    return spectre


def run_spectre(wout: Any, settings: SpectreSettings) -> SpectreOutput:
    """Screen the equilibrium, solve the field along the ladder, search the chain.

    ``wout`` is a ``VmecppWOut`` (``iota_full``, ``nfp``, ``phi``, ``rmnc``/``zmns``
    and the axis are read; nothing else). The screen runs without SPECTRE; a design
    refused there is returned immediately. The solve and the search need SPECTRE.
    """
    iota = np.asarray(wout.iota_full, dtype=float)
    phi = np.asarray(wout.phi, dtype=float)
    psi_n = phi / phi[-1]
    n_fp = int(wout.nfp)
    refusal, res = screen(iota, psi_n, n_fp, settings)
    out = SpectreOutput(settings=settings, n_field_periods=n_fp, refusal_class=refusal)
    if res is None:
        return out
    out.resonance = res

    spectre = _require_spectre()
    # The solve and the search are the SPECTRE-specific part of the pipeline: build
    # the single-volume input from the wout with the axis pinned, solve at
    # (mpol = settings.poloidal_modes(m), ntor from the ladder), read the Beltrami
    # residual, then search the chain at iota = n/m with ``spectre.fixed_points``.
    # The reference implementation lives in the commissioning repository and is not
    # part of this pull request; the interface above it -- what goes in, what comes
    # out -- is.
    version = getattr(spectre, "__version__", "")
    raise NotImplementedError(
        f"SPECTRE {version} is installed, but the solve and search steps are not wired"
        " into this module yet (see the pull request notes)."
    )


# ------------------------------------------------------------------------------------
# steps 5-7 -- the metrics
# ------------------------------------------------------------------------------------


def _severity(
    m: int, residue_o: float, residue_x: float, shear: float, n_fp: int
) -> float:
    return n_fp * abs(residue_o * residue_x) ** 0.25 / (m * m * abs(shear))


def trust_envelope_pct(residual: float | None) -> float | None:
    """The most M could still move with higher resolution, in percent of M."""
    if residual is None:
        return None
    for edge, c in TRUST_ENVELOPE_BANDS:
        if residual <= edge:
            return TRUST_POLOIDAL_PCT + 100.0 * c * residual
    return None


def in_pendulum_domain(residue_o: float, residue_x: float) -> bool:
    """|R_O| <= 0.10 and |R_X / R_O| in [0.85, 1.15]: a single-harmonic pendulum."""
    if residue_o == 0.0:
        return False
    ratio = abs(residue_x / residue_o)
    lo, hi = PENDULUM_RATIO_RANGE
    return abs(residue_o) <= PENDULUM_MAX_RESIDUE and lo <= ratio <= hi


def merged(chains: Sequence[IslandChain], excursion_psi: float, n_fp: int) -> bool:
    """Two crossings whose islands would overlap are one reconnected system.

    ``excursion_psi`` is how far the transform overshoots n/m between the crossings, in
    normalised flux; it is compared with the pendulum half-width the chains' own
    severity predicts, ``0.5 * kappa * (4 / pi) * max M``.
    """
    if len(chains) < 2:
        return False
    widest = max(_chain_severity(c, n_fp) for c in chains)
    half_width = 0.5 * KAPPA * (4.0 / math.pi) * widest
    return excursion_psi < half_width


def _chain_severity(chain: IslandChain, n_fp: int) -> float:
    return _severity(chain.m, chain.residue_o, chain.residue_x, chain.shear, n_fp)


def compute_field_integrity_metrics(
    output: SpectreOutput, excursion_psi: float | None = None
) -> FieldIntegrityMetrics:
    """Reduce a ``SpectreOutput`` to the metrics row.

    ``excursion_psi`` is needed only for designs with several crossings (the merge
    check); ``None`` skips that check.
    """
    row: dict[str, Any] = dict(
        severity_poloidal_mode=None,
        n_chains_enumerated=0,
        n_chains_found=0,
        residue_o=None,
        residue_x=None,
        residue_ratio=None,
        pendulum_domain=False,
        beltrami_residual=output.beltrami_residual,
        trust_pct=None,
        merged=False,
    )
    if output.refusal_class == RefusalClass.NO_RATIONAL:
        return FieldIntegrityMetrics(
            severity=0.0, refusal_class=output.refusal_class, **row
        )
    if output.refusal_class != RefusalClass.NONE or output.resonance is None:
        return FieldIntegrityMetrics(
            severity=None, refusal_class=output.refusal_class, **row
        )
    res = output.resonance
    n_fp = output.n_field_periods
    chains = output.chains
    row["n_chains_enumerated"] = res.n_crossings
    row["n_chains_found"] = len(chains)
    row["severity_poloidal_mode"] = res.m
    if not chains:
        return FieldIntegrityMetrics(
            severity=None, refusal_class=RefusalClass.SEARCH_INCOMPLETE, **row
        )
    is_merged = excursion_psi is not None and merged(chains, excursion_psi, n_fp)
    row["merged"] = is_merged
    lead = max(chains, key=lambda c: abs(c.residue_o))
    row["residue_o"] = lead.residue_o
    row["residue_x"] = lead.residue_x
    row["residue_ratio"] = (
        abs(lead.residue_x / lead.residue_o) if lead.residue_o else None
    )
    row["pendulum_domain"] = in_pendulum_domain(lead.residue_o, lead.residue_x)
    row["trust_pct"] = trust_envelope_pct(output.beltrami_residual)
    if is_merged:
        return FieldIntegrityMetrics(
            severity=None, refusal_class=RefusalClass.NONTWIST_MERGED, **row
        )
    severity = sum(_chain_severity(c, n_fp) for c in chains)
    return FieldIntegrityMetrics(
        severity=severity, refusal_class=RefusalClass.NONE, **row
    )


def predicted_flux_fraction(severity: float) -> float:
    """Delta Phi / Phi_edge = kappa (4 / pi) M, validated inside the pendulum domain."""
    return KAPPA * (4.0 / math.pi) * severity

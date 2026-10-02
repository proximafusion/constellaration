"""SPECTRE field-integrity metrics: the islands VMEC cannot see.

VMEC assumes nested flux surfaces, so its equilibrium cannot show the magnetic
islands its own boundary hosts. SPECTRE (https://gitlab.com/spectre-eq/spectre)
re-solves the boundary without that assumption and scores the design's field
integrity:

    metrics = compute_spectre_metrics(equilibrium, settings)

where ``equilibrium`` is the VMEC++ equilibrium the design was solved to (a
``vmec_utils.VmecppWOut``, as ``run_vmec`` returns it) and ``settings`` a
``SpectreSettings``.

SPECTRE is a dependency of this package (pinned to a commit in ``pyproject.toml``).
The resonance screen, the resolution-ladder controller and the reduction to
metrics are pure Python; the Beltrami solve and the fixed-point search use it.

The algorithm:

1. **Screen.** Read the rotational transform and list every rational n/m it
   crosses, with n a multiple of the field periods and m <= ``max_poloidal_order``
   (``crossed_resonances``); ``select_resonances`` then picks which of them the
   severity is built from -- today the lowest order alone. That chain's unperturbed
   shear at its crossing is the severity's denominator. A design that
   crosses none is served severity 0 (``NO_RATIONAL``); a rotational transform
   that passes through zero is served None (``SIGN_INDEFINITE``). No solve.
2. **Resolution.** ``mpol = max(poloidal_floor, ceil(poloidal_per_order * m))``,
   ``lrad = mpol + 4``, one volume, the magnetic axis pinned to VMEC's.
3. **Toroidal ladder.** Solve at each ``ntor`` of ``toroidal_ladder`` in turn,
   reading the Beltrami residual after each; stop at the first rung below
   ``stop_residual``. A residual above ``RESIDUAL_DIVERGED`` that no longer falls
   with resolution is refused as ``FIELD_DIVERGED``.
4. **Chain search.** One O-point and one X-point of the chain at iota = n/m,
   classified by the tangent map (Greene residues R_O, R_X), within
   ``SEARCH_BUDGET_S`` and its one ``SEARCH_ESCALATION`` retry when the chain is
   ill-conditioned. Nothing found -> ``SEARCH_INCOMPLETE``.
5. **Several crossings.** A rotational transform that crosses n/m more than once
   hosts a chain at each crossing, and their severities are summed. Two crossings
   close enough for their islands to have reconnected are one structure, and
   summing those double-counts; the threshold that decides it is not settled, so
   no merge check is applied here.
6. **Pendulum domain.** ``|R_O| <= PENDULUM_MAX_RESIDUE`` and ``|R_X/R_O|`` in
   ``PENDULUM_RATIO_RANGE``: where the flux relation was validated. Outside it the
   severity is still served and the predicted flux is flagged.
7. **The record**: severity, ``trust_pct``, the residues, ``pendulum_domain``, the
   Beltrami residual and ``refusal_class``.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pydantic
import spectre as spectre_backend

from constellaration.mhd import vmec_utils
from constellaration.mhd.spectre_settings import SpectreSettings

# Measured constants of the algorithm, not settings: each was fitted on the
# commissioning population and changing one changes what the metrics mean.

KAPPA = 0.648
"""Proportionality between the severity and the flux the island destroys."""

RESIDUAL_DIVERGED = 5.0
"""Beltrami residual above which a field that is no longer converging is refused."""

TRUST_ENVELOPE_BANDS = ((1e-3, 1.97), (5e-3, 3.28), (1e-2, 9.73), (1e-1, 11.5))
"""(residual, C) bands of the trust envelope; C multiplies the residual."""

TRUST_POLOIDAL_PCT = 0.41
"""Poloidal-resolution floor of the trust envelope, in percent of the severity."""

PENDULUM_MAX_RESIDUE = 0.10
"""Largest |R_O| for which the single-harmonic pendulum picture was validated."""

PENDULUM_RATIO_RANGE = (0.85, 1.15)
"""Range of |R_X / R_O| over which the pendulum picture was validated."""

SEARCH_BUDGET_S = 1800.0
"""Wall-clock budget of one chain search, in seconds."""

SEARCH_ESCALATION = 8.0
"""Budget multiplier of the single re-search granted to an ill-conditioned chain."""

N_LIB_ESCALATE = 1e5
"""Value of ``pi m / sqrt(R_O)`` above which that re-search is granted."""


class RefusalClass(str, enum.Enum):
    """Why ``severity`` is 0 or None. A design that enters always gets a row."""

    NONE = "NONE"
    NO_RATIONAL = "NO_RATIONAL"
    """No rational n/m with m <= max_poloidal_order in the rotational transform."""
    SIGN_INDEFINITE = "SIGN_INDEFINITE"
    """The rotational transform passes through zero: both helicities, severity None."""
    FIELD_DIVERGED = "FIELD_DIVERGED"
    """Beltrami residual above 5 and not falling: the field is not a solution."""
    SEARCH_INCOMPLETE = "SEARCH_INCOMPLETE"
    """No O/X pair within the budget and its one 8x re-search."""


class ScreenedResonance(pydantic.BaseModel):
    """A rational the rotational transform crosses, as the screen reads it off VMEC.

    Produced by :func:`screen` from the rotational transform alone, before any
    field is solved. One screened resonance yields up to ``n_crossings``
    :class:`SearchedChain` -- and none at all when the search comes up empty,
    which is ``SEARCH_INCOMPLETE`` rather than an absent resonance.
    """

    n: int
    m: int
    n_crossings: int
    shear: float
    """d iota / d psi_n at the crossing (the unperturbed shear, VMEC's profile)."""

    @property
    def iota(self) -> float:
        return self.n / self.m


class SearchedChain(pydantic.BaseModel):
    """One chain the fixed-point search found in the solved field.

    Produced by the search, one per crossing of its :class:`ScreenedResonance`, so
    its residues exist only once a field has been solved and searched.
    """

    n: int
    m: int
    psi_n: float
    residue_o: float
    residue_x: float
    shear: float
    budget_exhausted: bool = False


class FieldIntegrityMetrics(pydantic.BaseModel):
    """The metrics row for one design."""

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
    refusal_class: RefusalClass


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


def _resonance_at(
    io: npt.NDArray[np.float64], s: npt.NDArray[np.float64], n: int, m: int
) -> ScreenedResonance:
    """Crossing count and unperturbed shear of ``n/m`` on this transform."""
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
    return ScreenedResonance(
        n=n, m=m, n_crossings=int(crossings.size), shear=abs(shear)
    )


def crossed_resonances(
    iota: npt.ArrayLike,
    psi_n: npt.ArrayLike,
    n_field_periods: int,
    max_poloidal_order: int,
) -> list[ScreenedResonance]:
    """Every admissible rational the rotational transform crosses, lowest order first.

    Enumeration, with no policy in it: which of these the severity is built from is
    :func:`select_resonances`. ``iota`` is the signed VMEC profile on the
    normalised-flux grid ``psi_n``; the screen runs on |iota|, the sign being a
    coordinate convention. Empty when the transform crosses no admissible rational of
    order <= ``max_poloidal_order``.
    """
    io = np.abs(np.asarray(iota, dtype=float))
    s = np.asarray(psi_n, dtype=float)
    lo, hi = float(io.min()), float(io.max())
    nfp = n_field_periods
    out: list[ScreenedResonance] = []
    for m in range(1, max_poloidal_order + 1):
        for n in range(nfp, math.ceil(hi * m) + nfp, nfp):
            if lo <= n / m <= hi and admissible(n, m, nfp):
                out.append(_resonance_at(io, s, n, m))
    return out


def select_resonances(
    candidates: Sequence[ScreenedResonance],
) -> list[ScreenedResonance]:
    """Which of the crossed rationals the severity is built from.

    Today: the lowest order alone, on the premise that island width falls steeply with
    poloidal order. That premise is not safe design-by-design -- a higher-order chain
    in the same field can be the wider one -- so the choice lives here rather than
    welded into the enumeration. Alternatives that need only this function to change:
    the largest severity over the k lowest orders, every crossing below some order, or
    the sum over distinct rationals. Which the dataset should serve is undecided; a
    rule that needs tuning takes its knob on this function, not on the enumeration.
    """
    return list(candidates[:1])


def screen(
    iota: npt.ArrayLike,
    psi_n: npt.ArrayLike,
    n_field_periods: int,
    settings: SpectreSettings,
) -> tuple[RefusalClass, list[ScreenedResonance]]:
    """Step 1: refuse at the screen, or name the chains to solve for."""
    io = np.asarray(iota, dtype=float)
    if io.min() < 0.0 < io.max():
        return RefusalClass.SIGN_INDEFINITE, []
    candidates = crossed_resonances(
        io, psi_n, n_field_periods, settings.max_poloidal_order
    )
    if not candidates:
        return RefusalClass.NO_RATIONAL, []
    return RefusalClass.NONE, select_resonances(candidates)


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
# steps 2-4 -- the field: solve along the ladder, then search
# ------------------------------------------------------------------------------------


def _solve_and_search(
    equilibrium: vmec_utils.VmecppWOut,
    settings: SpectreSettings,
    resonance: ScreenedResonance,
) -> tuple[list[SearchedChain], float | None, bool]:
    """Solve the field along the ladder and search the chain at ``n/m``.

    Returns the chains found, the Beltrami residual of the field they were found in,
    and whether the field diverged.
    """
    # Build the single-volume input from the equilibrium with the axis pinned, solve
    # at (mpol, ntor from the ladder), read the Beltrami residual, then search the
    # chain at iota = n/m with ``spectre.fixed_points``. The reference implementation
    # lives in the commissioning repository and is not part of this pull request; the
    # interface above it -- what goes in, what comes out -- is.
    version = getattr(spectre_backend, "__version__", "") or "(unknown version)"
    mpol = settings.poloidal_modes(resonance.m)
    raise NotImplementedError(
        f"SPECTRE {version} is installed, but the solve and search steps are not"
        " wired into this module yet: this design would be solved at"
        f" N_fp = {int(equilibrium.n_field_periods)}, mpol = {mpol},"
        f" lrad = {settings.radial_modes(resonance.m)} along ntor ="
        f" {list(settings.toroidal_ladder)}, then searched at iota ="
        f" {resonance.n}/{resonance.m}."
    )


# ------------------------------------------------------------------------------------
# steps 5-7 -- the metrics
# ------------------------------------------------------------------------------------


def _severity(
    m: int, residue_o: float, residue_x: float, shear: float, n_fp: int
) -> float:
    return n_fp * abs(residue_o * residue_x) ** 0.25 / (m * m * abs(shear))


def trust_envelope_pct(residual: float | None) -> float | None:
    """The most the severity could still move with higher resolution, in percent."""
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


def _chain_severity(chain: SearchedChain, n_fp: int) -> float:
    return _severity(chain.m, chain.residue_o, chain.residue_x, chain.shear, n_fp)


def _metrics_from(
    *,
    refusal_class: RefusalClass,
    resonance: ScreenedResonance | None,
    chains: Sequence[SearchedChain],
    beltrami_residual: float | None,
    n_field_periods: int,
) -> FieldIntegrityMetrics:
    """Reduce a screened resonance and the chains found for it to the metrics row."""
    row: dict[str, Any] = dict(
        severity_poloidal_mode=None,
        n_chains_enumerated=0,
        n_chains_found=0,
        residue_o=None,
        residue_x=None,
        residue_ratio=None,
        pendulum_domain=False,
        beltrami_residual=beltrami_residual,
        trust_pct=None,
    )
    if refusal_class == RefusalClass.NO_RATIONAL:
        return FieldIntegrityMetrics(severity=0.0, refusal_class=refusal_class, **row)
    if refusal_class != RefusalClass.NONE or resonance is None:
        return FieldIntegrityMetrics(severity=None, refusal_class=refusal_class, **row)
    row["n_chains_enumerated"] = resonance.n_crossings
    row["n_chains_found"] = len(chains)
    row["severity_poloidal_mode"] = resonance.m
    if not chains:
        return FieldIntegrityMetrics(
            severity=None, refusal_class=RefusalClass.SEARCH_INCOMPLETE, **row
        )
    lead = max(chains, key=lambda c: abs(c.residue_o))
    row["residue_o"] = lead.residue_o
    row["residue_x"] = lead.residue_x
    row["residue_ratio"] = (
        abs(lead.residue_x / lead.residue_o) if lead.residue_o else None
    )
    row["pendulum_domain"] = in_pendulum_domain(lead.residue_o, lead.residue_x)
    row["trust_pct"] = trust_envelope_pct(beltrami_residual)
    severity = sum(_chain_severity(c, n_field_periods) for c in chains)
    return FieldIntegrityMetrics(
        severity=severity, refusal_class=RefusalClass.NONE, **row
    )


def compute_spectre_metrics(
    equilibrium: vmec_utils.VmecppWOut, settings: SpectreSettings
) -> FieldIntegrityMetrics:
    """Score an equilibrium for the magnetic islands VMEC cannot see.

    Screens the rotational transform, and where the screen names a chain, solves the
    field along the toroidal ladder and searches that chain. ``iotaf``, ``nfp``,
    ``phi``, ``rmnc``/``zmns`` and the axis are read; nothing else.
    """
    iota = np.asarray(equilibrium.iotaf, dtype=float)
    phi = np.asarray(equilibrium.phi, dtype=float)
    psi_n = phi / phi[-1]
    n_field_periods = int(equilibrium.n_field_periods)
    refusal_class, resonances = screen(iota, psi_n, n_field_periods, settings)
    if not resonances:
        return _metrics_from(
            refusal_class=refusal_class,
            resonance=None,
            chains=[],
            beltrami_residual=None,
            n_field_periods=n_field_periods,
        )
    # ``select_resonances`` serves one rational today, and the reduction below scores
    # that one. Serving several is the open question its docstring names.
    resonance = resonances[0]
    chains, beltrami_residual, diverged = _solve_and_search(
        equilibrium, settings, resonance
    )
    if diverged:
        refusal_class = RefusalClass.FIELD_DIVERGED
    return _metrics_from(
        refusal_class=refusal_class,
        resonance=resonance,
        chains=chains,
        beltrami_residual=beltrami_residual,
        n_field_periods=n_field_periods,
    )


def predicted_flux_fraction(severity: float) -> float:
    """Delta Phi / Phi_edge = kappa (4 / pi) M, validated inside the pendulum domain."""
    return KAPPA * (4.0 / math.pi) * severity

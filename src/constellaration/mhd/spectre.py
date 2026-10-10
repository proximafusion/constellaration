"""SPECTRE field-integrity metrics: the islands VMEC cannot see.

VMEC assumes nested flux surfaces through the ideal-MHD assumption, so its equilibrium
cannot show magnetic islands. SPECTRE (https://gitlab.com/spectre-eq/spectre) solves
for the magnetic field without that assumption. We can then assign a field integrity
score, M, to the design, which basically says how big its islands are.
This is done in two procedures:

    output = run_spectre(equilibrium, settings)
    metrics = compute_field_integrity_metrics(output, equilibrium)

where

- ``equilibrium`` is the VMEC++ equilibrium the design was solved to (a
  ``vmec_utils.VmecppWOut``, as ``run_vmec`` returns it)
- ``settings`` is a ``SpectreSettings``
- ``output`` holds the complete SPECTRE HDF5 file, so the metrics can be recomputed
  from a stored field without solving it again.

The algorithm:

First procedure (``run_spectre``):

1. **Resonance screen.** Read the VMEC rotational transform and list every
   crossing of a rational n/m (``_find_crossings``), where n is a multiple of the
   number of field periods and m is kept low (m <= ``max_poloidal_order``). Then
   select the ``max_rationals`` lowest-order rationals (e.g. the 3 lowest-order
   crossings). A design that crosses no low-order rational, so that no island is
   expected to open, is given a field integrity score of 0 (``NO_RATIONAL``) and its
   field is not solved.
2. **Poloidal resolution.** The field must be resolved finely enough for the islands
   to appear: ``mpol = max(poloidal_floor, ceil(poloidal_per_order * m))`` for the
   highest order m selected. The radial resolution is set to ``lrad = mpol + 4``.
   The number of volumes is 1 (volume discretization doesn't matter in vacuum).
   The magnetic axis is pinned to VMEC's (i.e. we re-use the same axis position
   VMEC found as our axis for the SPECTRE field calculation).
3. **Toroidal ladder.** The toroidal resolution is raised until the field is
   converged: solve at each ``ntor`` of ``toroidal_ladder`` in turn, reading the
   Beltrami residual after each, and stop at the first rung below ``stop_residual``.
   A residual above ``RESIDUAL_DIVERGED`` that no longer falls with resolution is
   refused as ``FIELD_DIVERGED`` and no field integrity score is given.

Second procedure (``compute_field_integrity_metrics``):

4. **Chain search.** Once the field is solved, find the island chain of each
   selected crossing (from step 1) with SPECTRE's own tools
   (``spectre.fixed_points``): one O-point and one X-point of the chain, and their
   Greene residues R_O and R_X.
5. **Field integrity score.** Each chain found gets assigned a field integrity score
   ``M = (4 / pi) N_fp |R_O R_X|^(1/4) / (m^2 |d iota / d psi_n|)`` using the
   unperturbed (VMEC) shear. The scores of each chain are summed.
   M is technically the full width of the islands as a fraction of the toroidal flux,
   and can be read as the island flux over the total flux (up to a factor of
   pi/2): it lies between 0 and 1, 0 meaning no island and 1 islands from the axis
   to the boundary.
6. **The record.** The code outputs the field integrity score, with what is needed
   to read it (the chains it sums, the Beltrami residual of the field solve, and
   the ``refusal_class``).
"""

from __future__ import annotations

import enum
import math
import os
import pathlib
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pydantic

from constellaration.mhd import vmec_utils
from constellaration.mhd.spectre_settings import SpectreSettings

# Constants of the score, not settings: each was measured on a sample of the dataset,
# and changing one changes what the score means.

RESIDUAL_DIVERGED = 5.0
"""Above this Beltrami residual a field that is no longer converging is taken as not
converged, and no field integrity score is given."""

SATURATION_RESIDUE = 0.5
"""|R_O| above which a chain is given the worst score, 1."""


class RefusalClass(str, enum.Enum):
    """Why the field integrity score is 0 or None. Every design gets a row."""

    NONE = "NONE"
    """Nothing was refused: the score is the sum over the chains found."""
    NO_RATIONAL = "NO_RATIONAL"
    """No rational n/m with m <= max_poloidal_order in the rotational transform."""
    SIGN_INDEFINITE = "SIGN_INDEFINITE"
    """The rotational transform passes through zero: both helicities, no score."""
    FIELD_DIVERGED = "FIELD_DIVERGED"
    """Beltrami residual above threshold and not falling: the field is not a
    solution."""
    SEARCH_INCOMPLETE = "SEARCH_INCOMPLETE"
    """No O/X pair of any selected chain was found. The islands are taken to be too
    thin to find: score 0."""


class ScreenedCrossing(pydantic.BaseModel):
    """One particular intersection of VMEC's rotational transform with a rational n/m.

    A monotonic profile crosses a rational once. A non-monotonic one can cross it
    several times and yields one ``ScreenedCrossing`` per crossing, each with its own
    position and shear. Produced by the screen from the rotational transform alone
    (step 1), before any field is solved. Each crossing yields one
    :class:`SearchedChain`, or none when the search comes up empty. Even though we
    can have more than one crossing per n/m rational, each
    crossing doesn't necessarily open up its own island chain as the chains of two
    really close crossings might "merge": they are then one chain, which lists both
    crossings. This is handled later, when the chains are scored
    (:func:`score_chains`).
    """

    model_config = pydantic.ConfigDict(use_attribute_docstrings=True)

    n: int
    """Numerator of the rational n/m; a multiple of the field periods."""
    m: int
    """Denominator of the rational n/m: the number of islands in the chain."""
    psi_n: float
    """Normalised toroidal flux at which the transform crosses n/m. It is equivalent
    to VMEC's radial coordinate s. It is 0 on the axis and 1 at the boundary.
    SPECTRE's own radial coordinate, also called s, is a different one and runs from
    -1 to 1."""
    shear: float
    """|d iota / d psi_n| at this crossing (the unperturbed shear, VMEC's profile)."""


class SearchedChain(pydantic.BaseModel):
    """One island chain found in the solved field.

    A chain usually sits on one :class:`ScreenedCrossing`. When the transform crosses
    a rational several times and the chains of neighbouring crossings have joined,
    they are one chain here, as they are in the field: it lists all the crossings it
    sits on. Its residues exist only once a field has been solved and searched.
    """

    model_config = pydantic.ConfigDict(use_attribute_docstrings=True)

    n: int
    """Numerator of the chain's rational n/m."""
    m: int
    """Denominator of the chain's rational n/m: its number of islands."""
    psi_n: float
    """Normalised toroidal flux at the centre of the chain's islands. For a chain on
    one crossing it is the ``psi_n`` of that crossing. The islands span from
    ``psi_n - score / 2`` to ``psi_n + score / 2``."""
    crossings: list[ScreenedCrossing]
    """The crossing(s) of VMEC's transform this chain sits on, inner first, each with
    its own position and shear: one usually, several for a chain that joins the
    chains of neighbouring crossings."""
    residue_o: float
    """Greene residue R = (2 - trace J) / 4 at the chain's O-point, J the tangent map
    of one return."""
    residue_x: float
    """Greene residue at the chain's X-point; negative for a hyperbolic point."""
    o_point: tuple[float, float]
    """One O-point of the chain on the plane zeta = 0, in SPECTRE's coordinates
    (s, theta) of the single volume, s running from -1 on the axis to 1."""
    x_point: tuple[float, float]
    """One X-point of the chain on the plane zeta = 0, in the same coordinates."""
    score: float
    """The full width of the chain's islands as a fraction of the toroidal flux,
    between 0 and 1. For a chain on one crossing it is
    M = (4 / pi) N_fp |R_O R_X|^(1/4) / (m^2 |d iota / d psi_n|). For a chain on
    several crossings it runs from the inner edge of the islands of the first
    crossing to the outer edge of those of the last."""

    @property
    def residue_ratio(self) -> float:
        """|R_X / R_O|: 1 for an ideal pendulum island."""
        if self.residue_o == 0.0:
            return math.inf
        return abs(self.residue_x / self.residue_o)


class SpectreOutput(pydantic.BaseModel):
    """What ``run_spectre`` returns: the solved field and what it was solved for."""

    model_config = pydantic.ConfigDict(
        use_attribute_docstrings=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    h5_file: bytes | None
    """The complete SPECTRE output file (HDF5) of the last rung of the toroidal
    ladder; None when the screen refused the design and nothing was solved."""
    crossings: list[ScreenedCrossing]
    """The crossings the field was solved to resolve, lowest order first; empty when
    the screen refused the design."""
    refusal_class: RefusalClass
    """``NO_RATIONAL`` or ``SIGN_INDEFINITE`` when the screen refused the design,
    ``FIELD_DIVERGED`` when the ladder did, otherwise ``NONE``."""
    beltrami_residuals: list[float]
    """Beltrami residual of each rung of the toroidal ladder that was solved."""


class FieldIntegrityMetrics(pydantic.BaseModel):
    """The metrics row for one design."""

    model_config = pydantic.ConfigDict(use_attribute_docstrings=True)

    field_integrity_score: float | None
    """The field integrity score M: the sum of the ``score`` of each of ``chains``,
    at most 1. It is the fraction of the toroidal flux
    the island chains span, so 0 means no island and 1 islands from the axis to the
    boundary. It is also 0 when the rotational transform doesn't cross a low-order
    rational (``NO_RATIONAL``) or the island search finds no chain
    (``SEARCH_INCOMPLETE``); None for every other refusal."""

    crossings: list[ScreenedCrossing]
    """Every crossing of a rational that was selected by the screen, lowest order
    first: the chains the search looked for."""

    chains: list[SearchedChain]
    """The island chains the search found, one entry per chain. A chain that spans
    several crossings of one rational is one entry."""

    beltrami_residual: float | None
    """Beltrami residual (largest volume-averaged error of curl B = mu B) of the field
    the chains were searched in; None when no field was solved."""

    refusal_class: RefusalClass
    """Why ``field_integrity_score`` is 0 or None; ``NONE`` when it was computed."""


# ------------------------------------------------------------------------------------
# step 1 -- the screen (pure Python)
# ------------------------------------------------------------------------------------


def _is_fundamental_harmonic(n: int, m: int, n_field_periods: int) -> bool:
    """Is (n, m) the harmonic that drives the rational n/m in an N_fp-periodic field?

    The field carries toroidal harmonics that are multiples of N_fp only. A rational
    p/q in lowest terms resonates with (k p, k q), k = N_fp / gcd(p, N_fp), and the
    chain has m = k q islands. ``(2, 4)`` is the fundamental at N_fp = 2 (iota = 1/2
    is a four-island chain there); ``(1, 2)`` and ``(4, 8)`` are not.

    The higher harmonics (2n, 2m), (3n, 3m)... resonate on the same surface and are
    deliberately not counted as fundamental: they are not other chains, they reshape
    this one. If the second harmonic dominates the first, the chain shows 2m islands
    instead of m, in two families whose O-points have different residues. It was seen
    on chains pressed against the boundary. The search for (n, m) finds such a chain,
    looks for both families and scores it with m and the largest R_O. For a pure
    two-harmonic field that over-states the width of the wider islands by a factor
    between sqrt(2) and 2, so the score errs on the safe side.
    """
    if n <= 0 or m <= 0 or n % n_field_periods != 0:
        return False
    g = math.gcd(n, m)
    p = n // g
    return g == n_field_periods // math.gcd(p, n_field_periods)


def _crossings_of(
    io: npt.NDArray[np.float64], s: npt.NDArray[np.float64], n: int, m: int
) -> list[ScreenedCrossing]:
    """Takes the VMEC transform profile and one rational n/m, and returns one
    ScreenedCrossing for each place where the profile passes through n/m, with its
    position and its shear. Empty when the profile never reaches n/m.
    """
    target = n / m
    side = np.sign(io - target)
    # +1 if profile above n/m, -1 if below, 0 if exactly on it
    for k in range(1, side.size):  # a grid point exactly on n/m keeps its previous side
        if side[k] == 0:
            side[k] = side[k - 1]
    crossings = np.flatnonzero(side[:-1] != side[1:])
    if crossings.size == 0:
        closest = int(np.argmin(np.abs(io - target)))
        if io[closest] == target:  # a tangency at a grid point
            crossings = np.array([closest])
        # otherwise the transform never reaches n/m: no crossing
    out = []
    for k in crossings.tolist():  # .tolist() converts a NumPy array to a Python list
        k0, k1 = max(k - 1, 0), min(k + 2, io.size - 1)  # Keep indices inside boundary
        shear = float((io[k1] - io[k0]) / (s[k1] - s[k0])) if s[k1] != s[k0] else 0.0
        k_next = min(k + 1, io.size - 1)  #  position is found by linear interpolation
        if io[k_next] != io[k]:
            fraction = float((target - io[k]) / (io[k_next] - io[k]))
            psi_n = float(s[k] + min(max(fraction, 0.0), 1.0) * (s[k_next] - s[k]))
        else:
            psi_n = float(s[k])
        out.append(
            ScreenedCrossing(
                n=n,
                m=m,
                psi_n=psi_n,
                shear=abs(shear),
            )
        )
    return out


def _find_crossings(
    iota: npt.ArrayLike,
    s_coordinate: npt.ArrayLike,
    n_field_periods: int,
    max_poloidal_order: int,
) -> list[ScreenedCrossing]:
    """Every crossing of a rational n/m, lowest order first.

    ``iota`` is the signed VMEC profile on the grid
    ``s_coordinate`` (VMEC's radial coordinate s, the normalised toroidal flux); the
    screen runs on |iota|, the sign being a coordinate convention. A rational crossed
    several times contributes one entry per crossing. Empty
    when the transform crosses no rational of order <= ``max_poloidal_order``.
    """
    # The iota profile: s is its x axis (where), io its y axis (the transform there).
    io = np.abs(np.asarray(iota, dtype=float))
    s = np.asarray(s_coordinate, dtype=float)
    hi = float(io.max())
    nfp = n_field_periods
    out: list[ScreenedCrossing] = []
    for m in range(1, max_poloidal_order + 1):
        for n in range(nfp, math.ceil(hi * m) + nfp, nfp):
            if _is_fundamental_harmonic(n, m, nfp):
                out.extend(_crossings_of(io, s, n, m))
    return out


def _select_crossings(
    candidates: Sequence[ScreenedCrossing],
    max_rationals: int | None = None,
) -> list[ScreenedCrossing]:
    """Keep every crossing of the ``max_rationals`` lowest-order rationals.

    It counts rationals, not crossings: a rational crossed twice keeps both of its
    crossings and counts once. All crossings are kept when ``max_rationals`` is None.
    """
    # The distinct rationals, lowest order first (candidates are already in order).
    rationals: list[tuple[int, int]] = []
    for crossing in candidates:
        if (crossing.n, crossing.m) not in rationals:
            rationals.append((crossing.n, crossing.m))
    kept = rationals[:max_rationals]
    return [c for c in candidates if (c.n, c.m) in kept]


def screen_rotational_transform(
    iota: npt.ArrayLike,
    s_coordinate: npt.ArrayLike,
    n_field_periods: int,
    settings: SpectreSettings,
) -> tuple[RefusalClass, list[ScreenedCrossing]]:
    """Step 1: read the VMEC rotational transform and name the crossings to solve for.

    This is the entry point of the screen. From the iota profile (``iota`` on the
    grid ``s_coordinate``) and the number of field periods it returns:

    - a refusal class: ``SIGN_INDEFINITE`` when the profile changes sign,
      ``NO_RATIONAL`` when it crosses no rational of order <=
      ``max_poloidal_order``, otherwise ``NONE``;
    - the crossings of the ``max_rationals`` lowest-order rationals
      (:func:`_find_crossings`, then :func:`_select_crossings`), empty when refused.

    These crossings set the resolution the field is solved at (step 2) and are the
    ones whose island chains SPECTRE's ``fixed_points`` then searches for, to get
    their Greene residues (step 4).
    """
    io = np.asarray(iota, dtype=float)
    if io.min() < 0.0 < io.max():
        return RefusalClass.SIGN_INDEFINITE, []
    candidates = _find_crossings(
        io, s_coordinate, n_field_periods, settings.max_poloidal_order
    )
    if not candidates:
        return RefusalClass.NO_RATIONAL, []
    return RefusalClass.NONE, _select_crossings(candidates, settings.max_rationals)


# ------------------------------------------------------------------------------------
# step 2 -- the resolution
# ------------------------------------------------------------------------------------


def choose_resolution(
    equilibrium: vmec_utils.VmecppWOut,
    crossings: Sequence[ScreenedCrossing],
    settings: SpectreSettings,
) -> tuple[int, int]:
    """Step 2: the poloidal and radial resolution the field is solved at.

    Returns ``(mpol, lrad)``. ``mpol`` follows the highest order m among the
    crossings (``settings.poloidal_modes``), and is never below the resolution of
    the VMEC boundary itself: the boundary is resampled into SPECTRE's basis, so a
    lower resolution would solve a different shape. ``lrad`` is
    ``settings.radial_resolution``, or ``mpol + 4`` when that is None.
    """
    highest_order = max(crossing.m for crossing in crossings)
    mpol = max(settings.poloidal_modes(highest_order), equilibrium.mpol - 1)
    lrad = settings.radial_resolution or mpol + 4
    return mpol, lrad


# ------------------------------------------------------------------------------------
# step 3 -- the toroidal ladder
# ------------------------------------------------------------------------------------


def _ladder(
    solve: Callable[[int], float], settings: SpectreSettings
) -> tuple[list[float], bool]:
    """Run the toroidal-resolution ladder on a solver ``solve(ntor) -> residual``.

    Climb while the residual is above ``settings.stop_residual``; refuse as diverged
    when it is above ``RESIDUAL_DIVERGED`` and did not fall from the previous rung (or
    there is no rung left) -- a first-rung residual above ``RESIDUAL_DIVERGED`` gets
    one more rung. Returns the residuals of the rungs solved and whether the field
    diverged.
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


def _build_spectre_input_parameters(
    equilibrium: vmec_utils.VmecppWOut,
    settings: SpectreSettings,
    poloidal_resolution: int,
    toroidal_resolution: int,
    radial_resolution: int,
) -> Any:
    """Builds the single-volume SPECTRE input of a VMEC equilibrium.

    This function does 3 things:

    1. It fills two empty arrays so the equilibrium can be handed to SPECTRE's
    converter (through simsopt).
    2. It calls the converter and sets the radial resolution.
    3. It copies VMEC's axis into SPECTRE's input, and pins it there (no axis
    reconstruction).
    """
    from spectre import converters

    # Step 1: the two current-density arrays are empty in equilibria loaded from the
    # dataset; zeros of the right shape are enough, the converter does not read them.
    unstored = {
        name: np.zeros_like(equilibrium.bsupumnc)
        for name in ("currumnc", "currvmnc")
        if np.size(getattr(equilibrium, name)) == 0
    }

    # Step 2
    vmec = vmec_utils.as_simsopt_vmec(equilibrium.model_copy(update=unstored))
    parameters = converters.simsvmec2spectre(
        vmec,
        nvol=settings.n_volumes,
        mpol=poloidal_resolution,
        ntor=toroidal_resolution,
        # run_vmec constrains the toroidal current, never the rotational transform
        profile_type="current",
    )
    physics = parameters.physics
    physics.lrad = np.full_like(physics.lrad, radial_resolution)

    # Step 3: the axis has m = 0 harmonics only, which the converter's change of
    # poloidal angle leaves unchanged, so it is copied without a sign change.
    def axis(coefficients: npt.ArrayLike) -> npt.NDArray[np.float64]:
        out = np.zeros_like(physics.rac)
        values = np.asarray(coefficients, dtype=float)[: toroidal_resolution + 1]
        out[: values.size] = values
        return out

    physics.rac = axis(equilibrium.raxis_cc)
    physics.zas = axis(equilibrium.zaxis_cs)
    if physics.rac[0] <= 0.0:
        raise RuntimeError(
            "SPECTRE discards a coordinate axis whose first harmonic is not positive,"
            f" and VMEC's magnetic axis maps to {physics.rac[0]}."
        )
    # Pin the axis to VMEC's, do not rebuild one of SPECTRE's own.
    parameters.numeric.lrzaxis = 0
    return parameters


def _solve_field(parameters: Any, directory: pathlib.Path, max_threads: int) -> float:
    """Solves the field of one SPECTRE input and returns its Beltrami residual.

    This function does 4 things:

    1. It writes the input parameters to ``field.toml`` in ``directory``.
    2. It deletes the ``field.h5`` of the previous solve, if any.
    3. It runs the solve in a separate Python process
       (``constellaration.mhd.spectre_runner``), which writes ``field.h5``.
    4. It reads the Beltrami residual back from ``field.h5``, and raises an error if
       it is not there.
    """
    from spectre.file_io import write_input_parameters_to_toml

    # Step 1
    write_input_parameters_to_toml(parameters, directory / "field.toml")

    # Step 2: the toroidal ladder reuses the same directory for every rung, so a
    # solve that fails must not leave the previous rung's file to be read instead.
    (directory / "field.h5").unlink(missing_ok=True)

    # Step 3: a separate process, because SPECTRE keeps its state in Fortran global
    # variables, which cannot be reset to solve again at another resolution.
    threads = str(max_threads)
    process = subprocess.run(
        [sys.executable, "-m", "constellaration.mhd.spectre_runner", "field.toml"],
        cwd=directory,
        env={**os.environ, "OMP_NUM_THREADS": threads, "OPENBLAS_NUM_THREADS": threads},
        capture_output=True,
        text=True,
    )

    # Step 4: SPECTRE can stop inside Fortran and still exit with status 0, so the
    # exit status is not trusted. The runner writes the residual last: finding it in
    # the file is what says the solve completed.
    try:
        return _beltrami_residual(directory / "field.h5")
    except (OSError, KeyError) as error:
        raise RuntimeError(
            f"The SPECTRE solve did not complete (exit status {process.returncode})."
            f" Its output ended with:\n{(process.stdout + process.stderr)[-2000:]}"
        ) from error


def _beltrami_residual(h5_path: pathlib.Path) -> float:
    """Largest volume-averaged Beltrami error stored in a SPECTRE output file."""
    import h5py

    with h5py.File(h5_path, "r") as h5:
        return float(np.max(np.abs(np.asarray(h5["errors/beltrami_avg"]))))


def solve_toroidal_ladder(
    equilibrium: vmec_utils.VmecppWOut,
    settings: SpectreSettings,
    mpol: int,
    lrad: int,
) -> tuple[bytes, list[float], bool]:
    """Step 3: solve the field, raising the toroidal resolution until it converges.

    Solves at each ``ntor`` of ``settings.toroidal_ladder`` in turn (:func:`_ladder`
    decides when to stop). Returns the SPECTRE output file (HDF5) of the last rung
    solved, the Beltrami residual of every rung, and whether the field diverged.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        directory = pathlib.Path(tmpdir)

        def solve(ntor: int) -> float:
            parameters = _build_spectre_input_parameters(
                equilibrium, settings, mpol, ntor, lrad
            )
            return _solve_field(parameters, directory, settings.max_threads)

        residuals, diverged = _ladder(solve, settings)
        h5_file = (directory / "field.h5").read_bytes()
    return h5_file, residuals, diverged


# ------------------------------------------------------------------------------------
# first procedure -- steps 1 to 3
# ------------------------------------------------------------------------------------


def run_spectre(
    equilibrium: vmec_utils.VmecppWOut, settings: SpectreSettings
) -> SpectreOutput:
    """First of the two procedures: output = run_spectre(equilibrium, settings).

    Screens the rotational transform for crossings, computes the single-volume vacuum
    field using SPECTRE until the Beltrami residual meets the tolerance, and finally
    stores a SpectreOutput instance.
    """
    # Step 1: the screen. The iota profile: s_coordinate is its x axis, iota its y
    # axis.
    iota = np.asarray(equilibrium.iotaf, dtype=float)
    s_coordinate = np.asarray(equilibrium.normalized_toroidal_flux_full_grid_mesh)
    refusal_class, crossings = screen_rotational_transform(
        iota, s_coordinate, equilibrium.n_field_periods, settings
    )
    if not crossings:
        return SpectreOutput(
            h5_file=None,
            crossings=[],
            refusal_class=refusal_class,
            beltrami_residuals=[],
        )

    # Step 2: the resolution.
    mpol, lrad = choose_resolution(equilibrium, crossings, settings)

    # Step 3: the toroidal ladder.
    h5_file, residuals, diverged = solve_toroidal_ladder(
        equilibrium, settings, mpol, lrad
    )
    return SpectreOutput(
        h5_file=h5_file,
        crossings=crossings,
        refusal_class=(RefusalClass.FIELD_DIVERGED if diverged else RefusalClass.NONE),
        beltrami_residuals=residuals,
    )


# ------------------------------------------------------------------------------------
# step 4 -- the chain search
# ------------------------------------------------------------------------------------


# Empirical constants of the fixed-point search that SPECTRE does not define. Lengths
# are in metres. The root tolerance and the duplicate distance are SPECTRE's own
# (``spectre.fixed_points.tracker``).

SCAN_RADIAL_POINTS = 31
"""Samples of the return-map scan along one radial line of the search window."""

ANGLE_HALVINGS = 4
"""How many times the search halves its angle, from pi / m, before it gives up."""

TRANSFORM_TOLERANCE = 0.01
"""Largest relative mismatch between a fixed point's transform and n/m."""

TANGENT_MAP_TOLERANCE = 1e-9
"""Largest |det J - 1| of a tangent map whose Greene residue is trusted; the return
map is area-preserving, so a larger value means the map is not converged."""

WINDOW_FLUX = 0.05
"""In SPECTRE, the chain is looked for between psi_n - 0.05 and psi_n + 0.05."""


def _search_window(
    field: Any, equilibrium: vmec_utils.VmecppWOut, psi_n: float
) -> tuple[float, float]:
    """Returns the radial window (SPECTRE's s coordinate) in which the chain of one
    crossing is searched.

    A chain lives on the flux surface where the transform crosses its rational, and
    VMEC gives that surface as ``psi_n``. SPECTRE's radial coordinate s is not a flux
    label: on the plane zeta = 0, where the search runs, one flux surface spans a
    range of s around the plasma. So the window is the range of s covered there by
    the VMEC flux surfaces from ``psi_n - WINDOW_FLUX`` to ``psi_n + WINDOW_FLUX``:
    the surfaces are drawn in (R, Z) and mapped to SPECTRE's coordinates
    (``get_st_coords``).
    """
    theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    angle = np.asarray(equilibrium.xm)[:, None] * theta[None, :]
    flux_grid = np.asarray(equilibrium.normalized_toroidal_flux_full_grid_mesh)

    def s_along_surface(surface_psi_n: float) -> list[float]:
        """SPECTRE's s along one VMEC flux surface, on the plane zeta = 0."""
        r, z = (
            np.array(
                [
                    np.interp(surface_psi_n, flux_grid, mode)
                    for mode in np.asarray(modes)
                ]
            )
            @ trig(angle)
            for modes, trig in ((equilibrium.rmnc, np.cos), (equilibrium.zmns, np.sin))
        )
        return [
            float(np.ravel(field.get_st_coords(0, float(ri), float(zi), 0.0)[0])[0])
            for ri, zi in zip(r, z)
        ]

    # The boundary is s = 1 by construction, and so is the axis at s = -1: SPECTRE's
    # coordinate axis is pinned to VMEC's magnetic axis, and the two codes solve the
    # same vacuum field, so the magnetic axis of the solved field sits there too
    # (within 0.3 % of the minor radius on the fields checked). The window stops
    # 1 % of the minor radius short of it, at s = -0.98, so that the axis, which is
    # a fixed point of every return map, is never inside a window.
    inner, outer = psi_n - WINDOW_FLUX, psi_n + WINDOW_FLUX
    lo = min(s_along_surface(inner)) if inner > 0.0 else -0.98
    hi = max(s_along_surface(outer)) if outer < 1.0 else 0.999
    return max(lo, -0.98), min(hi, 0.999)


def _fixed_point(
    field: Any,
    seed: tuple[float, float],
    n: int,
    m: int,
    window: tuple[float, float],
) -> dict[str, Any] | None:
    """Is this seed a fixed point of the n/m chain? Returns it classified, or None.

    The seed (s, theta) is polished with SPECTRE's root-finder, and the point it
    converges to is kept if it passes three checks:

    1. It is not further in than the inner edge of the search window ``window``.
       The chain is looked for in the window; what lies further in is something
       else, the magnetic axis first of all (a fixed point of every return map).
    2. It winds n/m
    3. Its tangent map has converged.

    Returns ``{kind, residue, s, theta, R, Z}``: ``kind`` is "o" or "x", ``(s, theta)``
    is the position of the point in SPECTRE's coordinates on the plane zeta = 0, and
    ``(R, Z)`` the same position in metres. It does not know which points were found
    before: the caller sorts out the duplicates.
    """
    from spectre import fixed_points
    from spectre.fixed_points import tracker

    n_transits = m + 1  # the trace records the starting plane as well
    point = fixed_points.root_fixed_point(  # Is the seed a fixed point?
        field,
        0,  # Volume 0 (i.e. innermost)
        n_transits,
        seed,
        method="RK45",
        rtol=tracker.ROOT_RTOL,
        atol=tracker.ROOT_RTOL,
        maxfev=tracker.ROOT_MAXFEV,
        ftol=tracker.ROOT_FTOL,
    )
    if not point.converged:
        return None  # Seed is not a fixed point

    # Check 1: is it further inside than the search window?
    if point.s < window[0]:
        return None

    # Check 2: does it wind n/m?
    _, _, transform, ok = field.trace_one_fieldline(
        [point.s, point.theta], 0, num_phi_planes=1, num_transits=n_transits
    )
    if not ok or abs(abs(float(transform)) - n / m) > TRANSFORM_TOLERANCE * n / m:
        return None

    # Check 3: is the determinant of its tangent map 1?
    classifier = fixed_points.FixedPointClassifier(
        field, 0, m, method="DOP853", rtol=1e-12, atol=1e-12
    )
    classified: dict[str, Any] | None = fixed_points.classify_point(
        classifier, (point.s, point.theta)
    )
    if classified is None or abs(classified["det"] - 1.0) >= TANGENT_MAP_TOLERANCE:
        return None
    # A positive residue is an O-point (above 1, one that has turned unstable); a
    # negative residue is an X-point.
    return dict(
        kind="o" if classified["residue"] > 0 else "x",
        residue=float(classified["residue"]),
        s=float(classified["s"]),
        theta=float(classified["theta"]),
        R=float(classified["R"]),
        Z=float(classified["Z"]),
    )


def _find_chain(
    field: Any,
    n: int,
    m: int,
    window: tuple[float, float],
) -> dict[str, Any] | None:
    """Looks, in one radial window, for the island chain of one rational n/m.
    It returns the chain's two Greene residues with one O-point and one X-point,
    or nothing if it cannot find both kinds.

    Stellarator symmetry puts a fixed point of every chain, an O-point or an
    X-point, on the radial lines theta = 0 and theta = pi of the plane zeta = 0. A
    radial line is all the points at one poloidal angle theta, from the inner edge
    of the window to its outer edge. So the search goes radial line by radial line:

    1. It scans the return map of ``m`` field periods along theta = 0, then along
       theta = pi.
    2. While it does not hold both an O-point and an X-point, it scans one more
       radial line, at half the angle each time: pi / m, pi / (2 m), pi / (4 m)... The
       points of a chain alternate around the plasma, so the missing kind sits
       between the ones already found. A chain with twice its m islands, in two
       families whose O-points have different residues, shows its second family on
       the way.

    Returns ``{residue_o, residue_x, o_point, x_point}``:

    - ``residue_o`` and ``residue_x`` are the largest residue of each kind;
    - ``o_point`` and ``x_point`` are the positions of the points that have them, each
      as ``(s, theta)`` in SPECTRE's coordinates on the plane zeta = 0.
    """
    from spectre import fixed_points
    from spectre.fixed_points import tracker

    # The points of the chain found so far, by kind.
    found: dict[str, list[dict[str, Any]]] = {"o": [], "x": []}

    # Step 1, then step 2
    # Lines θ = 0 and θ = π first, then π/m, π/2m, π/4m, π/8m
    angles = [0.0, np.pi] + [np.pi / (m * 2**k) for k in range(ANGLE_HALVINGS)]
    for theta in angles:
        if found["o"] and found["x"]:
            break
        # Sample evenly spaced points along the radial line at this angle (i.e. scan).
        # Those where the field line comes back closest to its start are used as
        # seeds for the root-finder, which polishes them into fixed points.
        candidates, minimum_at_the_end = fixed_points.scan_fixed_points(
            field,
            0,
            m + 1,
            theta_grid=[theta],
            s_range=window,
            n_s=SCAN_RADIAL_POINTS,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
        )
        seeds = [(candidate["s"], theta) for candidate in candidates]
        # The scan only reports a point as a seed when its neighbours on both sides
        # come back further from their start than it does. The last point of the
        # line has no neighbour beyond it, so it is never reported, even when it is
        # the best of the line. That happens when an island touches the boundary:
        # its X-points then sit on the boundary itself. The scan says so separately
        # ("outer"), and the last point of the line is then added as a seed.
        if "outer" in minimum_at_the_end.values():
            seeds.append((window[1], theta))
        # Keep the points of the chain these seeds lead to, unless already held.
        for seed in seeds:
            point = _fixed_point(field, seed, n, m, window)
            if point is None:
                continue
            # The point is new if it is further than DEDUP_TOL from every point held.
            is_new = all(
                np.hypot(point["R"] - held["R"], point["Z"] - held["Z"])
                > tracker.DEDUP_TOL
                for held in found["o"] + found["x"]
            )
            if is_new:
                found[point["kind"]].append(point)

    if not (found["o"] and found["x"]):
        return None
    o_point, x_point = (
        # Select the highest residue of each kind
        max(found[kind], key=lambda point: abs(point["residue"]))
        for kind in ("o", "x")
    )
    return dict(
        residue_o=o_point["residue"],
        residue_x=x_point["residue"],
        o_point=(o_point["s"], o_point["theta"]),
        x_point=(x_point["s"], x_point["theta"]),
    )


def search_chains(
    output: SpectreOutput, equilibrium: vmec_utils.VmecppWOut
) -> list[dict[str, Any] | None]:
    """Step 4: find the island chain of every crossing.

    For each crossing of ``output.crossings``, search for one O-point and one X-point
    of its chain. The search is focused in a radial window around the position VMEC
    gives the crossing (:func:`_search_window`). It leverages SPECTRE's
    ``fixed_points`` (:func:`_find_chain`).

    Returns a list with exactly one entry per crossing, in the order of
    ``output.crossings``: entry i belongs to crossing i. An entry is the chain found
    at that crossing, as ``{residue_o, residue_x, o_point, x_point}``, where
    ``o_point`` and ``x_point`` are positions ``(s, theta)`` in SPECTRE's coordinates
    on the plane zeta = 0. Where no chain was found the entry is None: it holds the
    place of that crossing, so that the list stays lined up with the crossings.

    For example, for two crossings, a chain found at the first and none at the
    second:

        [{"residue_o": 0.00172, "residue_x": -0.00180,
          "o_point": (0.066, 0.0), "x_point": (0.234, 1.105)},
         None]
    """
    field = load_field(output)  # SPECTRE's own SPECTREout object
    chains: list[dict[str, Any] | None] = []
    for crossing in output.crossings:
        window = _search_window(field, equilibrium, crossing.psi_n)
        chains.append(_find_chain(field, crossing.n, crossing.m, window))
    return chains


# ------------------------------------------------------------------------------------
# step 5 -- the scores
# ------------------------------------------------------------------------------------


def _chain_score(
    m: int, residue_o: float, residue_x: float, shear: float, n_fp: int
) -> float:
    """The field integrity score of one chain, between 0 and 1."""
    # An O-point on the way to period doubling (R = 1) is no longer the centre of a
    # pendulum island, and the formula under-states what such a chain destroys.
    if abs(residue_o) > SATURATION_RESIDUE:
        return 1.0
    # Where the transform only grazes the rational the shear vanishes and the formula
    # diverges: such a crossing gets the worst score too.
    if shear == 0.0:
        return 1.0
    # M of a chain
    amplitude = n_fp * abs(residue_o * residue_x) ** 0.25 / (m * m * abs(shear))
    return min(1.0, (4.0 / math.pi) * amplitude)


def _reconnection_parameter(
    inner: ScreenedCrossing,
    outer: ScreenedCrossing,
    score_inner: float,
    score_outer: float,
) -> float:
    """Computes if the chains of two neighbouring crossings of one rational have
    merged and should count as one chain.

    A rotational transform that is not monotonic can cross a rational twice, and each
    crossing has its own chain. If Lambda<1 the two chains are separate; if Lambda>=1
    their separatrices join and the two become one structure. With M the scores, s
    the shears at the two crossings and L their distance in normalised flux,

        Lambda = (3 / 4) (s_1 M_1^2 + s_2 M_2^2) / ((s_1 + s_2) L^2),

    the rotational transform between the crossings being taken as the cubic through
    them with those shears.
    """
    distance = abs(outer.psi_n - inner.psi_n)
    shears = inner.shear + outer.shear
    if distance == 0.0 or shears == 0.0:
        return math.inf
    weighted = inner.shear * score_inner**2 + outer.shear * score_outer**2
    return 0.75 * weighted / (shears * distance**2)


def score_chains(
    crossings: Sequence[ScreenedCrossing],
    chains_found: Sequence[dict[str, Any] | None],
    n_field_periods: int,
) -> tuple[list[SearchedChain], float]:
    """Step 5: give each chain its field integrity score, and the design its own.

    ``chains_found`` is what :func:`search_chains` returns: for each crossing, the
    chain found there or None. This function does 3 things:

    1. It scores the chain of each crossing with the shear of that crossing
       (:func:`_chain_score`).
    2. It groups the crossings whose chains are one and the same: neighbouring
       crossings of one rational whose islands have joined
       (:func:`_reconnection_parameter` >= 1).
    3. It writes one :class:`SearchedChain` per group. For a group of several
       crossings the islands run from the inner edge of the first to the outer edge
       of the last, and the residues are the largest of the group.

    Returns the chains and the score of the design, which is the sum of their scores.
    Islands cannot span more than the whole flux, hence the caps at 1.
    """
    # Step 1
    scored = [
        (
            crossing,
            chain,
            _chain_score(
                crossing.m,
                chain["residue_o"],
                chain["residue_x"],
                crossing.shear,
                n_field_periods,
            ),
        )
        for crossing, chain in zip(crossings, chains_found)
        if chain is not None
    ]
    scored.sort(key=lambda entry: (entry[0].m, entry[0].n, entry[0].psi_n))

    # Step 2
    groups: list[list[tuple[ScreenedCrossing, dict[str, Any], float]]] = []
    for crossing, chain, score in scored:
        if groups:
            previous, _, previous_score = groups[-1][-1]
            same_rational = (previous.n, previous.m) == (crossing.n, crossing.m)
            reconnection = _reconnection_parameter(
                previous, crossing, previous_score, score
            )
            if same_rational and reconnection >= 1.0:
                groups[-1].append((crossing, chain, score))
                continue
        groups.append([(crossing, chain, score)])

    # Step 3
    chains = []
    for group in groups:
        (first, _, first_score), (last, _, last_score) = group[0], group[-1]
        # For a single crossing this is its own position and its own score.
        centre = (first.psi_n + last.psi_n) / 2.0 + (last_score - first_score) / 4.0
        width = last.psi_n - first.psi_n + (first_score + last_score) / 2.0
        with_o = max(group, key=lambda entry: abs(entry[1]["residue_o"]))[1]
        with_x = max(group, key=lambda entry: abs(entry[1]["residue_x"]))[1]
        chains.append(
            SearchedChain(
                n=first.n,
                m=first.m,
                psi_n=centre,
                crossings=[crossing for crossing, _, _ in group],
                residue_o=with_o["residue_o"],
                residue_x=with_x["residue_x"],
                o_point=with_o["o_point"],
                x_point=with_x["x_point"],
                score=min(1.0, width),
            )
        )
    return chains, min(1.0, sum(chain.score for chain in chains))


# ------------------------------------------------------------------------------------
# step 6 -- the record
# ------------------------------------------------------------------------------------


def build_metrics(
    *,
    refusal_class: RefusalClass,
    crossings: Sequence[ScreenedCrossing],
    chains: Sequence[SearchedChain],
    score: float | None,
    beltrami_residual: float | None,
) -> FieldIntegrityMetrics:
    """Step 6: write the record of one design.

    ``score`` is the score of the design from step 5. It is replaced in two cases:
    a design that crosses no rational has no island, so its score is 0; a design
    refused for any other reason has no score. And when the field was searched but
    no chain was found, the record says so (``SEARCH_INCOMPLETE``): a chain the
    search cannot find is taken to be too thin to matter, and the score is 0.
    """
    if refusal_class == RefusalClass.NO_RATIONAL:
        score = 0.0
    elif refusal_class in (RefusalClass.SIGN_INDEFINITE, RefusalClass.FIELD_DIVERGED):
        score = None
    elif not chains:
        refusal_class = RefusalClass.SEARCH_INCOMPLETE
        score = 0.0
    return FieldIntegrityMetrics(
        field_integrity_score=score,
        crossings=list(crossings),
        chains=list(chains),
        beltrami_residual=beltrami_residual,
        refusal_class=refusal_class,
    )


# ------------------------------------------------------------------------------------
# second procedure -- steps 4 to 6
# ------------------------------------------------------------------------------------


def compute_field_integrity_metrics(
    output: SpectreOutput, equilibrium: vmec_utils.VmecppWOut
) -> FieldIntegrityMetrics:
    """Second of the two procedures: metrics = compute_field_integrity_metrics(output,
    equilibrium).

    Searches the solved field for the island chain of every crossing it was solved
    for, scores each chain, and stores the result in a FieldIntegrityMetrics
    instance. The equilibrium supplies what the field does not carry: the flux
    surfaces around each crossing, where the chains are looked for.
    """
    residuals = output.beltrami_residuals
    beltrami_residual = residuals[-1] if residuals else None

    # A design refused in the first procedure has no field to search.
    if output.refusal_class in (
        RefusalClass.NO_RATIONAL,
        RefusalClass.SIGN_INDEFINITE,
        RefusalClass.FIELD_DIVERGED,
    ):
        return build_metrics(
            refusal_class=output.refusal_class,
            crossings=output.crossings,
            chains=[],
            score=None,
            beltrami_residual=beltrami_residual,
        )

    # Step 4: the chain search.
    chains_found = search_chains(output, equilibrium)

    # Step 5: the scores.
    chains, score = score_chains(
        output.crossings, chains_found, equilibrium.n_field_periods
    )

    # Step 6: the record.
    return build_metrics(
        refusal_class=RefusalClass.NONE,
        crossings=output.crossings,
        chains=chains,
        score=score,
        beltrami_residual=beltrami_residual,
    )


# ------------------------------------------------------------------------------------
# reading a record
# ------------------------------------------------------------------------------------


def load_field(output: SpectreOutput) -> Any:
    """The solved field as SPECTRE's own ``SPECTREout``, to trace field lines in."""
    import spectre

    if output.h5_file is None:
        raise ValueError("This design was refused at the screen: no field was solved.")
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = pathlib.Path(tmpdir) / "field.h5"
        h5_path.write_bytes(output.h5_file)
        return spectre.SPECTREout(str(h5_path))


def predicted_flux_fraction(field_integrity_score: float) -> float:
    """Toroidal flux inside the islands, as a fraction of the total: (2 / pi) M.

    The score is the width of the band the islands span. An island is lens-shaped
    and fills 2 / pi of that band.
    """
    return (2.0 / math.pi) * field_integrity_score


def island_width(chain: SearchedChain, minor_radius: float) -> float:
    """Radial width of a chain's islands, in metres.

    The score of a chain is its width in normalised toroidal flux, centred on the
    crossing; the effective radius is ``minor_radius * sqrt(psi_n)``.
    """
    half = chain.score / 2.0
    inner = max(chain.psi_n - half, 0.0)
    outer = min(chain.psi_n + half, 1.0)
    return minor_radius * (math.sqrt(outer) - math.sqrt(inner))

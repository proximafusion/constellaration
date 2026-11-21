import base64
import io
import json
from contextlib import contextmanager
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import vmecpp
from vmecpp import autodiff_wout

from constellaration.geometry import surface_utils
from constellaration.mhd import vmec_settings, vmec_utils
from constellaration.utils import file_exporter
from tests.equilibrium_reference_test import NAMES, REFERENCES, reference_inputs


def test_read_old_wout_json():
    with Path("tests/utils/huggingface_dataset_wout.json").open("r") as f:
        vmec_utils.VmecppWOut.model_validate_json(f.read())


def _serialize_np_as_bytes(np_array: np.ndarray) -> bytes:
    """Serializes a numpy array to raw bytes."""
    with io.BytesIO() as out_f:
        np.save(out_f, np_array)
        np_bytes = out_f.getvalue()
    return np_bytes


def _encode_field_as_blob(value: list | np.ndarray) -> dict:
    """Encode an array as a dapper blob."""
    arr = np.array(value) if not isinstance(value, np.ndarray) else value
    np_bytes = _serialize_np_as_bytes(arr)
    encoded_content = base64.b64encode(np_bytes).decode("ascii")
    return {
        "dapper_is_blob": True,
        "content": encoded_content,
        "file_suffix": ".npy",
        "content_length": len(np_bytes),
    }


def test_read_binary_encoded_wout_json():
    with Path("tests/utils/huggingface_dataset_wout.json").open("r") as f:
        wout_original = vmec_utils.VmecppWOut.model_validate_json(f.read())

    # Create a blob-encoded version by converting some fields to blob format
    data = json.loads(Path("tests/utils/huggingface_dataset_wout.json").read_text())

    # Encode a few array fields as blobs (mix of small and medium sized arrays)
    fields_to_encode = ["am", "ac", "mass"]
    for field in fields_to_encode:
        if field in data and isinstance(data[field], list):
            data[field] = _encode_field_as_blob(data[field])

    # Validate the blob-encoded data
    wout_blob = vmec_utils.VmecppWOut.model_validate(data)

    # Verify that blob-decoded fields match the original values
    np.testing.assert_array_equal(wout_blob.am, wout_original.am)
    np.testing.assert_allclose(wout_blob.ac, wout_original.ac, rtol=1e-10)
    np.testing.assert_allclose(wout_blob.mass, wout_original.mass, rtol=1e-10)

    # Verify other properties remain the same
    assert wout_blob.nfp == wout_original.nfp
    assert wout_blob.ns == wout_original.ns
    assert wout_blob.mpol == wout_original.mpol
    assert wout_blob.ntor == wout_original.ntor


def test_write_old_wout_json_to_vmec2000_wout_file(tmp_path: Path) -> None:
    with Path("tests/utils/huggingface_dataset_wout.json").open("r") as f:
        wout = vmec_utils.VmecppWOut.model_validate_json(f.read())
    wout_path = tmp_path / "wout.nc"

    file_exporter.to_vmec2000_wout_file(wout, wout_path)

    read_back = vmec_utils.VmecppWOut.from_wout_file(wout_path)
    np.testing.assert_array_equal(read_back.rmnc, wout.rmnc)
    np.testing.assert_array_equal(read_back.zmns, wout.zmns)


def finite_beta_input():
    boundary, mhd, _ = reference_inputs("finite_beta")
    settings = vmec_settings.VmecSettings(
        n_poloidal_modes=4,
        max_toroidal_mode=3,
        n_poloidal_grid_points=20,
        n_toroidal_grid_points=18,
        multigrid_steps=[
            vmec_settings.VmecMultiGridStep(
                n_flux_surfaces=ns, force_tolerance=1e-19, n_max_iterations=18000
            )
            for ns in (15, 29)
        ],
        time_step=0.5,
    )
    return vmec_utils.build_vmecpp_indata(mhd, boundary, settings)


@pytest.mark.parametrize("name", NAMES)
def test_solved_equilibrium_against_pinned_wout(name):
    boundary, mhd, settings = reference_inputs(name)
    resolution = vmec_settings.create_vmec_settings_from_preset(
        boundary, settings.vmec_preset_settings
    )
    template = vmec_utils.build_vmecpp_indata(mhd, boundary, resolution)
    legacy = vmecpp.VmecWOut.from_wout_file(REFERENCES / name / "wout.nc")
    with vmecpp.enable_x64(True):
        solve = vmec_utils.make_equilibrium_solver(template)

        @jax.jit
        def evaluate(b):
            out = solve(
                b, jnp.asarray(mhd.pressure.coefficients), jnp.asarray(template.phiedge)
            )
            return (
                out["aspect"],
                out["phi"],
                out["betatotal"],
                out["bmnc"][0, 1:],
                out["iotaf"],
            )

        aspect, flux, beta, b00, iota = evaluate(
            jnp.stack((template.rbc, template.zbs))
        )
    np.testing.assert_allclose(aspect, legacy.aspect, rtol=1e-10)
    np.testing.assert_allclose(flux, legacy.phi, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(beta, legacy.betatotal, rtol=3e-5)
    np.testing.assert_allclose(b00, legacy.bmnc[0, 1:], rtol=3e-4)
    # The old VMEC++ version ran with a free m=1 gauge. At finite resolution
    # pinned-gauge iota differs slightly, most visibly at the finite-beta edge.
    np.testing.assert_allclose(iota, legacy.iotaf, rtol=5e-3)


def test_unconverged_equilibrium_is_not_a_penalty():
    template = finite_beta_input().model_copy(
        update={
            "ns_array": np.array([15]),
            "niter_array": np.array([1]),
            "ftol_array": np.array([1e-19]),
        }
    )
    with vmecpp.enable_x64(True):
        solve = vmec_utils.make_equilibrium_solver(template)
        boundary = jnp.stack((template.rbc, template.zbs))
        with pytest.raises(RuntimeError, match="FATAL ERROR in SolveEquilibrium"):
            jax.jit(solve)(
                boundary, jnp.array([3000.0, -3000.0]), jnp.asarray(template.phiedge)
            )["aspect"].block_until_ready()


def test_equilibrium_gradients():
    # Vary a boundary mode, pressure amplitude/shape, and the physical edge flux.
    template = finite_beta_input()
    boundary = np.stack([template.rbc, template.zbs])
    pressure = np.array([3000.0, -3450.0, 450.0])

    def parameters(t):
        b = jnp.asarray(boundary).at[0, 1, template.ntor].add(0.06 * t)
        p = pressure + t * np.array([300.0, -200.0, -100.0])
        return b, p, template.phiedge * (1 + 0.1 * t)

    keys = ("aspect", "betatotal", "betaxis", "edge_iota")

    def sample(wout):
        return jnp.stack(
            [wout[k] if k != "edge_iota" else wout["iotaf"][-1] for k in keys]
        )

    with vmecpp.enable_x64(True):
        solve = vmec_utils.make_equilibrium_solver(template)

        def traced(t):
            return sample(solve(*parameters(t)))

        def independently_resolved(t):
            b, p, flux = parameters(t)
            inp = template.model_copy(
                update={
                    "rbc": np.asarray(b[0]),
                    "zbs": np.asarray(b[1]),
                    "am": np.asarray(p),
                    "pres_scale": 1.0,
                    "phiedge": float(flux),
                    "always_fix_m1_gauge": True,
                }
            )
            out = vmecpp.run(inp, max_threads=1, verbose=False).wout
            return sample(
                {**{k: getattr(out, k) for k in keys[:-1]}, "iotaf": out.iotaf}
            )

        value = jax.jit(traced)(0.0)
        np.testing.assert_allclose(traced(0.0), value, rtol=2e-8, atol=1e-10)
        expected = independently_resolved(0.0)
        np.testing.assert_allclose(value, expected, rtol=2e-4, atol=2e-7)
        derivative = jax.jit(jax.jacrev(traced))(0.0)
        assert np.all(np.isfinite(derivative))
        for step in (5e-4, 2.5e-4):
            finite_difference = (
                independently_resolved(step) - independently_resolved(-step)
            ) / (2 * step)
            np.testing.assert_allclose(
                derivative, finite_difference, rtol=3e-5, atol=2e-7
            )


def grid(mpol, ntor, nfp):
    (
        ntheta,
        nphi,
    ) = surface_utils.n_poloidal_toroidal_points_to_satisfy_nyquist_criterion(
        mpol, ntor
    )
    return jnp.moveaxis(
        jnp.asarray(
            surface_utils.make_theta_phi_grid(
                ntheta, nphi, np.pi / nfp, include_endpoints=True
            )
        ),
        -1,
        0,
    )


def pinned_case(name):
    root = REFERENCES / name
    boundary, _, settings = reference_inputs(name)
    loaded = vmecpp.VmecWOut.from_wout_file(root / "wout.nc")
    wout = dict(loaded)
    arrays = {
        k: jnp.asarray(v) for k, v in wout.items() if isinstance(v, (np.ndarray, float))
    }
    return boundary, settings, wout, arrays


@contextmanager
def resolved_case():
    """Yield one finite-beta family with pressure and boundary perturbations."""
    with vmecpp.enable_x64(True):
        template = finite_beta_input()
        solve = vmec_utils.make_equilibrium_solver(template)
        modes = {
            k: jnp.asarray(v)
            for k, v in autodiff_wout.static_fields(template).items()
            if k in ("xm", "xn", "xm_nyq", "xn_nyq")
        }
        b = jnp.stack([template.rbc, template.zbs])
        p = jnp.array([3000.0, -3450.0, 450.0])

        def family(t):
            b_t = b.at[0, 1, template.ntor].add(0.06 * t)
            return b_t, solve(b_t, p * (1 + 0.2 * t), template.phiedge * (1 + 0.1 * t))

        yield template, family, modes


def assert_directional_derivative(
    function,
    *,
    steps=(2e-3, 1e-3),
    atol=2e-7,
    rtol=1e-3,
):
    """Test local autodiff against two independent central re-solves."""
    f = jax.jit(function)
    result = f(0.0)
    derivative = jax.jit(jax.jacrev(function))(0.0)
    assert np.all(np.isfinite(result))
    assert np.all(np.isfinite(derivative))
    for step in steps:
        fd = (f(step) - f(-step)) / (2 * step)
        np.testing.assert_allclose(derivative, fd, rtol=rtol, atol=atol)

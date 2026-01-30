import base64
import io
import json
from pathlib import Path

import numpy as np

from constellaration.mhd import vmec_utils


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

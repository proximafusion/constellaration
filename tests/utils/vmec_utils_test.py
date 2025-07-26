from pathlib import Path

from constellaration.mhd import vmec_utils


def test_read_old_wout_json():
    with Path("tests/utils/huggingface_dataset_wout.json").open("r") as f:
        vmec_utils.VmecppWOut.model_validate_json(f.read())

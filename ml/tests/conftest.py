import sys
from pathlib import Path

import pytest

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from src.data_loader import PTBXL, load_config  # noqa: E402


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def dataset(config):
    if not (Path(config["data"]["data_root"]) / "ptbxl_database.csv").exists():
        pytest.skip("PTB-XL not found at data_root")
    return PTBXL(config, sampling_rate=100)

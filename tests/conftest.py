from pathlib import Path

import pytest
import torch

from src.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return load_config(ROOT / "configs" / "tiny_cpu.yaml")


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)

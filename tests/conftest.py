"""
Pytest configuration and fixtures for OFFM tests.

This file is automatically loaded by pytest.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def device():
    """Return available device (cuda if available, else cpu)."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def random_trajectory():
    """Generate a random trajectory for testing."""
    def _make_trajectory(batch_size=5, n_t=20, n_x=64):
        return torch.randn(batch_size, n_t, n_x)
    return _make_trajectory


@pytest.fixture
def random_function_1d():
    """Generate a random 1D function for testing."""
    def _make_function(batch_size=5, n_x=64):
        return torch.randn(batch_size, n_x)
    return _make_function


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility in tests."""
    torch.manual_seed(42)
    yield


# Skip slow tests by default
def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

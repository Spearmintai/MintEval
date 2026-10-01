import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PRICE_FILE = str(ROOT / "data/prices/BTCUSDT_15m_2022_2023.csv")


@pytest.fixture(scope="session")
def prices():
    from minteval.data import load_prices
    return load_prices(PRICE_FILE)


@pytest.fixture(scope="session")
def cfg():
    from minteval.engine import EngineConfig
    return EngineConfig()

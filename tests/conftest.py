import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import db  # noqa: E402

DOCS = ROOT / "docs"


@pytest.fixture(autouse=True)
def no_demo_by_default(monkeypatch):
    """Тесты не зависят от DEMO_MODE в окружении или .env: по умолчанию демо выключено."""
    monkeypatch.setattr(config, "DEMO_MODE", False)


@pytest.fixture
def demo_mode(monkeypatch):
    monkeypatch.setattr(config, "DEMO_MODE", True)


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    path = str(tmp_path / "journal.db")
    monkeypatch.setattr(db, "DB_PATH", path)
    return path


@pytest.fixture
def tmp_index(tmp_path, monkeypatch):
    path = str(tmp_path / "index.npz")
    monkeypatch.setattr(config, "INDEX_PATH", path)
    monkeypatch.setattr(config, "DOCS_DIR", str(DOCS))
    return path

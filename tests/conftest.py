import pytest


@pytest.fixture
def tmp_db(tmp_path):
    """每个测试独立的 SQLite 路径。"""
    return str(tmp_path / "test.db")

import pytest
from app.config import Config, load_config


def test_defaults_match_spec():
    cfg = Config(db_path="/tmp/x.db")
    assert cfg.queue_capacity == 1000
    assert cfg.top_k == 16
    assert cfg.adaptive_k_steps == (64, 256)
    assert cfg.window_days == 30
    assert cfg.sweep_interval_seconds == 300
    assert cfg.pending_stale_seconds == 600
    assert cfg.slow_task_warn_seconds == 5.0
    assert cfg.download_retries == 3
    assert cfg.download_retry_backoff_seconds == (1.0, 4.0, 16.0)
    assert cfg.image_max_bytes == 20 * 1024 * 1024
    assert cfg.text_max_chars == 5000
    assert cfg.hard_downweight_threshold == 0.90
    assert cfg.soft_downweight_threshold == 0.75
    assert cfg.device == "cpu"


def test_env_override(monkeypatch):
    monkeypatch.setenv("SIM_TOP_K", "32")
    monkeypatch.setenv("SIM_DB_PATH", "/tmp/other.db")
    cfg = load_config()
    assert cfg.top_k == 32
    assert cfg.db_path == "/tmp/other.db"


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        Config(db_path="/tmp/x.db", hard_downweight_threshold=0.5,
               soft_downweight_threshold=0.9)  # hard 必须 >= soft

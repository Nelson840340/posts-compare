"""全局配置：spec §5.4/§10，所有阈值容量周期收敛于此，支持 SIM_ 前缀环境变量覆盖。"""
import json
from functools import lru_cache
from typing import Tuple

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SIM_")

    db_path: str = "data/similarity.db"
    device: str = "cpu"

    queue_capacity: int = 1000
    queue_wait_seconds: float = 5.0
    top_k: int = 16
    adaptive_k_steps: Tuple[int, ...] = (64, 256)
    window_days: int = 30

    sweep_interval_seconds: int = 300
    pending_stale_seconds: int = 600
    slow_task_warn_seconds: float = 5.0

    download_retries: int = 3
    download_retry_backoff_seconds: Tuple[float, ...] = (1.0, 4.0, 16.0)
    download_timeout_seconds: float = 30.0
    image_max_bytes: int = 20 * 1024 * 1024
    text_max_chars: int = 5000
    graceful_shutdown_seconds: float = 30.0

    hard_downweight_threshold: float = 0.90
    soft_downweight_threshold: float = 0.75

    @field_validator("adaptive_k_steps", "download_retry_backoff_seconds", mode="before")
    @classmethod
    def _parse_tuple(cls, v):
        if isinstance(v, str):
            return tuple(json.loads(v))
        return tuple(v)

    @field_validator("soft_downweight_threshold")
    @classmethod
    def _check_thresholds(cls, v, info):
        hard = info.data.get("hard_downweight_threshold", 0.90)
        if v > hard:
            raise ValueError("soft_downweight_threshold 不得大于 hard_downweight_threshold")
        return v


@lru_cache
def load_config() -> Config:
    return Config()

"""StageTimer：单帖生命周期分阶段耗时打点（spec §7.4）。"""
import time
from contextlib import contextmanager


class StageTimer:
    def __init__(self):
        self.stages: dict[str, float] = {}
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        s = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = (time.perf_counter() - s) * 1000.0

    def total_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def summary(self) -> str:
        parts = " ".join(f"{k}={v:.0f}ms" for k, v in self.stages.items())
        return f"breakdown: {parts}"

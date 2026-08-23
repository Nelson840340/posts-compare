import time

from app.timer import StageTimer


def test_stage_records_elapsed_ms():
    t = StageTimer()
    with t.stage("download"):
        time.sleep(0.02)
    assert t.stages["download"] >= 20.0
    assert t.total_ms() >= 20.0


def test_summary_format_contains_all_stages():
    t = StageTimer()
    with t.stage("download"):
        pass
    with t.stage("image_embed"):
        pass
    s = t.summary()
    assert "download=" in s and "image_embed=" in s and s.endswith("ms")


def test_failed_stage_still_recorded():
    """任一步骤失败时已完成阶段耗时仍可用（spec §7.4）。"""
    t = StageTimer()
    with t.stage("download"):
        time.sleep(0.01)
    try:
        with t.stage("image_embed"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert "download" in t.stages and "image_embed" in t.stages

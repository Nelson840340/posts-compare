import asyncio
import threading
from datetime import datetime, timezone

import pytest
from PIL import Image

from app.config import Config
from app.domain import PostRecord, utcnow
from app.embedder.fake import FakeEmbedder
from app.index.service import IndexService
from app.pipeline.processor import Processor
from app.store.sqlite_store import SqliteStore
from demo.runner import DemoResult, DemoRunner, DemoValidationError

NOW = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)


def _png(color=(9, 9, 9)) -> bytes:
    import io
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def runner(tmp_db):
    # submit() 内含 asyncio.run，测试必须同步执行（与 Gradio 线程池 handler 形态一致）
    cfg = Config(db_path=tmp_db)
    store = SqliteStore(tmp_db)
    r = DemoRunner(cfg, FakeEmbedder(), store, IndexService())
    asyncio.run(r.setup())
    yield r
    asyncio.run(store.close())


def test_submit_before_ready_rejected(tmp_db):
    """A-8：未就绪提交返回明确提示，不抛异常。"""
    cfg = Config(db_path=tmp_db)
    store = SqliteStore(tmp_db)
    r = DemoRunner(cfg, FakeEmbedder(), store, IndexService())
    result = r.submit(_png(), None, "hello")
    assert result.status == "not_ready"
    assert "服务准备中" in (result.error or "")
    asyncio.run(store.close())


def test_setup_rebuilds_index_from_db(tmp_db):
    """A-8 装配链：从库重建索引（SQLite 唯一事实源不变式）。"""
    cfg = Config(db_path=tmp_db)
    store = SqliteStore(tmp_db)
    r1 = DemoRunner(cfg, FakeEmbedder(), store, IndexService())
    asyncio.run(r1.setup())
    result = r1.submit(_png(), None, "种子帖")
    assert result.status == "indexed"
    # 模拟重启：新 runner 从同一库重建
    r2 = DemoRunner(cfg, FakeEmbedder(), store, IndexService())
    asyncio.run(r2.setup())
    assert r2.index.size == 1
    assert r2.ready is True
    asyncio.run(store.close())


def test_both_image_and_url_rejected(runner):
    with pytest.raises(DemoValidationError, match="二选一"):
        runner.submit(_png(), "https://example.com/a.jpg", "t")


def test_empty_content_rejected(runner):
    with pytest.raises(DemoValidationError, match="至少提供"):
        runner.submit(None, None, "")


def test_text_too_long_rejected(runner):
    runner.cfg.text_max_chars = 10
    with pytest.raises(DemoValidationError, match="字符上限"):
        runner.submit(None, None, "x" * 11)


def test_e2e_first_post_and_duplicate(runner):
    r1 = runner.submit(_png((9, 9, 9)), None, "第一帖")
    assert r1.status == "indexed"
    assert r1.post_id.startswith("demo-")
    assert r1.outcome is not None and r1.outcome.result is not None
    assert set(r1.outcome.timings_ms) == {
        "download", "image_embed", "text_embed", "persist",
        "search", "result_persist", "faiss_add"}
    assert r1.submitted_image is not None  # 原图回显
    # 同图重发：图片通道高相似
    r2 = runner.submit(_png((9, 9, 9)), None, "换个文案重发")
    res = r2.outcome.result
    assert res.sim_image > 0.99
    assert res.matched_post_id == r1.post_id


def test_failure_returns_outcome_not_exception(runner):
    """坏图：不抛异常，status=failed 且携带 error_code 与部分耗时（A-2）。"""
    result = runner.submit(b"not-an-image", None, "坏图")
    assert result.status == "failed"
    assert result.outcome is not None
    assert result.outcome.error_code.value == "image_decode_failed"
    assert "download" in result.outcome.timings_ms
    assert result.error == "image_decode_failed"


def test_run_timeout_reported(runner, monkeypatch):
    import demo.runner as mod
    monkeypatch.setattr(mod, "_RUN_TIMEOUT", 0.0001)
    result = runner.submit(_png(), None, "慢帖")
    assert result.status == "failed"
    assert "超时" in (result.error or "")


def test_submit_serialized_by_lock(runner, monkeypatch):
    """A-1：并发 submit 被全局锁串行化——锁内区间（_run）任意时刻最多一个在途。"""
    from app.pipeline.processor import Processor

    events = []
    evlock = threading.Lock()
    original_process = Processor.process

    async def spy_process(self, post_id, *, raise_on_error=True):
        # process 全程在 DemoRunner 锁内执行；探针重叠即锁失效
        with evlock:
            events.append("enter")
        try:
            return await original_process(self, post_id, raise_on_error=raise_on_error)
        finally:
            with evlock:
                events.append("exit")

    monkeypatch.setattr(Processor, "process", spy_process)

    results, errors = [], []

    def worker():
        try:
            results.append(runner.submit(_png(), None, "并发帖"))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(results) == 3 and all(r.status == "indexed" for r in results)
    assert len(set(r.post_id for r in results)) == 3
    depth = peak = 0
    for ev in events:
        depth += 1 if ev == "enter" else -1
        peak = max(peak, depth)
    assert peak == 1  # 锁内全程无重叠

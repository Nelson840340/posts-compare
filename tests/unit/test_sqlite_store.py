import numpy as np
import pytest
from datetime import datetime, timedelta, timezone

from app.domain import (ErrorCode, IndexStatus, PostRecord, SimilarityResult)
from app.store.sqlite_store import SqliteStore, decode_vector, encode_vector


@pytest.fixture
async def store(tmp_db):
    s = SqliteStore(tmp_db)
    await s.init()
    yield s
    await s.close()


def _rec(pid="p1", text="你好世界", **kw):
    return PostRecord(post_id=pid, text=text, **kw)


async def test_upsert_is_idempotent(store):
    assert await store.upsert_post(_rec()) is True
    assert await store.upsert_post(_rec(text="被覆盖?")) is False
    rec = await store.get_post("p1")
    assert rec.text == "你好世界"  # 重复提交不覆盖


async def test_vector_roundtrip_lossless(store):
    vec = np.random.rand(512).astype(np.float32)
    await store.upsert_post(_rec())
    await store.persist_vectors("p1", encode_vector(vec), encode_vector(vec[:384]))
    rows = await store.list_indexed_within(datetime(2000, 1, 1, tzinfo=timezone.utc))
    assert rows == []  # 尚未 indexed，重建不含它


async def test_mark_indexed_with_result_in_one_transaction(store):
    await store.upsert_post(_rec())
    vec = np.ones(512, dtype=np.float32)
    await store.persist_vectors("p1", encode_vector(vec), encode_vector(vec))
    res = SimilarityResult(post_id="p1", max_sim=0.0, sim_image=0.0, sim_text=0.0,
                           matched_post_id=None,
                           computed_at=datetime.now(timezone.utc), note=None)
    await store.mark_indexed("p1", res)
    rec = await store.get_post("p1")
    assert rec.status == IndexStatus.INDEXED
    got = await store.get_result("p1")
    assert got.max_sim == 0.0 and got.matched_post_id is None
    rows = await store.list_indexed_within(datetime(2000, 1, 1, tzinfo=timezone.utc))
    assert len(rows) == 1 and rows[0][0] == "p1"
    assert np.allclose(decode_vector(rows[0][1]), vec)


async def test_mark_failed_and_reset(store):
    await store.upsert_post(_rec())
    await store.mark_failed("p1", ErrorCode.IMAGE_DOWNLOAD_FAILED)
    rec = await store.get_post("p1")
    assert rec.status == IndexStatus.FAILED
    assert rec.note == "image_download_failed"
    assert await store.reset_failed_to_pending("p1") is True
    assert (await store.get_post("p1")).status == IndexStatus.PENDING


async def test_list_stale_pending_filters_by_time(store):
    await store.upsert_post(_rec("fresh"))
    await store.upsert_post(_rec("stale"))
    # 手工把 stale帖的 enqueued_at 拨早（_execute 为同步方法，直接调用）
    store._execute(
        "UPDATE posts SET enqueued_at=? WHERE post_id=?",
        ("2026-01-01T00:00:00+00:00", "stale"))
    stale = await store.list_stale_pending(datetime(2026, 8, 1, tzinfo=timezone.utc))
    assert stale == ["stale"]


async def test_counts(store):
    await store.upsert_post(_rec("a"))
    await store.upsert_post(_rec("b"))
    await store.mark_failed("b", ErrorCode.IMAGE_DECODE_FAILED)
    c = await store.counts()
    assert c["failed_count"] == 1


async def test_close_drains_all_worker_connections(tmp_db):
    s = SqliteStore(tmp_db)
    await s.init()
    await s.upsert_post(_rec())
    assert len(s._conns) >= 1  # worker 线程已建连并登记
    await s.close()
    assert s._conns == []  # close 必须关闭全部登记连接，不只当前线程
    assert (await s.get_post("p1")).text == "你好世界"  # 关闭后重开仍可用
    await s.close()

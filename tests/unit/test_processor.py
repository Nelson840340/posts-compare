import base64
import io
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from PIL import Image

from app.config import Config
from app.domain import IndexStatus, PostRecord
from app.embedder.fake import FakeEmbedder
from app.index.service import IndexService
from app.pipeline.processor import Processor
from app.store.sqlite_store import SqliteStore

NOW = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)


def _png(color=(9, 9, 9)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
async def env(tmp_db):
    cfg = Config(db_path=tmp_db)
    store = SqliteStore(tmp_db)
    await store.init()
    index = IndexService()
    proc = Processor(store=store, index=index, embedder=FakeEmbedder(), cfg=cfg)
    yield cfg, store, index, proc
    await store.close()


async def _submit(store, pid, png, text, created_at):
    rec = PostRecord(post_id=pid, image_base64=base64.b64encode(png).decode(),
                     text=text, created_at=created_at)
    assert await store.upsert_post(rec)


async def test_first_post_scores_zero(env):
    _, store, index, proc = env
    await _submit(store, "p1", _png(), "第一帖", NOW)
    await proc.process("p1")
    res = await store.get_result("p1")
    assert res.max_sim == 0.0 and res.matched_post_id is None
    assert (await store.get_post("p1")).status == IndexStatus.INDEXED
    assert index.size == 1  # 已注册


async def test_duplicate_image_high_similarity(env):
    _, store, index, proc = env
    png = _png()
    await _submit(store, "p1", png, "第一帖", NOW - timedelta(hours=1))
    await proc.process("p1")
    await _submit(store, "p2", png, "换个文案重发", NOW)
    await proc.process("p2")
    res = await store.get_result("p2")
    assert res.sim_image > 0.99
    assert res.max_sim == res.sim_image
    assert res.matched_post_id == "p1"


async def test_outside_window_short_circuits(env):
    """spec §3.4⑧：created_at 超窗 → 零分短路，不提特征不注册。"""
    cfg, store, index, proc = env
    await _submit(store, "old", _png(), "回填老帖", NOW - timedelta(days=40))
    await proc.process("old")
    res = await store.get_result("old")
    assert res.max_sim == 0.0 and res.note == "created_at_outside_window"
    assert index.size == 0  # 未注册索引


async def test_failure_marks_failed_and_reraises(env):
    _, store, index, proc = env
    rec = PostRecord(post_id="bad", image_base64=base64.b64encode(b"broken").decode(),
                     text="坏图", created_at=NOW)
    await store.upsert_post(rec)
    from app.domain import ProcessingError
    with pytest.raises(ProcessingError):
        await proc.process("bad")
    post = await store.get_post("bad")
    assert post.status == IndexStatus.FAILED
    assert post.note == "image_decode_failed"
    assert index.size == 0


async def test_order_vector_persisted_before_index_add(env):
    """顺序铁律：mark_indexed 后向量已在库（重启可重建）。"""
    _, store, index, proc = env
    await _submit(store, "p1", _png(), "t", NOW)
    await proc.process("p1")
    rows = await store.list_indexed_within(NOW - timedelta(days=30))
    assert len(rows) == 1 and rows[0][1] is not None and rows[0][2] is not None


async def test_no_image_post_text_only(env):
    """无图帖（前置字段检查）：图片通道零分，文字通道正常检索与注册。"""
    _, store, index, proc = env
    rec1 = PostRecord(post_id="p1", text="完全相同的正文内容", created_at=NOW - timedelta(hours=1))
    assert await store.upsert_post(rec1)
    await proc.process("p1")
    rec2 = PostRecord(post_id="p2", text="完全相同的正文内容", created_at=NOW)
    assert await store.upsert_post(rec2)
    await proc.process("p2")
    res = await store.get_result("p2")
    assert res.sim_image == 0.0
    assert res.sim_text > 0.99 and res.matched_post_id == "p1"
    assert index.size == 2
    # 重建路径：image_vec NULL 不得把无图帖挡在重建集外
    rows = await store.list_indexed_within(NOW - timedelta(days=30))
    assert len(rows) == 2

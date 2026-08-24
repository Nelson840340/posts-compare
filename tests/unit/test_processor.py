import base64
import io
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from PIL import Image

from app.config import Config
from app.domain import EncodeError, IndexStatus, PostRecord
from app.embedder.fake import FakeEmbedder
from app.index.service import IndexService
from app.pipeline.processor import ProcessOutcome, Processor
from app.store.sqlite_store import SqliteStore, decode_vector

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


async def test_order_vector_persisted_before_index_add(env, monkeypatch):
    """顺序铁律：persist 后 index.add 崩溃 → 向量已在库（重启可重建），状态 failed 不入重建集。"""
    _, store, index, proc = env
    await _submit(store, "p1", _png(), "t", NOW)

    def _boom(*args, **kwargs):
        raise RuntimeError("faiss add 崩溃")
    monkeypatch.setattr(index, "add", _boom)
    with pytest.raises(RuntimeError):
        await proc.process("p1")
    row = store._query("SELECT image_vec, text_vec FROM posts WHERE post_id=?", ("p1",))
    assert row[0][0] is not None and row[0][1] is not None  # 向量已落库
    post = await store.get_post("p1")
    assert post.status == IndexStatus.FAILED  # 不会被误标 indexed
    assert index.size == 0  # 未注册


async def test_unexpected_exception_marks_internal_error(env, monkeypatch):
    """无 code 属性的意外异常 → mark_failed(INTERNAL_ERROR) 后 re-raise。"""
    _, store, index, proc = env
    await _submit(store, "p1", _png(), "t", NOW)

    async def _boom(*args, **kwargs):
        raise RuntimeError("存储爆炸")
    # 注入非编码路径（编码异常归 encode_failed，最终审查 Warning-3）
    monkeypatch.setattr(proc.store, "persist_vectors", _boom)
    with pytest.raises(RuntimeError):
        await proc.process("p1")
    post = await store.get_post("p1")
    assert post.status == IndexStatus.FAILED and post.note == "internal_error"


async def test_encode_failure_marks_encode_failed(env, monkeypatch):
    """spec §7.2：编码异常 → failed(encode_failed)，不再落入 internal_error。"""
    _, store, index, proc = env
    await _submit(store, "p1", _png(), "t", NOW)

    def _boom(image_bytes):
        raise RuntimeError("模型爆炸")
    monkeypatch.setattr(proc.embedder, "embed_image", _boom)
    with pytest.raises(EncodeError):  # 编码异常被包裹为 EncodeError（带 code 归类）
        await proc.process("p1")
    post = await store.get_post("p1")
    assert post.status == IndexStatus.FAILED and post.note == "encode_failed"


async def test_prefix_contract_query_search_passage_persist(env):
    """spec §5.1：检索传 query 前缀向量；落库/注册用 passage 前缀向量。"""
    _, store, index, proc = env
    await _submit(store, "p1", _png(), "前缀契约正文", NOW)
    seen = {}
    orig_search = index.search
    orig_add = index.add

    def spy_search(iv, tv, **kwargs):
        seen["query_vec"] = tv
        return orig_search(iv, tv, **kwargs)

    def spy_add(pid, created_at, iv, tv):
        seen["add_vec"] = tv
        return orig_add(pid, created_at, iv, tv)
    index.search = spy_search
    index.add = spy_add
    await proc.process("p1")

    emb = FakeEmbedder()
    expected_query = emb.embed_text("前缀契约正文", prefix="query: ")
    expected_passage = emb.embed_text("前缀契约正文", prefix="passage: ")
    assert not np.allclose(expected_query, expected_passage)  # 前缀确实改变向量
    assert np.allclose(seen["query_vec"], expected_query)
    assert np.allclose(seen["add_vec"], expected_passage)  # 注册也用 passage 向量
    row = store._query("SELECT text_vec FROM posts WHERE post_id=?", ("p1",))
    assert np.allclose(decode_vector(row[0][0]), expected_passage)


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
    assert res.sim_text > 0.95 and res.matched_post_id == "p1"
    assert index.size == 2
    # 重建路径：image_vec NULL 不得把无图帖挡在重建集外
    rows = await store.list_indexed_within(NOW - timedelta(days=30))
    assert len(rows) == 2


async def test_process_returns_outcome_with_stage_timings(env):
    _, store, index, proc = env
    await _submit(store, "p1", _png(), "第一帖", NOW)
    outcome = await proc.process("p1")
    assert isinstance(outcome, ProcessOutcome)
    assert outcome.result is not None and outcome.error_code is None
    assert set(outcome.timings_ms) == {
        "download", "image_embed", "text_embed", "persist",
        "search", "result_persist", "faiss_add"}
    assert all(v >= 0.0 for v in outcome.timings_ms.values())
    assert outcome.total_ms > 0


async def test_process_failure_returns_outcome_when_raise_disabled(env):
    _, store, index, proc = env
    rec = PostRecord(post_id="bad", image_base64=base64.b64encode(b"broken").decode(),
                     text="坏图", created_at=NOW)
    await store.upsert_post(rec)
    outcome = await proc.process("bad", raise_on_error=False)
    assert outcome.result is None
    assert outcome.error_code.value == "image_decode_failed"
    assert "download" in outcome.timings_ms  # 失败前已完成阶段的耗时保留
    post = await store.get_post("bad")
    assert post.status == IndexStatus.FAILED


async def test_process_default_still_raises(env):
    """raise_on_error 默认 True：生产 worker 的异常契约逐字节不变。"""
    _, store, index, proc = env
    rec = PostRecord(post_id="bad", image_base64=base64.b64encode(b"broken").decode(),
                     text="坏图", created_at=NOW)
    await store.upsert_post(rec)
    from app.domain import ProcessingError
    with pytest.raises(ProcessingError):
        await proc.process("bad")

import asyncio
import base64
import io

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from PIL import Image

from app.api.routes import COMPACT_SENTINEL, create_router, register_error_handlers
from app.config import Config
from app.embedder.fake import FakeEmbedder
from app.index.service import IndexService
from app.pipeline.processor import Processor
from app.store.sqlite_store import SqliteStore


def _png() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (5, 5, 5)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
async def env(tmp_db):
    """纯 ASGI 测试：httpx.AsyncClient + ASGITransport，与 pytest-asyncio 同循环。"""
    cfg = Config(db_path=tmp_db)
    store = SqliteStore(tmp_db)
    await store.init()
    index = IndexService()
    proc = Processor(store=store, index=index, embedder=FakeEmbedder(), cfg=cfg)
    enqueued: list[str] = []
    app = FastAPI()
    app.include_router(create_router(store, proc, cfg, enqueued.append, {"ready": True}))
    register_error_handlers(app)
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://test") as client:
        yield client, store, proc, enqueued
    await store.close()


async def test_submit_returns_202_and_enqueues(env):
    client, _, _, enqueued = env
    r = await client.post("/posts", json={"post_id": "p1", "image_base64": _png(), "text": "你好"})
    assert r.status_code == 202
    assert r.json() == {"post_id": "p1", "status": "pending"}
    assert enqueued == ["p1"]


async def test_no_image_post_is_legal(env):
    """Demo 契约：1 张图 + 1 段正文，无图帖合法（仅 text）。"""
    client, _, _, enqueued = env
    r = await client.post("/posts", json={"post_id": "txt", "text": "纯文字帖"})
    assert r.status_code == 202
    assert enqueued == ["txt"]


async def test_duplicate_post_id_is_idempotent(env):
    client, _, _, enqueued = env
    body = {"post_id": "p1", "image_base64": _png(), "text": "你好"}
    assert (await client.post("/posts", json=body)).status_code == 202
    r = await client.post("/posts", json=body)
    assert r.status_code == 200
    assert r.json()["status"] == "pending"
    assert enqueued == ["p1"]  # 未二次入队


async def test_concurrent_duplicate_submits_enqueue_once(env):
    """TOCTOU 回归：并发重复提交仅一个 202，只入队一次（upsert 原子判定）。"""
    client, _, _, enqueued = env
    body = {"post_id": "p1", "image_base64": _png(), "text": "你好"}
    rs = await asyncio.gather(*[client.post("/posts", json=body) for _ in range(5)])
    codes = sorted(r.status_code for r in rs)
    assert codes == [200, 200, 200, 200, 202]
    assert enqueued == ["p1"]


async def test_validation_errors(env):
    client, _, _, _ = env
    r = await client.post("/posts", json={"post_id": "p"})  # 缺 text
    assert r.status_code == 422 and "error" in r.json()
    r = await client.post("/posts", json={"image_base64": _png(), "text": "无id"})
    assert r.status_code == 422
    r = await client.post("/posts", json={"post_id": "x" * 129, "image_base64": _png(), "text": "t"})
    assert r.status_code == 422
    r = await client.post("/posts", json={"post_id": "p", "image_base64": _png(), "text": "t" * 5001})
    assert r.status_code == 422
    r = await client.post("/posts", json={"post_id": "p", "text": "t", "created_at": "不是日期"})
    assert r.status_code == 422
    r = await client.post("/posts", json={"post_id": COMPACT_SENTINEL, "text": "t"})
    assert r.status_code == 422  # 保留哨兵不可用作 post_id


async def test_similarity_three_states(env):
    client, _, proc, _ = env
    assert (await client.get("/posts/nope/similarity")).status_code == 404
    await client.post("/posts", json={"post_id": "p1", "image_base64": _png(), "text": "你好"})
    r = await client.get("/posts/p1/similarity")
    assert r.status_code == 202 and r.json()["status"] == "pending"

    await proc.process("p1")  # 同循环直接驱动处理器
    r = await client.get("/posts/p1/similarity")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "indexed" and body["max_sim"] == 0.0
    assert set(body) >= {"max_sim", "sim_image", "sim_text", "matched_post_id", "computed_at"}


async def test_failed_state_returns_reason(env):
    client, _, proc, _ = env
    await client.post("/posts", json={"post_id": "bad",
                                      "image_base64": base64.b64encode(b"x").decode(),
                                      "text": "t"})
    from app.domain import ProcessingError
    with pytest.raises(ProcessingError):
        await proc.process("bad")
    r = await client.get("/posts/bad/similarity")
    assert r.status_code == 200
    assert r.json() == {"status": "failed", "reason": "image_decode_failed"}


async def test_health_ready_200(env):
    client, _, _, _ = env
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["models_loaded"] is True
    assert set(body) >= {"index_count", "failed_count"}


async def test_text_exceeds_configured_limit_returns_422(tmp_db):
    """text 上限走 cfg.text_max_chars（env 可覆盖），不再硬编码 schema（最终审查 Warning-4）。"""
    cfg = Config(db_path=tmp_db, text_max_chars=10)
    store = SqliteStore(tmp_db)
    await store.init()
    proc = Processor(store=store, index=IndexService(), embedder=FakeEmbedder(), cfg=cfg)
    app = FastAPI()
    app.include_router(create_router(store, proc, cfg, lambda p: None, {"ready": True}))
    register_error_handlers(app)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/posts", json={"post_id": "p1", "text": "超" * 11})
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "validation_error"
    await store.close()


async def test_health_not_ready_503(tmp_db):
    cfg = Config(db_path=tmp_db)
    store = SqliteStore(tmp_db)
    await store.init()
    proc = Processor(store=store, index=IndexService(),
                     embedder=FakeEmbedder(), cfg=cfg)
    app = FastAPI()
    app.include_router(create_router(store, proc, cfg, lambda p: None, {"ready": False}))
    register_error_handlers(app)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        assert (await c.get("/health")).status_code == 503
    await store.close()


async def test_admin_replay_resets_failed(env):
    client, store, proc, enqueued = env
    await client.post("/posts", json={"post_id": "bad",
                                      "image_base64": base64.b64encode(b"x").decode(),
                                      "text": "t"})
    from app.domain import IndexStatus, ProcessingError
    with pytest.raises(ProcessingError):
        await proc.process("bad")
    r = await client.post("/admin/replay", json={"post_id": "bad"})
    assert r.status_code == 200 and r.json() == {"post_id": "bad", "reset": True}
    assert (await store.get_post("bad")).status == IndexStatus.PENDING
    assert "bad" in enqueued
    # 未 failed 的帖子不重置
    r = await client.post("/admin/replay", json={"post_id": "ghost"})
    assert r.json() == {"post_id": "ghost", "reset": False}
    assert "ghost" not in enqueued
    # 非标量 post_id → 422 统一格式，不得 500
    r = await client.post("/admin/replay", json={"post_id": ["a"]})
    assert r.status_code == 422 and "error" in r.json()


async def test_admin_replay_compact_enqueues_sentinel(env):
    """Carry-forward：compact 不得在请求路径直接调 index.compact，须入队经 Worker 串行化。"""
    client, _, _, enqueued = env
    r = await client.post("/admin/replay", json={"mode": "compact"})
    assert r.status_code == 200
    assert r.json()["mode"] == "compact"
    assert enqueued == [COMPACT_SENTINEL]

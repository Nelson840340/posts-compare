import asyncio
import base64
import io

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from app.config import Config
from app.domain import PostRecord
from app.embedder.fake import FakeEmbedder
from app.service import build_app
from app.store.sqlite_store import SqliteStore


def _png(color=(7, 7, 7)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (12, 12), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


async def _wait_until(pred, timeout=5.0, interval=0.05):
    async def wrap():
        while True:
            if await pred():
                return
            await asyncio.sleep(interval)
    await asyncio.wait_for(wrap(), timeout)


def _indexed(client, pid):
    async def pred():
        return (await client.get(f"/posts/{pid}/similarity")).json().get("status") == "indexed"
    return pred


@pytest.fixture
async def app_env(tmp_db):
    cfg = Config(db_path=tmp_db, sweep_interval_seconds=600)
    app = build_app(cfg, embedder=FakeEmbedder())
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as client:
            yield client, cfg
    # lifespan 退出即完成优雅停机


async def test_end_to_end_submit_and_result(app_env):
    client, _ = app_env
    r = await client.post("/posts", json={"post_id": "p1", "image_base64": _png(), "text": "第一帖"})
    assert r.status_code == 202
    await _wait_until(_indexed(client, "p1"))
    body = (await client.get("/posts/p1/similarity")).json()
    assert body["max_sim"] == 0.0  # 首帖


async def test_duplicate_detected_by_worker(app_env):
    client, _ = app_env
    png = _png()
    await client.post("/posts", json={"post_id": "p1", "image_base64": png, "text": "原帖"})
    await _wait_until(_indexed(client, "p1"))

    await client.post("/posts", json={"post_id": "p2", "image_base64": png, "text": "换文案重发"})
    await _wait_until(_indexed(client, "p2"))
    body = (await client.get("/posts/p2/similarity")).json()
    assert body["matched_post_id"] == "p1" and body["sim_image"] > 0.99


async def test_startup_rebuild_and_replay(tmp_db):
    """spec §3.4②：重启后索引从库重建 + pending 回放，零数据丢失。"""
    cfg = Config(db_path=tmp_db)
    png = _png()

    # ---- 第一生命周期：完整索引 p1 ----
    app = build_app(cfg, embedder=FakeEmbedder())
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            await c.post("/posts", json={"post_id": "p1", "image_base64": png, "text": "原帖"})
            await _wait_until(_indexed(c, "p1"))

    # ---- 模拟崩溃残留：直接向库插入 pending 的 p2（队列任务已丢失）----
    store = SqliteStore(tmp_db)
    await store.init()
    await store.upsert_post(PostRecord(post_id="p2", text="重发帖", image_base64=png))
    await store.close()

    # ---- 第二生命周期：启动重建索引 + 回放 p2 ----
    app2 = build_app(cfg, embedder=FakeEmbedder())
    async with app2.router.lifespan_context(app2):
        async with AsyncClient(transport=ASGITransport(app=app2), base_url="http://t") as c:
            await _wait_until(_indexed(c, "p2"))
            body = (await c.get("/posts/p2/similarity")).json()
            assert body["matched_post_id"] == "p1"  # p1 由启动重建恢复进索引
            health = (await c.get("/health")).json()
            assert health["status"] == "ok" and health["index_count"] == 2


async def test_sweep_reenqueues_stale_pending(tmp_db):
    cfg = Config(db_path=tmp_db, sweep_interval_seconds=1, pending_stale_seconds=0)
    store = SqliteStore(tmp_db)
    await store.init()
    await store.upsert_post(PostRecord(post_id="stale", text="滞留帖", image_base64=_png()))
    await store.close()

    app = build_app(cfg, embedder=FakeEmbedder())
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            await _wait_until(_indexed(c, "stale"), timeout=10.0)  # 回放/sweep 兜底处理


async def test_compact_sentinel_consumed_by_worker(app_env):
    """Carry-forward：mode=compact 入队哨兵，Worker 串行执行且不中断后续消费。"""
    client, _ = app_env
    await client.post("/posts", json={"post_id": "p1", "image_base64": _png(), "text": "t"})
    await _wait_until(_indexed(client, "p1"))

    r = await client.post("/admin/replay", json={"mode": "compact"})
    assert r.json() == {"mode": "compact", "status": "scheduled"}

    # 哨兵被 worker 消费后仍正常处理后续帖子（未把哨兵当 post_id）
    await client.post("/posts", json={"post_id": "p2", "image_base64": _png(), "text": "t2"})
    await _wait_until(_indexed(client, "p2"))
    health = (await client.get("/health")).json()
    assert health["index_count"] == 2  # 窗内无过期条目，compact 剔除 0


async def test_graceful_shutdown_drains_queue(tmp_db):
    """spec §7.5：lifespan 退出前排空队列（哨兵在队尾，先处理完存量帖子）。"""
    cfg = Config(db_path=tmp_db, sweep_interval_seconds=600)
    app = build_app(cfg, embedder=FakeEmbedder())
    store_probe = SqliteStore(tmp_db)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            for i in range(3):
                await c.post("/posts", json={"post_id": f"p{i}",
                                             "image_base64": _png((i, i, i)), "text": f"帖{i}"})
    # 停机后全部入库（worker 排空至哨兵）
    await store_probe.init()
    counts = await store_probe.counts()
    await store_probe.close()
    assert counts["index_count"] == 3

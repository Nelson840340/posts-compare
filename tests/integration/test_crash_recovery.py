import asyncio
import base64
import io
import sqlite3
from datetime import datetime, timezone

from httpx import ASGITransport, AsyncClient
from PIL import Image

from app.config import Config
from app.embedder.fake import FakeEmbedder
from app.service import build_app


def _png(color=(3, 4, 5)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (12, 12), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


async def _wait_until(pred, timeout=5.0):
    async def wrap():
        while True:
            if await pred():
                return
            await asyncio.sleep(0.05)
    await asyncio.wait_for(wrap(), timeout)


async def test_invariants_after_simulated_crash(tmp_db):
    """spec §7.1 验收：
    不变式1 —— indexed 帖库中必有向量与结果；
    不变式2 —— 重启后索引 == 库中 indexed 集合（重建是纯函数）；
    不变式3 —— 崩溃丢失的在途任务（pending）被回放补齐，零数据丢失。
    """
    cfg = Config(db_path=tmp_db)
    png = _png()

    # 实例一：索引 p1 后正常结束（p1 已 indexed，无在途任务，库状态与崩溃等价）；
    # 崩溃的核心现场——"已落库 pending 但队列任务丢失"——由下方原生 INSERT 模拟
    app = build_app(cfg, embedder=FakeEmbedder())
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            await c.post("/posts", json={"post_id": "p1", "image_base64": png, "text": "原帖"})

            async def p1():
                return (await c.get("/posts/p1/similarity")).json().get("status") == "indexed"
            await _wait_until(p1)

    # 崩溃现场取证：p1 的向量与结果确实在库（不变式1）
    conn = sqlite3.connect(tmp_db)
    row = conn.execute(
        "SELECT image_vec IS NOT NULL, text_vec IS NOT NULL, status FROM posts "
        "WHERE post_id='p1'").fetchone()
    res = conn.execute("SELECT COUNT(*) FROM results WHERE post_id='p1'").fetchone()
    conn.close()
    assert row == (1, 1, "indexed") and res[0] == 1

    # 模拟崩溃丢失在途任务：p2 直接以 pending 落库（未出队即进程死亡）
    conn = sqlite3.connect(tmp_db)
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO posts(post_id,text,image_base64,created_at,enqueued_at,status) "
        "VALUES('p2','重发帖',?,?,?, 'pending')", (png, now, now))
    conn.commit()
    conn.close()

    # 实例二：全新进程（空索引）启动
    app2 = build_app(cfg, embedder=FakeEmbedder())
    async with app2.router.lifespan_context(app2):
        async with AsyncClient(transport=ASGITransport(app=app2), base_url="http://t") as c:
            # 不变式2：p1 经启动重建回到索引 —— 通过 p2 能匹配到 p1 验证
            async def p2():
                return (await c.get("/posts/p2/similarity")).json().get("status") == "indexed"
            await _wait_until(p2)
            body = (await c.get("/posts/p2/similarity")).json()
            assert body["matched_post_id"] == "p1"
            assert body["sim_image"] > 0.99  # 不变式3：pending 回放补齐，零丢失

            # 不变式2 等式方向：mark_indexed 先于 faiss_add，需先 join 消除观测窗口
            await app2.state.queue.join()
            assert app2.state.index.size == 2  # faiss 条目数 == 库中 indexed 集合

            health = (await c.get("/health")).json()
            assert health["index_count"] == 2

"""应用装配与生命周期（spec §3.4②启动序列 / §7.3 in-flight / §7.5 优雅停机）。"""
import asyncio
import contextlib
import logging
import os
from contextlib import asynccontextmanager
from datetime import timedelta

from fastapi import FastAPI

from app.api.routes import create_router, register_error_handlers
from app.api.schemas import COMPACT_SENTINEL
from app.config import Config
from app.domain import ErrorCode, ProcessingError, utcnow, window_cutoff
from app.embedder.base import Embedder
from app.index.service import IndexService
from app.pipeline.processor import Processor
from app.store.sqlite_store import SqliteStore, decode_vector

logger = logging.getLogger("service")

_SHUTDOWN_SENTINEL = ""  # 停机哨兵：worker 消费到即退出循环


def build_app(cfg: Config, embedder: Embedder | None = None) -> FastAPI:
    if embedder is None:
        from app.embedder.clip_e5 import ClipE5Embedder
        embedder = ClipE5Embedder(device=cfg.device)

    store = SqliteStore(cfg.db_path)
    index = IndexService()
    processor = Processor(store=store, index=index, embedder=embedder, cfg=cfg)

    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=cfg.queue_capacity)
    in_flight: set[str] = set()
    health_state: dict = {"ready": False, "live": {}}
    tasks: dict[str, asyncio.Task] = {}

    async def enqueue(post_id: str) -> None:
        try:
            queue.put_nowait(post_id)
            return
        except asyncio.QueueFull:
            pass
        try:
            await asyncio.wait_for(queue.put(post_id), timeout=cfg.queue_wait_seconds)
        except asyncio.TimeoutError:
            # 帖子已以 pending 落库，sweep 兜底（spec §4.3）
            logger.warning("post_id=%s 入队超时，等待兜底扫描", post_id)

    async def worker():
        while True:
            post_id = await queue.get()
            if post_id == _SHUTDOWN_SENTINEL:  # 停机哨兵
                queue.task_done()
                break
            if post_id == COMPACT_SENTINEL:
                # 压缩与 search/add 同一 worker 串行执行（IndexService 非线程安全）
                try:
                    removed = await asyncio.to_thread(
                        index.compact, utcnow(), cfg.window_days)
                    logger.info("索引压缩完成：剔除 %d 条过期条目", removed)
                except Exception:
                    logger.exception("索引压缩失败")
                queue.task_done()
                continue
            in_flight.add(post_id)
            try:
                await processor.process(post_id)
            except ProcessingError:
                pass  # 已 mark_failed
            except Exception:
                logger.exception("post_id=%s 处理异常", post_id)
                with contextlib.suppress(Exception):
                    await store.mark_failed(post_id, ErrorCode.INTERNAL_ERROR)
            finally:
                in_flight.discard(post_id)
                queue.task_done()

    async def sweep():
        while True:
            await asyncio.sleep(cfg.sweep_interval_seconds)
            older = utcnow() - timedelta(seconds=cfg.pending_stale_seconds)
            for pid in await store.list_stale_pending(older):
                if pid not in in_flight:
                    logger.info("post_id=%s 兜底扫描重新入队", pid)
                    await enqueue(pid)

    async def stats():
        while True:
            await asyncio.sleep(5)
            health_state["live"] = {
                "queue_size": queue.qsize(),
                "in_flight": len(in_flight),
                # spec §6.3/§7.4 /health 四指标之一：最早 pending 滞留秒数
                "oldest_pending_seconds": await store.oldest_pending_age()}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        os.makedirs(os.path.dirname(cfg.db_path) or ".", exist_ok=True)
        await asyncio.to_thread(embedder.load)
        await store.init()
        # ---- 启动序列（spec §3.4②）：重建索引 → 回放 pending ----
        cutoff = window_cutoff(utcnow(), cfg.window_days)
        rows = await store.list_indexed_within(cutoff)
        index.rebuild([(pid, ts, decode_vector(iv), decode_vector(tv))
                       for pid, iv, tv, ts in rows])
        logger.info("启动重建索引完成：%d 条", index.size)
        # worker 先启动再回放：回放 put 满队列时由 worker 消费形成背压，避免启动死锁
        tasks["worker"] = asyncio.create_task(worker())
        tasks["sweep"] = asyncio.create_task(sweep())
        tasks["stats"] = asyncio.create_task(stats())
        far_future = utcnow() + timedelta(days=1)
        for pid in await store.list_stale_pending(far_future):  # 全部 pending 回放
            await queue.put(pid)
        health_state["ready"] = True
        try:
            yield
        finally:
            health_state["ready"] = False
            # 哨兵必须送达 worker 才能排空退出；队列满时阻塞等待而非丢弃
            # （否则哨兵丢失必然退化为超时强杀，最终审查 Warning-2）
            sentinel_in = True
            try:
                await asyncio.wait_for(queue.put(_SHUTDOWN_SENTINEL),
                                       timeout=cfg.graceful_shutdown_seconds)
            except asyncio.TimeoutError:
                sentinel_in = False  # 塞不进哨兵才走超时强杀
            if sentinel_in:
                try:
                    await asyncio.wait_for(tasks["worker"],
                                           timeout=cfg.graceful_shutdown_seconds)
                except asyncio.TimeoutError:
                    tasks["worker"].cancel()
            else:
                tasks["worker"].cancel()
            for name in ("sweep", "stats"):
                tasks[name].cancel()
            await store.close()
            logger.info("优雅停机完成")

    app = FastAPI(title="post-similarity", lifespan=lifespan)
    app.state.queue = queue  # 供运维/测试直接投递（如重复入队回归验证）
    app.state.index = index  # 供测试直接观测 faiss 条目数（ntotal）
    register_error_handlers(app)
    app.include_router(create_router(store, processor, cfg, enqueue, health_state))
    return app

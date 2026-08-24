"""单帖处理编排（spec §4.1 顺序铁律 + §7.4 耗时打点 + §3.4⑧ 超窗短路）。"""
import asyncio
import logging
from dataclasses import dataclass

import numpy as np

from app.config import Config
from app.domain import (EncodeError, ErrorCode, IndexStatus, SimilarityResult,
                        utcnow, window_cutoff)
from app.embedder.base import Embedder
from app.index.service import IndexService
from app.pipeline.downloader import decode_and_validate, fetch_image
from app.store.base import PostStore
from app.store.sqlite_store import encode_vector
from app.timer import StageTimer

logger = logging.getLogger("pipeline")


@dataclass
class ProcessOutcome:
    """单帖处理结果（spec 2026-08-24 §3.1 / 附录 A-2）：供 demo 同步展示分数与分阶段耗时。"""
    result: SimilarityResult | None
    error_code: ErrorCode | None      # 带 code 的失败原因；意外异常（无 code）为 None
    timings_ms: dict[str, float]
    total_ms: float


class Processor:
    def __init__(self, store: PostStore, index: IndexService,
                 embedder: Embedder, cfg: Config):
        self.store = store
        self.index = index
        self.embedder = embedder
        self.cfg = cfg

    async def process(self, post_id: str, *, raise_on_error: bool = True) -> ProcessOutcome:
        timer = StageTimer()
        rec = await self.store.get_post(post_id)
        if rec is None:
            logger.warning("post_id=%s 不存在，跳过", post_id)
            return ProcessOutcome(None, None, dict(timer.stages), timer.total_ms())
        if rec.status != IndexStatus.PENDING:
            # 幂等守卫：拦截 sweep/入队竞态下的重复处理（避免重复注册索引条目）
            logger.info("post_id=%s status=%s，跳过重复处理", post_id, rec.status.value)
            return ProcessOutcome(None, None, dict(timer.stages), timer.total_ms())
        try:
            # ---- 超窗短路（spec §3.4⑧）----
            cutoff = window_cutoff(utcnow(), self.cfg.window_days)
            if rec.created_at < cutoff:
                result = SimilarityResult(
                    post_id=post_id, max_sim=0.0, sim_image=0.0, sim_text=0.0,
                    matched_post_id=None, computed_at=utcnow(),
                    note="created_at_outside_window")
                await self.store.mark_indexed(post_id, result)
                logger.info("post_id=%s phase=skip reason=created_at_outside_window", post_id)
                return ProcessOutcome(result, None, dict(timer.stages), timer.total_ms())

            # ---- 无图帖：前置字段检查跳过图片通道，零向量落库（sim 恒 0 不误匹配）----
            has_image = bool(rec.image_url or rec.image_base64)
            if has_image:
                with timer.stage("download"):
                    raw = await fetch_image(rec, self.cfg)
                    image_bytes = decode_and_validate(raw, self.cfg)
                with timer.stage("image_embed"):
                    try:
                        image_vec = await asyncio.to_thread(
                            self.embedder.embed_image, image_bytes)
                    except Exception as e:
                        raise EncodeError(f"图片向量化失败: {e}") from e
            else:
                image_vec = np.zeros(self.embedder.image_dim, dtype=np.float32)
            with timer.stage("text_embed"):
                # 检索用 query 前缀；落库/注册用 passage 前缀（库存侧契约，spec §5.1）
                try:
                    text_vec = await asyncio.to_thread(
                        lambda: self.embedder.embed_text(rec.text, prefix="query: "))
                    passage_vec = await asyncio.to_thread(
                        lambda: self.embedder.embed_text(rec.text, prefix="passage: "))
                except Exception as e:
                    raise EncodeError(f"文字向量化失败: {e}") from e

            with timer.stage("persist"):
                await self.store.persist_vectors(
                    post_id, encode_vector(image_vec), encode_vector(passage_vec))

            with timer.stage("search"):
                # 检索在 to_thread 中执行（IndexService 非线程安全，单 Worker 串行）
                hit = await asyncio.to_thread(
                    self.index.search, image_vec, text_vec,
                    exclude_id=post_id, cutoff=cutoff,
                    top_k=self.cfg.top_k,
                    adaptive_k_steps=self.cfg.adaptive_k_steps)

            max_sim = max(hit.sim_image, hit.sim_text)
            result = SimilarityResult(
                post_id=post_id,
                max_sim=round(max_sim, 4),
                sim_image=round(hit.sim_image, 4),
                sim_text=round(hit.sim_text, 4),
                matched_post_id=hit.matched_post_id,
                computed_at=utcnow(),
                note="adaptive_k_expanded" if hit.adaptive_expanded else None)
            with timer.stage("result_persist"):
                await self.store.mark_indexed(post_id, result)

            with timer.stage("faiss_add"):
                # 注册用 passage 前缀向量（库存侧契约，spec §5.1）
                await asyncio.to_thread(
                    self.index.add, post_id, rec.created_at, image_vec, passage_vec)

            self._log_complete(post_id, timer)
            return ProcessOutcome(result, None, dict(timer.stages), timer.total_ms())
        except Exception as e:
            code = getattr(e, "code", None)
            if code is None:  # 意外异常兜底：记 INTERNAL_ERROR，避免 pending 被无限重试
                code = ErrorCode.INTERNAL_ERROR
            await self.store.mark_failed(post_id, code)
            logger.error("post_id=%s phase=failed reason=%s %s",
                         post_id, code.value, timer.summary())
            if not raise_on_error:
                # demo 契约（spec 2026-08-24 附录 A-2）：失败不 raise，
                # 返回带已完成阶段耗时的 outcome；error_code 仅携带原始错误码
                # （意外异常为 None，意外异常统一归 internal_error）
                return ProcessOutcome(None, getattr(e, "code", None),
                                      dict(timer.stages), timer.total_ms())
            raise

    def _log_complete(self, post_id: str, timer: StageTimer):
        total = timer.total_ms()
        msg = f"post_id={post_id} phase=complete total={total:.0f}ms {timer.summary()}"
        if total > self.cfg.slow_task_warn_seconds * 1000:
            logger.warning("SLOW %s", msg)
        else:
            logger.info(msg)

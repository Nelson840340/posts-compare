"""Gradio demo 处理内核（spec 2026-08-24 §4 / 附录 A）。

进程内复用生产组件：同步提交 → 完整流水线 → 分数 + 分阶段耗时 + Top-3。
- 全局锁串行化提交（A-1）：IndexService 非线程安全，与生产单 Worker 不变式对齐
- 立即开门、后台装配（A-8）：ready=False 时提交返回"服务准备中"
- 失败不抛异常（A-2）：process(raise_on_error=False) 返回 outcome
"""
import asyncio
import base64
import io
import logging
import threading
import uuid
from dataclasses import dataclass, field

from PIL import Image

from app.config import Config
from app.domain import PostRecord, utcnow, window_cutoff
from app.embedder.base import Embedder
from app.index.service import IndexService
from app.pipeline.processor import ProcessOutcome, Processor
from app.store.base import PostStore
from app.store.sqlite_store import decode_vector

logger = logging.getLogger("demo")

# submit 内 asyncio.run 的超时上界（真模型 CPU 推理余量；测试可 monkeypatch 压缩）
_RUN_TIMEOUT = 600.0


class DemoValidationError(ValueError):
    """输入校验失败：UI 层捕获后原文展示。"""


@dataclass
class DemoPost:
    post_id: str
    text: str
    image: Image.Image | None
    note: str = ""
    sim_image: float = 0.0
    sim_text: float = 0.0


@dataclass
class DemoResult:
    post_id: str | None
    status: str               # not_ready | indexed | failed
    error: str | None
    outcome: ProcessOutcome | None
    top_posts: list[DemoPost] = field(default_factory=list)
    submitted_image: Image.Image | None = None


class DemoRunner:
    def __init__(self, cfg: Config, embedder: Embedder,
                 store: PostStore, index: IndexService):
        self.cfg = cfg
        self.embedder = embedder
        self.store = store
        self.index = index
        self.ready = False
        self.lock = threading.Lock()  # A-1：IndexService 串行化

    async def setup(self) -> None:
        """加载模型 + 打开库 + 从库重建索引（与生产 lifespan 同逻辑）。"""
        await asyncio.to_thread(self.embedder.load)
        await self.store.init()
        cutoff = window_cutoff(utcnow(), self.cfg.window_days)
        rows = await self.store.list_indexed_within(cutoff)
        self.index.rebuild([(pid, ts, decode_vector(iv), decode_vector(tv))
                            for pid, iv, tv, ts in rows])
        self.ready = True
        logger.info("demo 就绪：索引重建 %d 条", self.index.size)

    def submit(self, image_bytes: bytes | None, image_url: str | None,
               text: str) -> DemoResult:
        """同步提交：校验 → 串行执行流水线 → 组装结果。UI handler 直接调用。"""
        if not self.ready:
            return DemoResult(None, "not_ready", "服务准备中，请稍后", None)
        text = (text or "").strip()
        if image_bytes is not None and image_url:
            raise DemoValidationError("图片上传与图片 URL 请二选一")
        if image_bytes is None and not image_url and not text:
            raise DemoValidationError("请至少提供一张图片或一段文字")
        if len(text) > self.cfg.text_max_chars:
            raise DemoValidationError(f"文字超过 {self.cfg.text_max_chars} 字符上限")

        with self.lock:  # A-1：串行化全程（含 _finish 内的 top_hits，IndexService 非线程安全）
            post_id = f"demo-{uuid.uuid4().hex[:12]}"
            try:
                outcome = asyncio.run(asyncio.wait_for(
                    self._run(post_id, image_bytes, image_url, text),
                    timeout=_RUN_TIMEOUT))
            except TimeoutError:
                return DemoResult(post_id, "failed", "处理超时，请重试", None)
            return self._finish(post_id, outcome, image_bytes)

    async def _run(self, post_id: str, image_bytes: bytes | None,
                   image_url: str | None, text: str) -> ProcessOutcome:
        rec = PostRecord(
            post_id=post_id, text=text, image_url=image_url or None,
            image_base64=(base64.b64encode(image_bytes).decode()
                          if image_bytes else None),
            created_at=utcnow())
        if not await self.store.upsert_post(rec):
            raise DemoValidationError("post_id 冲突，请重试")
        proc = Processor(store=self.store, index=self.index,
                         embedder=self.embedder, cfg=self.cfg)
        return await proc.process(post_id, raise_on_error=False)

    def _finish(self, post_id: str, outcome: ProcessOutcome,
                image_bytes: bytes | None) -> DemoResult:
        submitted = None
        if image_bytes:
            try:
                submitted = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except Exception:
                submitted = None
        if outcome.result is None:
            code = outcome.error_code
            return DemoResult(post_id, "failed",
                              code.value if code else "internal_error",
                              outcome, top_posts=[], submitted_image=submitted)
        # Task 4 在此处补齐 Top-3 内容加载；第一版返回空列表
        return DemoResult(post_id, "indexed", None, outcome,
                          top_posts=[], submitted_image=submitted)

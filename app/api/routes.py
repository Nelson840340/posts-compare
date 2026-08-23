"""API 端点（spec §6）。enqueue 与 health_state 由 service 层注入。"""
import inspect
import logging
from datetime import datetime, timezone
from typing import Callable

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.schemas import COMPACT_SENTINEL, SubmitPostRequest, SubmitPostResponse
from app.config import Config
from app.domain import IndexStatus, PostRecord, utcnow
from app.pipeline.processor import Processor
from app.store.base import PostStore

logger = logging.getLogger("api")


def create_router(store: PostStore, processor: Processor, cfg: Config,
                  enqueue: Callable[[str], bool], health_state: dict) -> APIRouter:
    router = APIRouter()

    @router.post("/posts", status_code=202, response_model=SubmitPostResponse)
    async def submit_post(req: SubmitPostRequest):
        created_at = utcnow()
        if req.created_at:
            created_at = datetime.fromisoformat(req.created_at)
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
        rec = PostRecord(post_id=req.post_id, text=req.text,
                         image_url=req.image_url, image_base64=req.image_base64,
                         created_at=created_at)
        # ON CONFLICT DO NOTHING 原子判定：并发重复提交仅一个成功，避免 TOCTOU 多次入队
        if not await store.upsert_post(rec):
            existing = await store.get_post(req.post_id)
            return JSONResponse(status_code=200, content={
                "post_id": req.post_id, "status": existing.status.value})
        # enqueue 同步/异步兼容：生产为协程（service 层），测试桩为同步函数
        result = enqueue(req.post_id)
        if inspect.isawaitable(result):
            await result
        return SubmitPostResponse(post_id=req.post_id, status="pending")

    @router.get("/posts/{post_id}/similarity")
    async def get_similarity(post_id: str):
        post = await store.get_post(post_id)
        if post is None:
            return JSONResponse(status_code=404, content={
                "error": {"code": "not_found", "message": f"post_id={post_id}"}})
        if post.status == IndexStatus.PENDING:
            return JSONResponse(status_code=202, content={"status": "pending"})
        if post.status == IndexStatus.FAILED:
            return {"status": "failed", "reason": post.note}
        res = await store.get_result(post_id)
        return {
            "status": "indexed",
            "max_sim": res.max_sim, "sim_image": res.sim_image,
            "sim_text": res.sim_text, "matched_post_id": res.matched_post_id,
            "computed_at": res.computed_at.isoformat(), "note": res.note,
        }

    @router.get("/health")
    async def health():
        if not health_state.get("ready"):
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        counts = await store.counts()
        return {"status": "ok", "models_loaded": True,
                "index_count": counts["index_count"],
                "failed_count": counts["failed_count"],
                **health_state.get("live", {})}  # queue_size/oldest_pending_seconds 由 service 注入

    @router.post("/admin/replay")
    async def admin_replay(payload: dict | None = None):
        payload = payload or {}
        if payload.get("mode") == "compact":
            # IndexService 非线程安全：compact 入队经 Worker 串行执行（与 search/add 互斥）
            result = enqueue(COMPACT_SENTINEL)
            if inspect.isawaitable(result):
                await result
            return {"mode": "compact", "status": "scheduled"}
        pid = payload.get("post_id")
        if pid is not None:
            if not isinstance(pid, str):
                return JSONResponse(status_code=422, content={
                    "error": {"code": "validation_error", "message": "post_id 必须是字符串"}})
            ok = await store.reset_failed_to_pending(pid)
            if ok:
                result = enqueue(pid)
                if inspect.isawaitable(result):
                    await result
            return {"post_id": pid, "reset": ok}
        # 无参数：全量回放由 service 层触发（返回提示）
        return {"hint": "提供 post_id 或 mode=compact"}

    return router


def register_error_handlers(app: FastAPI) -> None:
    """统一错误格式 {"error": {code, message}}（spec §6）。必须在 app 层注册。"""

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={
            "error": {"code": "validation_error", "message": str(exc.errors()[:3])}})

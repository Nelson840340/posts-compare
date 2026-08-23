"""API 数据契约（spec §6）。"""
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

# 保留哨兵：Worker 识别后执行索引压缩（不可用作 post_id）
COMPACT_SENTINEL = "__compact__"


class SubmitPostRequest(BaseModel):
    post_id: str = Field(min_length=1, max_length=128)
    image_url: str | None = None
    image_base64: str | None = None
    text: str = Field(min_length=1, max_length=5000)
    created_at: str | None = None  # ISO8601，可选（spec §3.4⑧）

    @field_validator("post_id")
    @classmethod
    def _reserved_sentinel(cls, v: str) -> str:
        if v == COMPACT_SENTINEL:
            raise ValueError("post_id 为保留字")
        return v

    @field_validator("created_at")
    @classmethod
    def _iso8601(cls, v: str | None) -> str | None:
        if v is not None:
            try:
                datetime.fromisoformat(v)
            except ValueError as e:
                raise ValueError("created_at 必须是 ISO8601 格式") from e
        return v


class SubmitPostResponse(BaseModel):
    post_id: str
    status: str


class SimilarityResponse(BaseModel):
    status: str
    max_sim: float | None = None
    sim_image: float | None = None
    sim_text: float | None = None
    matched_post_id: str | None = None
    computed_at: str | None = None
    note: str | None = None
    reason: str | None = None


class ErrorResponse(BaseModel):
    error: dict

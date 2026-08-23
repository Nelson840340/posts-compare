"""领域模型与错误分类（spec §4.3/§6/§7.2）。"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum


class IndexStatus(str, Enum):
    PENDING = "pending"
    INDEXED = "indexed"
    FAILED = "failed"


class ErrorCode(str, Enum):
    IMAGE_DOWNLOAD_FAILED = "image_download_failed"
    IMAGE_DECODE_FAILED = "image_decode_failed"
    IMAGE_TOO_LARGE = "image_too_large"
    ENCODE_FAILED = "encode_failed"
    INTERNAL_ERROR = "internal_error"


class ProcessingError(Exception):
    code: ErrorCode = ErrorCode.INTERNAL_ERROR


class ImageDownloadError(ProcessingError):
    code = ErrorCode.IMAGE_DOWNLOAD_FAILED


class ImageDecodeError(ProcessingError):
    code = ErrorCode.IMAGE_DECODE_FAILED


class ImageTooLargeError(ProcessingError):
    code = ErrorCode.IMAGE_TOO_LARGE


class EncodeError(ProcessingError):
    code = ErrorCode.ENCODE_FAILED


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def window_cutoff(now: datetime, window_days: int) -> datetime:
    return now - timedelta(days=window_days)


@dataclass
class PostRecord:
    post_id: str
    text: str
    image_url: str | None = None
    image_base64: str | None = None
    created_at: datetime = field(default_factory=utcnow)
    enqueued_at: datetime = field(default_factory=utcnow)
    status: IndexStatus = IndexStatus.PENDING
    note: str | None = None


@dataclass
class SimilarityResult:
    post_id: str
    max_sim: float
    sim_image: float
    sim_text: float
    matched_post_id: str | None
    computed_at: datetime
    note: str | None = None

from datetime import datetime, timedelta, timezone

import pytest

from app.domain import (
    ErrorCode, ImageDecodeError, ImageDownloadError, IndexStatus,
    PostRecord, ProcessingError, SimilarityResult, window_cutoff,
)


def test_status_values_match_api_contract():
    assert IndexStatus.PENDING.value == "pending"
    assert IndexStatus.INDEXED.value == "indexed"
    assert IndexStatus.FAILED.value == "failed"


def test_processing_error_carries_code():
    err = ImageDownloadError("404")
    assert isinstance(err, ProcessingError)
    assert err.code == ErrorCode.IMAGE_DOWNLOAD_FAILED
    assert ImageDecodeError("bad").code == ErrorCode.IMAGE_DECODE_FAILED


def test_window_cutoff():
    now = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)
    assert window_cutoff(now, 30) == now - timedelta(days=30)


def test_post_record_defaults_utc_now():
    rec = PostRecord(post_id="p1", text="t")
    assert rec.status == IndexStatus.PENDING
    assert rec.created_at.tzinfo is not None  # 必须带时区
    assert rec.enqueued_at.tzinfo is not None


def test_similarity_result_fields():
    r = SimilarityResult(post_id="p1", max_sim=0.9123, sim_image=0.9123,
                         sim_text=0.621, matched_post_id="p0",
                         computed_at=datetime.now(timezone.utc), note=None)
    assert r.matched_post_id == "p0"

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app.index.service import IndexService

NOW = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)


def _vec(dim, seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def svc():
    return IndexService()


def test_empty_index_returns_zero(svc):
    hit = svc.search(_vec(512, 1), _vec(384, 2), exclude_id="x", cutoff=NOW - timedelta(days=30),
                     top_k=16, adaptive_k_steps=(64, 256))
    assert hit.sim_image == 0.0 and hit.sim_text == 0.0 and hit.matched_post_id is None


def test_exact_match_detected(svc):
    iv, tv = _vec(512, 1), _vec(384, 2)
    svc.add("p0", NOW - timedelta(days=1), iv, tv)
    hit = svc.search(iv, tv, exclude_id="x", cutoff=NOW - timedelta(days=30),
                     top_k=16, adaptive_k_steps=(64, 256))
    assert hit.sim_image == pytest.approx(1.0, abs=1e-5)
    assert hit.matched_post_id == "p0"


def test_self_excluded(svc):
    iv, tv = _vec(512, 1), _vec(384, 2)
    svc.add("p0", NOW - timedelta(days=1), iv, tv)
    hit = svc.search(iv, tv, exclude_id="p0", cutoff=NOW - timedelta(days=30),
                     top_k=16, adaptive_k_steps=(64, 256))
    assert hit.matched_post_id is None  # 自身被排除


def test_window_filter_and_max_of_two_signals(svc):
    iv, tv = _vec(512, 1), _vec(384, 2)
    svc.add("expired", NOW - timedelta(days=40), iv, tv)          # 窗口外
    other_iv = _vec(512, 99)
    svc.add("fresh", NOW - timedelta(days=1), other_iv, tv)       # 窗口内，仅文字命中
    hit = svc.search(iv, tv, exclude_id="x", cutoff=NOW - timedelta(days=30),
                     top_k=16, adaptive_k_steps=(64, 256))
    assert hit.matched_post_id == "fresh"
    assert hit.sim_text == pytest.approx(1.0, abs=1e-5)
    assert hit.sim_image < 1.0


def test_adaptive_k_when_topk_all_expired(svc):
    """spec §5.2 自适应K：Top-K 被过期副本占满时扩 K 重查。"""
    query = _vec(512, 1)
    # 16 个过期的高相似副本（完全相同向量 → 必然占满 Top-16）
    for i in range(16):
        svc.add(f"stale{i}", NOW - timedelta(days=40), query, _vec(384, 100 + i))
    # 1 个窗口内的相同向量
    svc.add("fresh", NOW - timedelta(days=1), query, _vec(384, 999))
    hit = svc.search(query, _vec(384, 999), exclude_id="x",
                     cutoff=NOW - timedelta(days=30),
                     top_k=16, adaptive_k_steps=(64, 256))
    assert hit.adaptive_expanded is True
    assert hit.matched_post_id == "fresh"
    assert hit.sim_image == pytest.approx(1.0, abs=1e-5)


def test_adaptive_k_gives_up_at_cap(svc):
    query = _vec(512, 1)
    for i in range(300):  # 超过最大 K=256，全部过期
        svc.add(f"stale{i}", NOW - timedelta(days=40), query, _vec(384, i))
    hit = svc.search(query, _vec(384, 1), exclude_id="x",
                     cutoff=NOW - timedelta(days=30),
                     top_k=16, adaptive_k_steps=(64, 256))
    assert hit.matched_post_id is None and hit.sim_image == 0.0


def test_rebuild_replaces_index(svc):
    svc.add("p0", NOW, _vec(512, 1), _vec(384, 1))
    svc.rebuild([("p1", NOW, _vec(512, 2), _vec(384, 2))])
    assert svc.size == 1
    hit = svc.search(_vec(512, 2), _vec(384, 2), exclude_id="x",
                     cutoff=NOW - timedelta(days=30), top_k=16, adaptive_k_steps=(64, 256))
    assert hit.matched_post_id == "p1"


def test_compact_removes_expired(svc):
    svc.add("old", NOW - timedelta(days=40), _vec(512, 1), _vec(384, 1))
    svc.add("new", NOW - timedelta(days=1), _vec(512, 2), _vec(384, 2))
    removed = svc.compact(NOW, window_days=30)
    assert removed == 1 and svc.size == 1


def test_negative_sims_yield_no_matched(svc):
    """双路内积均为负时 clamp 到 0，matched 必须 None（审查 Minor-2：文字分支防护）。"""
    iv, tv = _vec(512, 1), _vec(384, 2)
    # 构造文字路内积 ≈ -0.707 > 图片路 -1，强制走文字分支
    orth = _vec(384, 77)
    orth = orth - tv * float(tv @ orth)
    stored_tv = -tv + orth / np.linalg.norm(orth)
    stored_tv = (stored_tv / np.linalg.norm(stored_tv)).astype(np.float32)
    svc.add("p0", NOW - timedelta(days=1), (-iv).astype(np.float32), stored_tv)
    hit = svc.search(iv, tv, exclude_id="x", cutoff=NOW - timedelta(days=30),
                     top_k=16, adaptive_k_steps=(64, 256))
    assert hit.sim_image == 0.0 and hit.sim_text == 0.0
    assert hit.matched_post_id is None

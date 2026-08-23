"""IndexService：Faiss 双索引检索（spec §3.4/§5.2）。

- IndexFlatIP + IndexIDMap2：归一化向量内积 = 余弦相似度
- 30 天窗口与自排除在检索后按元数据过滤（spec §3.4①）
- 自适应 K：Top-K 过滤后全空且确有命中时逐级扩 K 重查（spec §5.2 grill-me 决议3）
- 非线程安全：search/add/rebuild/compact 必须由调用方串行化
  （当前由单 Worker 协程顺序 await to_thread 保证；compact 触发点同样须串行）
"""
from dataclasses import dataclass
from datetime import datetime, timedelta

import faiss
import numpy as np

from app.embedder.base import IMAGE_DIM, TEXT_DIM


@dataclass
class SearchHit:
    sim_image: float
    sim_text: float
    matched_post_id: str | None
    adaptive_expanded: bool


class _SingleIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self.meta: dict[int, tuple[str, datetime]] = {}
        self._next_id = 0

    def add(self, post_id: str, created_at: datetime, vec: np.ndarray):
        fid = self._next_id
        self._next_id += 1
        self.index.add_with_ids(np.asarray([vec], dtype=np.float32), np.array([fid]))
        self.meta[fid] = (post_id, created_at)

    def rebuild(self, rows):
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))
        self.meta.clear()
        self._next_id = 0
        for post_id, created_at, vec in rows:
            self.add(post_id, created_at, vec)

    def compact(self, cutoff: datetime) -> int:
        keep = [fid for fid, (_, ts) in self.meta.items() if ts >= cutoff]
        removed = len(self.meta) - len(keep)
        if removed:
            vecs = self.index.reconstruct_batch(keep) if keep else np.zeros((0, self.dim), np.float32)
            metas = [(self.meta[f]) for f in keep]
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))
            self.meta.clear()
            self._next_id = 0
            for (post_id, created_at), vec in zip(metas, vecs):
                self.add(post_id, created_at, vec)
        return removed

    def query(self, vec: np.ndarray, k: int, exclude_id: str, cutoff: datetime):
        """返回 (最佳命中元组|None, 是否有过期/自身命中挤占候选)。"""
        n = self.index.ntotal
        if n == 0:
            return None, False
        k = min(k, n)
        sims, ids = self.index.search(np.asarray([vec], dtype=np.float32), k)
        polluted = False  # 候选中存在被过滤者（过期/自身）
        for sim, fid in zip(sims[0], ids[0]):
            if fid < 0:
                continue
            post_id, created_at = self.meta[int(fid)]
            if post_id == exclude_id or created_at < cutoff:
                polluted = True
                continue
            return (float(sim), post_id), polluted
        return None, polluted


class IndexService:
    def __init__(self, image_dim: int = IMAGE_DIM, text_dim: int = TEXT_DIM):
        self._img = _SingleIndex(image_dim)
        self._txt = _SingleIndex(text_dim)

    @property
    def size(self) -> int:
        return self._img.index.ntotal

    def add(self, post_id, created_at, image_vec, text_vec):
        self._img.add(post_id, created_at, image_vec)
        self._txt.add(post_id, created_at, text_vec)

    def rebuild(self, rows):
        """rows: [(post_id, created_at, image_vec, text_vec)]。启动重建/压缩共用。"""
        self._img.rebuild([(p, ts, iv) for p, ts, iv, _ in rows])
        self._txt.rebuild([(p, ts, tv) for p, ts, _, tv in rows])

    def compact(self, now: datetime, window_days: int) -> int:
        cutoff = now - timedelta(days=window_days)
        removed = self._img.compact(cutoff)
        self._txt.compact(cutoff)  # 双索引同步剔除（add 成对调用保证条目一致）
        return removed

    def search(self, image_vec, text_vec, *, exclude_id, cutoff,
               top_k, adaptive_k_steps) -> SearchHit:
        expanded = False
        img_hit, img_polluted = self._img.query(image_vec, top_k, exclude_id, cutoff)
        txt_hit, txt_polluted = self._txt.query(text_vec, top_k, exclude_id, cutoff)
        # 自适应 K：有命中但全被过滤 → 逐级扩 K 重查（spec §5.2）
        for k in adaptive_k_steps:
            if not ((img_hit is None and img_polluted) or (txt_hit is None and txt_polluted)):
                break
            expanded = True
            if img_hit is None and img_polluted:
                img_hit, img_polluted = self._img.query(image_vec, k, exclude_id, cutoff)
            if txt_hit is None and txt_polluted:
                txt_hit, txt_polluted = self._txt.query(text_vec, k, exclude_id, cutoff)
        sim_image = img_hit[0] if img_hit else 0.0
        sim_text = txt_hit[0] if txt_hit else 0.0
        # matched_post_id 取 max 信号方；并列时优先图片信号；零分/负分不带匹配对象
        if sim_image >= sim_text:
            matched = img_hit[1] if img_hit and sim_image > 0 else None
        else:
            matched = txt_hit[1] if txt_hit and sim_text > 0 else None
        return SearchHit(sim_image=max(sim_image, 0.0), sim_text=max(sim_text, 0.0),
                         matched_post_id=matched, adaptive_expanded=expanded)

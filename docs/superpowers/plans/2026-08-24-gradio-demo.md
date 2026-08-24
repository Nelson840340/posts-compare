# Gradio 交互演示（Demo）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为帖子相似度系统提供 Gradio 网页 Demo：上传图片/输入文字提交新帖，页面展示综合与分模态相似度分数、流水线各阶段耗时、Top-3 相似帖内容。

**Architecture:** 进程内直接复用生产组件（SqliteStore/IndexService/Embedder/Processor），独立入口 `demo/app.py`；业务逻辑收敛在可测的 `DemoRunner`（全局锁串行 + `asyncio.run` 桥接），Gradio UI 只做渲染与事件绑定。对生产代码仅两处最小侵入：`Processor.process` 增加 `raise_on_error` 开关返回 `ProcessOutcome`，`IndexService` 新增 `top_hits()`。

**Tech Stack:** Python 3.11+ / Gradio（dev 组）/ 既有 FastAPI 项目的 app 包组件 / pytest(asyncio_mode=auto) / Poetry

**Spec:** `docs/superpowers/specs/2026-08-24-gradio-demo-design.md`（含附录 A 八项 grill-me 决议，任务中以 A-1~A-8 引用）

## Global Constraints

- 当前分支 `feat/gradio-demo`，所有提交落在该分支；提交信息使用 conventional 前缀（feat/test/docs/chore）。
- 生产 worker 行为逐字节不变：`process()` 默认 `raise_on_error=True`，失败仍 `mark_failed` 后 raise；`search()` 不改动。
- demo post_id 格式：`demo-{uuid4.hex[:12]}`；demo 帖直接落生产库 `data/similarity.db`（默认 `cfg.db_path`），**不做清理脚本（A-7）**。
- 输入图片上传与图片 URL 二选一，都填抛错（A-5）；URL 提交走生产 `fetch_image` 完整下载链路（含重试退避）。
- Top-3 渲染图片轻量下载：单次请求、3 秒超时、不重试，失败返回 None（A-4）。
- `top_hits()` 不做自适应扩 K：双模态各 `candidate_k`（默认 `k*2`=6）条候选，不足按实际条数（A-3）。
- 并发：`DemoRunner.submit` 全程持全局 `threading.Lock`（A-1）；启动立即开门，后台装配，未就绪提交返回"服务准备中，请稍后"（A-8）。
- 阈值标档文案从 `cfg.hard_downweight_threshold` / `cfg.soft_downweight_threshold` 动态读取，注明判定权在下游（A-6）。
- Gradio 只进 `[dependency-groups] dev`；demo 监听 `127.0.0.1:7860`。
- 测试命令统一 `poetry run pytest`（pyproject 已配 `-m 'not slow'`、asyncio_mode=auto，异步测试直接写 `async def`）。

## 文件结构

| 文件 | 动作 | 职责 |
|---|---|---|
| `app/pipeline/processor.py` | 修改 | 新增 `ProcessOutcome` 数据类；`process()` 增加 `raise_on_error` 参数与 `result_persist` 计时点 |
| `tests/unit/test_processor.py` | 修改 | 追加 outcome 契约测试 |
| `app/index/service.py` | 修改 | 新增 `TopHit` 数据类与 `IndexService.top_hits()`、`_SingleIndex.topk()` |
| `tests/unit/test_index_service.py` | 修改 | 追加 top_hits 测试 |
| `demo/__init__.py` | 创建 | 使 `demo` 成为可导入包（测试导入 `demo.runner` 所需） |
| `demo/runner.py` | 创建 | `DemoRunner`（装配/就绪/串行提交）、`DemoValidationError`、`DemoPost`、`DemoResult`、`fetch_top_image`、`tier_text`、`_RUN_TIMEOUT` |
| `tests/demo/__init__.py` | 创建 | 测试包标记 |
| `tests/demo/test_runner.py` | 创建 | runner 就绪/校验/端到端/并发/启动重建/Top-3 内容加载测试 |
| `demo/app.py` | 创建 | Gradio UI：布局、handler、耗时表标签映射、真实模型装配与后台加载线程 |
| `tests/demo/test_ui.py` | 创建 | 纯函数（标签映射/tier 文案）测试 + build_ui 冒烟 |
| `pyproject.toml` | 修改 | dev 组加 gradio |
| `README.md` | 修改 | 新增"Gradio Demo"章节（启动方式与运行约束） |

---

### Task 1: Processor 返回 ProcessOutcome + result_persist 计时点

**Files:**
- Modify: `app/pipeline/processor.py`
- Test: `tests/unit/test_processor.py`

**Interfaces:**
- Consumes: 现有 `StageTimer`（`app/timer.py`）、`SimilarityResult`/`ErrorCode`（`app/domain.py`）
- Produces:
  - `ProcessOutcome`（定义于 `app/pipeline/processor.py`）：`result: SimilarityResult | None`、`error_code: ErrorCode | None`、`timings_ms: dict[str, float]`、`total_ms: float`
  - `Processor.process(post_id: str, *, raise_on_error: bool = True) -> ProcessOutcome`
  - 新增 timer 阶段键 `result_persist`；成功路径 timings 键集合：`download`（有图时）、`image_embed`（有图时）、`text_embed`、`persist`、`search`、`result_persist`、`faiss_add`

- [ ] **Step 1: Write the failing tests**

在 `tests/unit/test_processor.py` 末尾追加（`ProcessOutcome` 从 `app.pipeline.processor` 导入，加入文件顶部 import）：

```python
async def test_process_returns_outcome_with_stage_timings(env):
    _, store, index, proc = env
    await _submit(store, "p1", _png(), "第一帖", NOW)
    outcome = await proc.process("p1")
    assert isinstance(outcome, ProcessOutcome)
    assert outcome.result is not None and outcome.error_code is None
    assert set(outcome.timings_ms) == {
        "download", "image_embed", "text_embed", "persist",
        "search", "result_persist", "faiss_add"}
    assert all(v >= 0.0 for v in outcome.timings_ms.values())
    assert outcome.total_ms > 0


async def test_process_failure_returns_outcome_when_raise_disabled(env):
    _, store, index, proc = env
    rec = PostRecord(post_id="bad", image_base64=base64.b64encode(b"broken").decode(),
                     text="坏图", created_at=NOW)
    await store.upsert_post(rec)
    outcome = await proc.process("bad", raise_on_error=False)
    assert outcome.result is None
    assert outcome.error_code.value == "image_decode_failed"
    assert "download" in outcome.timings_ms  # 失败前已完成阶段的耗时保留
    post = await store.get_post("bad")
    assert post.status == IndexStatus.FAILED


async def test_process_default_still_raises(env):
    """raise_on_error 默认 True：生产 worker 的异常契约逐字节不变。"""
    _, store, index, proc = env
    rec = PostRecord(post_id="bad", image_base64=base64.b64encode(b"broken").decode(),
                     text="坏图", created_at=NOW)
    await store.upsert_post(rec)
    from app.domain import ProcessingError
    with pytest.raises(ProcessingError):
        await proc.process("bad")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/unit/test_processor.py -v -k outcome`
Expected: FAIL —— `ProcessOutcome` 不存在导致 ImportError/NameError。

- [ ] **Step 3: Implement**

`app/pipeline/processor.py` 修改点：

① 顶部 import：`from dataclasses import dataclass`；`from app.domain import ...` 中补充 `ErrorCode`。

② 在 `logger = ...` 之后、`class Processor` 之前新增：

```python
@dataclass
class ProcessOutcome:
    """单帖处理结果（spec 2026-08-24 §3.1 / 附录 A-2）：供 demo 同步展示分数与分阶段耗时。"""
    result: SimilarityResult | None
    error_code: ErrorCode | None      # 带 code 的失败原因；意外异常（无 code）为 None
    timings_ms: dict[str, float]
    total_ms: float
```

③ 方法签名改为 `async def process(self, post_id: str, *, raise_on_error: bool = True) -> ProcessOutcome:`，方法体改为如下完整版本（原有顺序铁律、幂等守卫、超窗短路、各阶段计时全部保留，仅新增 result_persist 计时、outcome 构造与失败路径分支）：

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/unit/test_processor.py -v`
Expected: 全部 PASS（新增 3 个 + 原有 9 个；原有测试验证生产异常契约未破坏）。

- [ ] **Step 5: Run full regression**

Run: `poetry run pytest -q`
Expected: 全量 PASS（service/api/integration 测试不受返回值新增影响）。

- [ ] **Step 6: Commit**

```bash
git add app/pipeline/processor.py tests/unit/test_processor.py
git commit -m "feat: Processor.process 返回 ProcessOutcome 并新增 result_persist 计时点"
```

---

### Task 2: IndexService.top_hits()

**Files:**
- Modify: `app/index/service.py`
- Test: `tests/unit/test_index_service.py`

**Interfaces:**
- Consumes: 既有 `_SingleIndex`（faiss IndexIDMap2 + meta）、`SearchHit`
- Produces:
  - `TopHit(post_id: str, sim_image: float, sim_text: float)`
  - `IndexService.top_hits(image_vec, text_vec, *, exclude_id: str, cutoff: datetime, k: int = 3, candidate_k: int | None = None) -> list[TopHit]`（按 `max(sim_image, sim_text)` 降序，`candidate_k` 缺省为 `k * 2`，不足按实际条数，A-3）
  - `_SingleIndex.topk(vec, k, exclude_id, cutoff) -> list[tuple[float, str]]`（已过滤的 (sim, post_id) 列表，保持 faiss 返回的降序）

- [ ] **Step 1: Write the failing tests**

在 `tests/unit/test_index_service.py` 末尾追加（`TopHit` 加入文件顶部 `from app.index.service import ...`）：

```python
def _search_all(svc, iv, tv, **kw):
    args = dict(exclude_id="x", cutoff=NOW - timedelta(days=30), k=3)
    args.update(kw)
    return svc.top_hits(iv, tv, **args)


def test_top_hits_empty_index(svc):
    assert _search_all(svc, _vec(512, 1), _vec(384, 2)) == []


def test_top_hits_merges_modality_scores(svc):
    """同一候选同时出现在双索引：合并后携带两个模态分数，按 max 降序。"""
    iv, tv = _vec(512, 1), _vec(384, 2)
    svc.add("a", NOW - timedelta(days=1), iv, _vec(384, 3))          # 图片完全命中
    svc.add("b", NOW - timedelta(days=1), _vec(512, 4), tv)          # 文字完全命中
    svc.add("c", NOW - timedelta(days=1), _vec(512, 5), _vec(384, 6))
    hits = _search_all(svc, iv, tv)
    ids = [h.post_id for h in hits]
    assert set(ids[:2]) == {"a", "b"}
    top = {h.post_id: h for h in hits}
    assert top["a"].sim_image == pytest.approx(1.0, abs=1e-5)
    assert top["b"].sim_text == pytest.approx(1.0, abs=1e-5)
    assert top["c"].sim_image < 1.0 and top["c"].sim_text < 1.0


def test_top_hits_self_excluded(svc):
    iv, tv = _vec(512, 1), _vec(384, 2)
    svc.add("self", NOW - timedelta(days=1), iv, tv)
    svc.add("other", NOW - timedelta(days=1), _vec(512, 9), _vec(384, 10))
    hits = svc.top_hits(iv, tv, exclude_id="self", cutoff=NOW - timedelta(days=30), k=3)
    assert "self" not in [h.post_id for h in hits]
    assert len(hits) == 1


def test_top_hits_window_filtered(svc):
    iv, tv = _vec(512, 1), _vec(384, 2)
    svc.add("expired", NOW - timedelta(days=40), iv, tv)
    svc.add("fresh", NOW - timedelta(days=1), _vec(512, 9), tv)
    hits = _search_all(svc, iv, tv)
    assert [h.post_id for h in hits] == ["fresh"]
    assert hits[0].sim_text == pytest.approx(1.0, abs=1e-5)


def test_top_hits_negative_clamped_to_zero(svc):
    """负内积归零（与 search 一致），候选保留但分数为 0。"""
    iv, tv = _vec(512, 1), _vec(384, 2)
    svc.add("neg", NOW - timedelta(days=1), (-iv).astype(np.float32),
            (-tv).astype(np.float32))
    hits = _search_all(svc, iv, tv)
    assert len(hits) == 1
    assert hits[0].sim_image == 0.0 and hits[0].sim_text == 0.0


def test_top_hits_shorter_than_k(svc):
    """候选不足 k：按实际条数返回，不补齐不报错（A-3）。"""
    svc.add("p0", NOW - timedelta(days=1), _vec(512, 1), _vec(384, 1))
    hits = _search_all(svc, _vec(512, 2), _vec(384, 2))
    assert len(hits) == 1


def test_top_hits_k_limits_count(svc):
    for i in range(5):
        svc.add(f"p{i}", NOW - timedelta(days=1), _vec(512, 10 + i), _vec(384, 10 + i))
    hits = svc.top_hits(_vec(512, 1), _vec(384, 1), exclude_id="x",
                        cutoff=NOW - timedelta(days=30), k=2)
    assert len(hits) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/unit/test_index_service.py -v -k top_hits`
Expected: FAIL —— `top_hits` 属性不存在。

- [ ] **Step 3: Implement**

`app/index/service.py`：

① 在 `SearchHit` 之后新增：

```python
@dataclass
class TopHit:
    """Top-K 展示候选（spec 2026-08-24 §3.2）：双模态分数合并。"""
    post_id: str
    sim_image: float
    sim_text: float
```

② `_SingleIndex` 内、`query` 方法之后新增：

```python
    def topk(self, vec: np.ndarray, k: int, exclude_id: str, cutoff: datetime):
        """返回过滤后的全部候选 [(sim, post_id)]，保持相似度降序。"""
        n = self.index.ntotal
        if n == 0:
            return []
        k = min(k, n)
        sims, ids = self.index.search(np.asarray([vec], dtype=np.float32), k)
        out = []
        for sim, fid in zip(sims[0], ids[0]):
            if fid < 0:
                continue
            post_id, created_at = self.meta[int(fid)]
            if post_id == exclude_id or created_at < cutoff:
                continue
            out.append((float(sim), post_id))
        return out
```

③ `IndexService` 内、`search` 方法之后新增：

```python
    def top_hits(self, image_vec, text_vec, *, exclude_id, cutoff,
                 k: int = 3, candidate_k: int | None = None) -> list[TopHit]:
        """Top-K 展示检索（spec 2026-08-24 §3.2 / 附录 A-3）。

        双索引各取 candidate_k（默认 k*2）候选，按 post_id 合并双模态分数，
        按 max 降序取前 k。不做自适应扩 K：挤占场景不足 k 条按实际返回，
        仅服务 demo 展示，判重检测仍走 search()。
        """
        ck = candidate_k if candidate_k is not None else k * 2
        merged: dict[str, list[float]] = {}
        for sim, pid in self._img.topk(image_vec, ck, exclude_id, cutoff):
            merged.setdefault(pid, [0.0, 0.0])[0] = max(sim, 0.0)
        for sim, pid in self._txt.topk(text_vec, ck, exclude_id, cutoff):
            merged.setdefault(pid, [0.0, 0.0])[1] = max(sim, 0.0)
        hits = [TopHit(pid, si, st) for pid, (si, st) in merged.items()]
        hits.sort(key=lambda h: max(h.sim_image, h.sim_text), reverse=True)
        return hits[:k]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/unit/test_index_service.py -v`
Expected: 全部 PASS（新增 7 个 + 原有 9 个）。

- [ ] **Step 5: Commit**

```bash
git add app/index/service.py tests/unit/test_index_service.py
git commit -m "feat: IndexService 新增 top_hits() 双模态合并 Top-K 检索"
```

---

### Task 3: DemoRunner 核心（装配/就绪/校验/串行提交）

**Files:**
- Create: `demo/__init__.py`（空文件，使 demo 成为可导入包）
- Create: `demo/runner.py`
- Create: `tests/demo/__init__.py`（空文件）
- Create: `tests/demo/test_runner.py`

**Interfaces:**
- Consumes: `Processor.process(post_id, *, raise_on_error=False) -> ProcessOutcome`（Task 1）、`SqliteStore`、`IndexService`、`Embedder` 协议、`PostRecord`、`fetch_image`/`decode_and_validate`、`window_cutoff`/`utcnow`/`decode_vector`
- Produces:
  - `demo.runner.DemoValidationError`（ValueError 子类，UI 层捕获后原文展示）
  - `demo.runner.DemoPost(post_id, text, image: PIL.Image | None, note: str = "", sim_image: float = 0.0, sim_text: float = 0.0)`
  - `demo.runner.DemoResult(post_id, status, error, outcome: ProcessOutcome | None, top_posts: list[DemoPost], submitted_image: PIL.Image | None)`
  - `demo.runner.DemoRunner(cfg, embedder, store, index)`：`.ready: bool`、`.lock: threading.Lock`、`async setup() -> None`、`submit(image_bytes: bytes | None, image_url: str | None, text: str) -> DemoResult`（同步，内部 `asyncio.run` + 全局锁）
  - `demo.runner._RUN_TIMEOUT = 600.0`（submit 内 asyncio.run 的超时上界，测试用 monkeypatch 压缩）

- [ ] **Step 1: Create package markers**

创建空文件 `demo/__init__.py` 与 `tests/demo/__init__.py`（内容均为空）。

- [ ] **Step 2: Write the failing tests**

`tests/demo/test_runner.py`：

```python
import threading
from datetime import datetime, timezone

import pytest
from PIL import Image

from app.config import Config
from app.domain import PostRecord, utcnow
from app.embedder.fake import FakeEmbedder
from app.index.service import IndexService
from app.pipeline.processor import Processor
from app.store.sqlite_store import SqliteStore
from demo.runner import DemoResult, DemoRunner, DemoValidationError

NOW = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)


def _png(color=(9, 9, 9)) -> bytes:
    import io
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
async def runner(tmp_db):
    cfg = Config(db_path=tmp_db)
    store = SqliteStore(tmp_db)
    r = DemoRunner(cfg, FakeEmbedder(), store, IndexService())
    await r.setup()
    yield r
    await store.close()


async def test_submit_before_ready_rejected(tmp_db):
    """A-8：未就绪提交返回明确提示，不抛异常。"""
    cfg = Config(db_path=tmp_db)
    store = SqliteStore(tmp_db)
    r = DemoRunner(cfg, FakeEmbedder(), store, IndexService())
    result = r.submit(_png(), None, "hello")
    assert result.status == "not_ready"
    assert "服务准备中" in (result.error or "")
    await store.close()


async def test_setup_rebuilds_index_from_db(tmp_db):
    """A-8 装配链：从库重建索引（SQLite 唯一事实源不变式）。"""
    cfg = Config(db_path=tmp_db)
    store = SqliteStore(tmp_db)
    r1 = DemoRunner(cfg, FakeEmbedder(), store, IndexService())
    await r1.setup()
    result = r1.submit(_png(), None, "种子帖")
    assert result.status == "indexed"
    # 模拟重启：新 runner 从同一库重建
    r2 = DemoRunner(cfg, FakeEmbedder(), store, IndexService())
    await r2.setup()
    assert r2.index.size == 1
    assert r2.ready is True
    await store.close()


async def test_both_image_and_url_rejected(runner):
    with pytest.raises(DemoValidationError, match="二选一"):
        runner.submit(_png(), "https://example.com/a.jpg", "t")


async def test_empty_content_rejected(runner):
    with pytest.raises(DemoValidationError, match="至少提供"):
        runner.submit(None, None, "")


async def test_text_too_long_rejected(runner):
    runner.cfg.text_max_chars = 10
    with pytest.raises(DemoValidationError, match="字符上限"):
        runner.submit(None, None, "x" * 11)


async def test_e2e_first_post_and_duplicate(runner):
    r1 = runner.submit(_png((9, 9, 9)), None, "第一帖")
    assert r1.status == "indexed"
    assert r1.post_id.startswith("demo-")
    assert r1.outcome is not None and r1.outcome.result is not None
    assert set(r1.outcome.timings_ms) == {
        "download", "image_embed", "text_embed", "persist",
        "search", "result_persist", "faiss_add"}
    assert r1.submitted_image is not None  # 原图回显
    # 同图重发：图片通道高相似
    r2 = runner.submit(_png((9, 9, 9)), None, "换个文案重发")
    res = r2.outcome.result
    assert res.sim_image > 0.99
    assert res.matched_post_id == r1.post_id


async def test_failure_returns_outcome_not_exception(runner):
    """坏图：不抛异常，status=failed 且携带 error_code 与部分耗时（A-2）。"""
    result = runner.submit(b"not-an-image", None, "坏图")
    assert result.status == "failed"
    assert result.outcome is not None
    assert result.outcome.error_code.value == "image_decode_failed"
    assert "download" in result.outcome.timings_ms
    assert result.error == "image_decode_failed"


async def test_run_timeout_reported(runner, monkeypatch):
    import demo.runner as mod
    monkeypatch.setattr(mod, "_RUN_TIMEOUT", 0.0001)
    result = runner.submit(_png(), None, "慢帖")
    assert result.status == "failed"
    assert "超时" in (result.error or "")


async def test_submit_serialized_by_lock(runner, monkeypatch):
    """A-1：并发 submit 被全局锁串行化——锁内区间（_run）任意时刻最多一个在途。"""
    from app.pipeline.processor import Processor

    events = []
    evlock = threading.Lock()
    original_process = Processor.process

    async def spy_process(self, post_id, *, raise_on_error=True):
        # process 全程在 DemoRunner 锁内执行；探针重叠即锁失效
        with evlock:
            events.append("enter")
        try:
            return await original_process(self, post_id, raise_on_error=raise_on_error)
        finally:
            with evlock:
                events.append("exit")

    monkeypatch.setattr(Processor, "process", spy_process)

    results, errors = [], []

    def worker():
        try:
            results.append(runner.submit(_png(), None, "并发帖"))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(results) == 3 and all(r.status == "indexed" for r in results)
    assert len(set(r.post_id for r in results)) == 3
    depth = peak = 0
    for ev in events:
        depth += 1 if ev == "enter" else -1
        peak = max(peak, depth)
    assert peak == 1  # 锁内全程无重叠
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `poetry run pytest tests/demo/ -v`
Expected: FAIL —— `demo.runner` 不存在（ModuleNotFoundError）。

- [ ] **Step 4: Implement demo/runner.py（第一版：不含 Top-3 内容加载，top_posts 恒为空，Task 4 补齐）**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `poetry run pytest tests/demo/ -v`
Expected: 全部 PASS（9 个）。

- [ ] **Step 6: Run full regression**

Run: `poetry run pytest -q`
Expected: 全量 PASS。

- [ ] **Step 7: Commit**

```bash
git add demo/__init__.py demo/runner.py tests/demo/__init__.py tests/demo/test_runner.py
git commit -m "feat: DemoRunner 核心——就绪装配、输入校验、锁串行同步提交"
```

---

### Task 4: Top-3 内容加载 + 轻量图片下载 + 阈值标档

**Files:**
- Modify: `demo/runner.py`
- Modify: `tests/demo/test_runner.py`

**Interfaces:**
- Consumes: `IndexService.top_hits(...)`（Task 2）、`DemoRunner._finish` 的挂载点、`httpx`（生产依赖已含）
- Produces:
  - `demo.runner.fetch_top_image(rec: PostRecord, timeout: float) -> Image.Image | None`（A-4：base64 解码 / 单次请求、不重试，失败 None）
  - `demo.runner.tier_text(max_sim: float, hard: float, soft: float) -> str`（A-6）
  - `DemoRunner.load_top_posts(outcome: ProcessOutcome, image_bytes: bytes | None, text: str, k: int = 3) -> list[DemoPost]`
  - `_finish` 成功路径填充 `top_posts`；`DemoResult` 追加字段 `top_note: str = ""`（不足 3 条时的提示）

- [ ] **Step 1: Write the failing tests**

在 `tests/demo/test_runner.py` 顶部补充 import：`from unittest.mock import MagicMock`、`from app.domain import PostRecord`、`from demo.runner import fetch_top_image, tier_text`。文件末尾追加：

```python
def test_tier_text_boundaries():
    assert tier_text(0.95, 0.90, 0.75) == "≥0.9：强降权区间（参考）"
    assert tier_text(0.90, 0.90, 0.75) == "≥0.9：强降权区间（参考）"  # 边界归强档
    assert tier_text(0.80, 0.90, 0.75) == "0.75~0.9：软降权区间（参考）"
    assert tier_text(0.50, 0.90, 0.75) == "<0.75：正常"


def test_fetch_top_image_from_base64():
    import base64
    rec = PostRecord(post_id="p", text="t",
                     image_base64=base64.b64encode(_png()).decode())
    img = fetch_top_image(rec, timeout=3.0)
    assert img is not None and img.size == (16, 16)


def test_fetch_top_image_no_image():
    rec = PostRecord(post_id="p", text="t")
    assert fetch_top_image(rec, timeout=3.0) is None


def test_fetch_top_image_bad_base64():
    rec = PostRecord(post_id="p", text="t", image_base64="!!!")
    assert fetch_top_image(rec, timeout=3.0) is None


async def test_fetch_top_image_url_single_attempt(monkeypatch):
    """A-4：URL 渲染走轻量下载——单次请求、异常即 None，不重试。"""
    import demo.runner as mod

    def fake_get(url, timeout, follow_redirects):
        raise RuntimeError("连不上")
    monkeypatch.setattr(mod.httpx, "get", fake_get)
    rec = PostRecord(post_id="p", text="t", image_url="https://example.com/a.jpg")
    assert fetch_top_image(rec, timeout=3.0) is None


async def test_fetch_top_image_url_success(monkeypatch):
    import demo.runner as mod
    resp = MagicMock(status_code=200, content=_png())
    monkeypatch.setattr(mod.httpx, "get", lambda url, timeout, follow_redirects: resp)
    rec = PostRecord(post_id="p", text="t", image_url="https://example.com/a.jpg")
    img = fetch_top_image(rec, timeout=3.0)
    assert img is not None and img.size == (16, 16)


async def test_top3_loaded_with_images(runner):
    r1 = runner.submit(_png((9, 9, 9)), None, "第一帖")
    r2 = runner.submit(_png((9, 9, 9)), None, "换个文案重发")
    assert len(r2.top_posts) == 1
    top = r2.top_posts[0]
    assert top.post_id == r1.post_id
    assert top.text == "第一帖"
    assert top.image is not None  # base64 库存解码成功
    assert top.sim_image > 0.99 and top.sim_text >= 0.0  # 卡片携带双模态分数


async def test_top3_short_note(runner):
    """不足 3 条时 top_note 给出提示。"""
    runner.submit(_png((9, 9, 9)), None, "第一帖")
    r2 = runner.submit(_png((9, 9, 9)), None, "重发")
    assert len(r2.top_posts) < 3
    assert r2.top_note  # 非空提示


async def test_url_submit_download_stage_real(runner):
    """A-5：URL 提交 PostRecord 带 image_url 入库，processor download 阶段走生产 fetch_image。"""
    resp = MagicMock(status_code=200, content=_png())

    class _Client:
        def __init__(self, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, timeout=None): return resp

    import app.pipeline.downloader as dl
    orig_client = dl.httpx.AsyncClient
    dl.httpx.AsyncClient = _Client
    try:
        result = runner.submit(None, "https://example.com/a.jpg", "URL 帖")
    finally:
        dl.httpx.AsyncClient = orig_client
    assert result.status == "indexed"
    assert "download" in result.outcome.timings_ms
    assert result.submitted_image is None  # URL 提交不回显原图（库中不落 base64），UI 展示 URL 文本
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/demo/test_runner.py -v -k "tier or fetch_top or top3 or url_submit"`
Expected: FAIL —— `tier_text` / `fetch_top_image` 不存在，top_posts 恒空。

- [ ] **Step 3: Implement**

`demo/runner.py` 修改点：

① 顶部补充 import：`import httpx`。

② `DemoResult` 追加字段：`top_note: str = ""`。

③ 模块级新增（`DemoRunner` 类之前）：

```python
def fetch_top_image(rec, timeout: float) -> Image.Image | None:
    """Top-3 渲染用图片获取（A-4）：base64 解码；URL 单次请求、不重试。失败 None。"""
    if rec.image_base64:
        try:
            raw = base64.b64decode(rec.image_base64, validate=True)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            return None
    if rec.image_url:
        try:
            resp = httpx.get(rec.image_url, timeout=timeout, follow_redirects=True)
            if resp.status_code != 200:
                return None
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception:
            return None
    return None


def tier_text(max_sim: float, hard: float, soft: float) -> str:
    """阈值标档辅助标注（A-6）：阈值由 cfg 动态传入，判定权在下游。"""
    if max_sim >= hard:
        return f"≥{hard:g}：强降权区间（参考）"
    if max_sim >= soft:
        return f"{soft:g}~{hard:g}：软降权区间（参考）"
    return f"<{soft:g}：正常"
```

④ `DemoRunner` 新增方法（`_finish` 之后）：

```python
    def load_top_posts(self, outcome: ProcessOutcome, image_bytes: bytes | None,
                       text: str, k: int = 3) -> tuple[list[DemoPost], str]:
        """Top-K 检索 + 内容加载。返回 (帖子列表, 不足 k 条时的提示)。"""
        image_vec = (self.embedder.embed_image(image_bytes) if image_bytes
                     else np.zeros(self.embedder.image_dim, dtype=np.float32))
        text_vec = self.embedder.embed_text(text, prefix="query: ")
        cutoff = window_cutoff(utcnow(), self.cfg.window_days)
        hits = self.index.top_hits(image_vec, text_vec,
                                   exclude_id=outcome.result.post_id,
                                   cutoff=cutoff, k=k)

        async def _load():
            out = []
            for h in hits:
                rec = await self.store.get_post(h.post_id)
                if rec is None:
                    out.append(DemoPost(h.post_id, "（帖子记录缺失）", None))
                    continue
                img = await asyncio.to_thread(fetch_top_image, rec, 3.0)
                out.append(DemoPost(h.post_id, rec.text, img,
                                    sim_image=h.sim_image, sim_text=h.sim_text))
            return out

        try:
            posts = asyncio.run(asyncio.wait_for(_load(), timeout=_RUN_TIMEOUT))
        except TimeoutError:
            posts = []
        note = "" if len(posts) >= k else f"仅检索到 {len(posts)} 条相似帖"
        return posts, note
```

⑤ 顶部补充 `import numpy as np`；`_finish` 签名追加参数 `text: str`，`submit()` 内调用改为 `self._finish(post_id, outcome, image_bytes, text)`；`_finish` 成功分支替换为：

```python
        top_posts, top_note = self.load_top_posts(outcome, image_bytes, text=text)
        return DemoResult(post_id, "indexed", None, outcome,
                          top_posts=top_posts, top_note=top_note,
                          submitted_image=submitted)
```

并删除 `_finish` 内 Task 3 的占位注释。

**URL 提交的回显契约（钉死，不再摇摆）**：`PostRecord` 直接以 `image_url` 入库，由 processor 的 `download` 阶段走生产 `fetch_image`（网络下载含重试退避，A-5 计时语义）；库中不落 base64，因此 URL 提交 `submitted_image` 恒为 `None`，UI 以 URL 文本代替回显。不做任何二次下载回显。

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/demo/ -v`
Expected: 全部 PASS。

- [ ] **Step 5: Run full regression**

Run: `poetry run pytest -q`
Expected: 全量 PASS。

- [ ] **Step 6: Commit**

```bash
git add demo/runner.py tests/demo/test_runner.py
git commit -m "feat: demo Top-3 内容加载、轻量图片下载与阈值标档"
```

---

### Task 5: Gradio UI + 依赖安装 + README

**Files:**
- Modify: `pyproject.toml`（dev 组加 gradio）
- Create: `demo/app.py`
- Create: `tests/demo/test_ui.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `DemoRunner`/`DemoResult`/`DemoValidationError`/`tier_text`（Task 3/4）、`load_config()`、`ClipE5Embedder(device=cfg.device)`、`SqliteStore(cfg.db_path)`、`IndexService()`
- Produces: `demo.app.build_ui(runner: DemoRunner) -> gr.Blocks`、`demo.app.stage_rows(timings: dict, url_mode: bool) -> list[list[str]]`、`demo.app._STAGE_ROWS`、可执行入口 `poetry run python demo/app.py`

- [ ] **Step 1: 安装 gradio 到 dev 组**

```bash
poetry add --group dev "gradio>=4.44"
```

Expected: `pyproject.toml` 的 `[dependency-groups] dev` 中出现 gradio，`poetry.lock` 更新。若遇 keyring 挂起（无头环境已知陷阱），先 `export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` 再重试。

- [ ] **Step 2: Write the failing tests**

`tests/demo/test_ui.py`：

```python
import pytest


def test_stage_rows_order_and_dynamic_label():
    """耗时表 7 行按真实执行顺序；download 标签随输入形态切换；缺失阶段显示未执行。"""
    from demo.app import stage_rows
    timings = {"download": 12.3, "image_embed": 456.7, "text_embed": 89.0,
               "persist": 3.2, "search": 1.1, "result_persist": 2.4,
               "faiss_add": 0.9}
    rows = stage_rows(timings, url_mode=False)
    labels = [r[0] for r in rows]
    assert labels == ["图片下载（本地读取+校验）", "图片向量化", "文字向量化",
                      "向量落库", "Faiss 检索", "结果写库", "向量注册"]
    assert rows[0][1] == "12 ms"
    rows_url = stage_rows(timings, url_mode=True)
    assert rows_url[0][0] == "图片下载（网络）"


def test_stage_rows_missing_stage_shows_skipped():
    from demo.app import stage_rows
    rows = stage_rows({"text_embed": 5.0}, url_mode=False)  # 无图帖
    assert rows[1] == ["图片向量化", "未执行"]


def test_build_ui_returns_blocks(tmp_db):
    """build_ui 可构建且绑定全部输出组件。"""
    gr = pytest.importorskip("gradio")
    import asyncio
    from app.config import Config
    from app.embedder.fake import FakeEmbedder
    from app.index.service import IndexService
    from app.store.sqlite_store import SqliteStore
    from demo.app import build_ui
    from demo.runner import DemoRunner

    async def _go():
        cfg = Config(db_path=tmp_db)
        store = SqliteStore(tmp_db)
        runner = DemoRunner(cfg, FakeEmbedder(), store, IndexService())
        ui = build_ui(runner)
        assert isinstance(ui, gr.Blocks)
        await store.close()
        return ui

    ui = asyncio.run(_go())
    # 四个输出组件均已绑定（score/timing/top 文本与画廊）
    assert len(ui.blocks) > 8
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `poetry run pytest tests/demo/test_ui.py -v`
Expected: FAIL —— `demo.app` 不存在。

- [ ] **Step 4: Implement demo/app.py**

```python
"""Gradio demo 入口（spec 2026-08-24 §4/§5）。

启动：poetry run python demo/app.py（127.0.0.1:7860）
约束：不得与 FastAPI 生产服务同时运行（双份内存索引互不可见、SQLite 写锁竞争）。
页面立即开门，模型加载与索引重建后台进行（A-8）；未就绪提交返回"服务准备中"。
"""
import logging
import threading

import gradio as gr

from app.config import load_config
from demo.runner import DemoRunner, DemoValidationError, tier_text

logger = logging.getLogger("demo.ui")

# 耗时表：(timer 阶段键, 上传形态标签, URL 形态标签)，按真实执行顺序排列
_STAGE_ROWS = (
    ("download", "图片下载（本地读取+校验）", "图片下载（网络）"),
    ("image_embed", "图片向量化", "图片向量化"),
    ("text_embed", "文字向量化", "文字向量化"),
    ("persist", "向量落库", "向量落库"),
    ("search", "Faiss 检索", "Faiss 检索"),
    ("result_persist", "结果写库", "结果写库"),
    ("faiss_add", "向量注册", "向量注册"),
)


def stage_rows(timings: dict, url_mode: bool) -> list[list[str]]:
    rows = []
    for key, upload_label, url_label in _STAGE_ROWS:
        v = timings.get(key)
        rows.append([(url_label if url_mode else upload_label),
                     "未执行" if v is None else f"{v:.0f} ms"])
    return rows


def build_ui(runner: DemoRunner) -> gr.Blocks:
    cfg = runner.cfg

    def _render(result, url_mode: bool):
        if result.status == "not_ready":
            return result.error, [["—", "—"]], "", []
        timings = result.outcome.timings_ms if result.outcome else {}
        rows = stage_rows(timings, url_mode)
        if result.outcome and result.outcome.total_ms:
            rows.append(["总耗时", f"{result.outcome.total_ms:.0f} ms"])
        if result.status == "failed":
            return (f"**处理失败**：`{result.error}`\n\n已完成阶段耗时见下表。",
                    rows, "", [])
        res = result.outcome.result
        score_md = (
            "| 指标 | 分数 |\n|---|---|\n"
            f"| 综合 max_sim | **{res.max_sim:.4f}** |\n"
            f"| 图片 sim_image | {res.sim_image:.4f} |\n"
            f"| 文字 sim_text | {res.sim_text:.4f} |\n\n"
            f"**{tier_text(res.max_sim, cfg.hard_downweight_threshold, cfg.soft_downweight_threshold)}**\n\n"
            "> 阈值仅供参考，降权判定由下游推荐系统执行。"
        )
        lines, images = [], []
        for p in result.top_posts:
            lines.append(f"**{p.post_id}** —— 图片 {p.sim_image:.4f} / 文字 {p.sim_text:.4f}\n\n{p.text}")
            if p.image is not None:
                images.append(p.image)
            else:
                lines.append("*图片不可用*")
        if result.top_note:
            lines.append(f"*{result.top_note}*")
        top_md = "\n\n---\n\n".join(lines) if lines else "无相似帖子（索引可能为空）。"
        return score_md, rows, top_md, images

    def on_submit(image_path, url, text):
        try:
            image_bytes = open(image_path, "rb").read() if image_path else None
        except OSError as e:
            return f"**读取上传文件失败**：{e}", [["—", "—"]], "", []
        url = (url or "").strip() or None
        try:
            result = runner.submit(image_bytes, url, text or "")
        except DemoValidationError as e:
            return f"**输入校验失败**：{e}", [["—", "—"]], "", []
        return _render(result, url_mode=url is not None)

    with gr.Blocks(title="帖子相似度 Demo") as ui:
        gr.Markdown(
            "# 帖子相似度检测 Demo\n"
            "上传图片或填写图片 URL（二选一）+ 文字，查看相似度分数、"
            "流水线各阶段耗时与 Top-3 相似帖。\n"
            "**请勿与 FastAPI 生产服务同时运行。**")
        with gr.Row():
            with gr.Column():
                img_in = gr.Image(type="filepath", label="图片上传（与 URL 二选一）")
                url_in = gr.Textbox(label="图片 URL（与上传二选一）",
                                    placeholder="https://...")
                text_in = gr.Textbox(label="文字", lines=4, placeholder="输入帖子正文")
                submit_btn = gr.Button("提交", variant="primary")
            with gr.Column():
                score_md = gr.Markdown("")
                timing_tbl = gr.Dataframe(headers=["阶段", "耗时"],
                                          label="流水线耗时（按真实执行顺序）")
        gr.Markdown("## Top-3 相似帖")
        top_md = gr.Markdown("")
        top_gallery = gr.Gallery(columns=3, label="相似帖图片")
        submit_btn.click(on_submit, inputs=[img_in, url_in, text_in],
                         outputs=[score_md, timing_tbl, top_md, top_gallery])
    return ui


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config()
    logger.warning("请确保 FastAPI 生产服务未同时运行（db=%s）", cfg.db_path)
    from app.embedder.clip_e5 import ClipE5Embedder
    from app.index.service import IndexService
    from app.store.sqlite_store import SqliteStore
    runner = DemoRunner(cfg, ClipE5Embedder(device=cfg.device),
                        SqliteStore(cfg.db_path), IndexService())

    def boot():
        import asyncio
        try:
            asyncio.run(runner.setup())
        except Exception:
            logger.exception("demo 后台装配失败：页面将持续返回'服务准备中'")

    threading.Thread(target=boot, daemon=True).start()  # A-8：立即开门
    ui = build_ui(runner)
    ui.queue().launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `poetry run pytest tests/demo/ -v`
Expected: 全部 PASS。

- [ ] **Step 6: 更新 README**

在 `README.md` 的「启动」章节之后插入新章节：

```markdown
## Gradio Demo（网页试用）

```bash
poetry run python demo/app.py        # 127.0.0.1:7860
```

上传图片或填写图片 URL（二选一）+ 文字提交新帖，页面展示：综合与分模态相似度分数（附阈值标档参考）、流水线各阶段耗时（按真实执行顺序）、Top-3 相似帖内容。设计文档见 [gradio-demo spec](docs/superpowers/specs/2026-08-24-gradio-demo-design.md)。

- **不得与 FastAPI 生产服务同时运行**（两进程内存索引互不可见、SQLite 写锁竞争），运行前先停服务；
- demo 提交的帖子以 `demo-` 前缀 ID 落生产库并注册索引，**不提供清理工具**，30 天窗口自然过期；窗口期内切回生产服务，真实帖可能与 demo 帖碰撞被判重（已接受风险，见 spec 附录 A-7）；
- 页面立即开门，模型加载与索引重建后台进行，就绪前提交返回"服务准备中"。
```

- [ ] **Step 7: Full regression**

Run: `poetry run pytest -q`
Expected: 全量 PASS。

- [ ] **Step 8: 手工冒烟（可选但推荐，需真模型缓存）**

```bash
HF_ENDPOINT=https://hf-mirror.com poetry run python demo/app.py
```

浏览器打开 `http://127.0.0.1:7860`：① 提交一张图+文字，确认分数/耗时表/Top-3 渲染；② 提交前未就绪时确认"服务准备中"；③ 图+URL 同填确认校验提示。若无真模型缓存可跳过（CI 已由 FakeEmbedder 冒烟覆盖）。

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml poetry.lock demo/app.py tests/demo/test_ui.py README.md
git commit -m "feat: Gradio demo UI——分数/耗时/Top-3 展示与后台装配启动"
```

---

## 完成标准

- `poetry run pytest -q` 全量绿（含新增 processor outcome、top_hits、runner、ui 四组测试）；
- 生产 worker 行为零变化（既有 service/api/integration 测试未改动且通过）；
- `poetry run python demo/app.py` 可启动并在浏览器完成一次完整提交演示。

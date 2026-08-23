# 帖子相似度检测系统实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个 FastAPI 服务：接收新帖（图片+文字），异步计算其与最近 30 天所有帖子的最大相似度（max(图,文)），结果落库可查。

**Architecture:** 单进程 FastAPI + asyncio 有界队列 + 单 Worker 串行消费。CLIP ViT-B/32 图片编码（全图 resize 不 center crop）+ multilingual-e5-small 文字编码（passage/query 前缀），双 Faiss IndexFlatIP 索引，SQLite(WAL) 为唯一事实源、索引启动时从库重建。PostStore Protocol 抽象为生产 MongoDB 迁移留边界。

**Tech Stack:** Python 3.11、FastAPI、uvicorn、sqlite3 + asyncio.to_thread、faiss-cpu、open_clip_torch、sentence-transformers、httpx、Pillow、pytest、pytest-asyncio、Faker。

**Spec:** `docs/superpowers/specs/2026-08-23-post-similarity-design.md`（含 grill-me 修订 9 项决议，执行前必读 §3.4⑦⑧、§5.1、§5.2 自适应K、§5.3 两阶段校准、§6.4、§8.1、§9、§11）

## Global Constraints

- Python 3.11+；包管理器优先 uv（`uv venv && uv pip install`），无 uv 时退回 `python -m venv` + `pip`。
- 相似度输出恒在 [0,1]（余弦负值截断为 0）；分数落库保留 4 位小数。
- e5 前缀强制：库存旧帖 `"passage: "`，新帖查询 `"query: "`；漏前缀视为实现缺陷。
- CLIP 预处理：全图 resize 到 224×224（关闭 center crop，允许轻微形变），不得用 open_clip 默认 transform。
- 30 天窗口基准为帖子 `created_at`（UTC），API 未上报时缺省取入库时间；创建时间已超窗的帖子跳过提特征与索引注册，仅登记元数据并记零分结果（note="created_at_outside_window"）。
- SQLite 连接不得跨线程复用（线程局部连接）；所有 DB 调用经 `asyncio.to_thread` 桥接。
- 顺序铁律：先落库向量 → 再写结果(indexed) → 最后 Faiss add；先检索后注册自身，检索结果排除自身 post_id。
- failed 帖不自动重试；兜底扫描只捞 pending（周期 5 分钟、滞留阈值 10 分钟，均可配置）。
- 生产 Worker 单点全量索引（不分片）——实现上 Worker 永远只有一个实例。
- 所有阈值/容量/周期收敛到 `config.py`，支持环境变量覆盖（前缀 `SIM_`）。
- 提交信息格式：`feat:/fix:/test:/docs:/chore: + 中文描述`；每任务一次提交。

## File Structure

```
post_compare/
├── pyproject.toml                    # 依赖与 pytest 配置
├── README.md                         # 运行与联调说明（Task 13）
├── .gitignore
├── app/
│   ├── __init__.py
│   ├── config.py                     # Config dataclass + env 覆盖 + 单例加载
│   ├── domain.py                     # IndexStatus/ErrorCode/PostRecord/SimilarityResult/失败异常层次
│   ├── timer.py                      # StageTimer 分阶段耗时打点（spec §7.4）
│   ├── embedder/
│   │   ├── __init__.py
│   │   ├── base.py                   # Embedder Protocol + normalize 工具
│   │   ├── fake.py                   # FakeEmbedder（确定性伪向量，测试主力）
│   │   └── clip_e5.py                # ClipE5Embedder（真模型，含全图 resize 预处理）
│   ├── index/
│   │   └── service.py                # IndexService：Faiss 双索引/窗口过滤/自排除/自适应K/重建/压缩
│   ├── store/
│   │   ├── __init__.py
│   │   ├── base.py                   # PostStore Protocol
│   │   └── sqlite_store.py           # SqliteStore（WAL + 线程局部连接）
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── downloader.py             # 图片下载（退避重试）+ base64 解码 + 格式校验
│   │   └── processor.py              # 单帖处理编排（下载→编码→落库→检索→注册）
│   ├── api/
│   │   ├── __init__.py
│   │   ├── schemas.py                # Pydantic 请求/响应模型 + 统一错误格式
│   │   └── routes.py                 # 4 个端点
│   └── service.py                    # 应用装配：生命周期/Worker/队列/兜底扫描/回放/优雅停机
├── tests/
│   ├── conftest.py                   # 共享 fixtures（tmp 库/FakeEmbedder/索引）
│   ├── unit/                         # config/domain/store/fake_embedder/timer/index_service/downloader
│   ├── api/                          # 端点层测试（TestClient + FakeEmbedder）
│   ├── integration/                  # 端到端 + 崩溃恢复（子进程强杀）
│   └── slow/                         # 真模型冒烟（@pytest.mark.slow，默认 skip）
└── eval/
    └── synthetic_calibration.py      # 合成校准集生成 + 分数分布统计（Task 13）
```

---

### Task 1: 项目脚手架与配置模块

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `app/__init__.py`, `app/config.py`
- Test: `tests/conftest.py`, `tests/unit/test_config.py`

**Interfaces:**
- Consumes: 无
- Produces: `app.config.load_config() -> Config`；`Config` 字段（后续任务全部依赖其默认值）：`queue_capacity=1000, top_k=16, adaptive_k_steps=(64,256), window_days=30, sweep_interval_seconds=300, pending_stale_seconds=600, slow_task_warn_seconds=5.0, download_retries=3, download_retry_backoff_seconds=(1.0,4.0,16.0), image_max_bytes=20*1024*1024, text_max_chars=5000, download_timeout_seconds=30.0, graceful_shutdown_seconds=30.0, queue_wait_seconds=5.0, db_path, hard_downweight_threshold=0.90, soft_downweight_threshold=0.75, device="cpu"`

- [ ] **Step 1: 初始化环境与依赖**

```bash
cd /home/nelson/Documents/post_compare
uv venv --python 3.11 && source .venv/bin/activate   # 无 uv 则: python3.11 -m venv .venv && source .venv/bin/activate
```

`pyproject.toml`:

```toml
[project]
name = "post-similarity"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115", "uvicorn[standard]>=0.30", "httpx>=0.27",
    "pydantic>=2.8", "pydantic-settings>=2.4", "pillow>=10.4",
    "faiss-cpu>=1.8", "open-clip-torch>=2.26", "sentence-transformers>=3.0",
    "numpy>=1.26,<2",
]

[project.optional-dependencies]
dev = ["pytest>=8.3", "pytest-asyncio>=0.24", "faker>=26"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
markers = ["slow: 真模型冒烟测试，默认 skip（-m slow 显式运行）"]
addopts = "-m 'not slow'"
testpaths = ["tests"]
```

`.gitignore`:

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
data/
*.db
*.db-wal
*.db-shm
```

安装：`uv pip install -e ".[dev]"`（或 `pip install -e ".[dev]"`）。
创建空包：`app/__init__.py`（内容 `"""帖子相似度检测服务。"""`），并创建空的 `app/embedder/__init__.py`、`app/index/__init__.py`、`app/store/__init__.py`、`app/pipeline/__init__.py`、`app/api/__init__.py`、`tests/__init__.py`、`tests/unit/__init__.py`、`tests/api/__init__.py`、`tests/integration/__init__.py`、`tests/slow/__init__.py`、`eval/.gitkeep`。

- [ ] **Step 2: 写失败测试**

`tests/unit/test_config.py`:

```python
import pytest
from app.config import Config, load_config


def test_defaults_match_spec():
    cfg = Config(db_path="/tmp/x.db")
    assert cfg.queue_capacity == 1000
    assert cfg.top_k == 16
    assert cfg.adaptive_k_steps == (64, 256)
    assert cfg.window_days == 30
    assert cfg.sweep_interval_seconds == 300
    assert cfg.pending_stale_seconds == 600
    assert cfg.slow_task_warn_seconds == 5.0
    assert cfg.download_retries == 3
    assert cfg.download_retry_backoff_seconds == (1.0, 4.0, 16.0)
    assert cfg.image_max_bytes == 20 * 1024 * 1024
    assert cfg.text_max_chars == 5000
    assert cfg.hard_downweight_threshold == 0.90
    assert cfg.soft_downweight_threshold == 0.75
    assert cfg.device == "cpu"


def test_env_override(monkeypatch):
    monkeypatch.setenv("SIM_TOP_K", "32")
    monkeypatch.setenv("SIM_DB_PATH", "/tmp/other.db")
    cfg = load_config()
    assert cfg.top_k == 32
    assert cfg.db_path == "/tmp/other.db"


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        Config(db_path="/tmp/x.db", hard_downweight_threshold=0.5,
               soft_downweight_threshold=0.9)  # hard 必须 >= soft
```

- [ ] **Step 3: 运行确认失败**

Run: `pytest tests/unit/test_config.py -v`
Expected: FAIL（`ModuleNotFoundError: app.config`）

- [ ] **Step 4: 最小实现**

`app/config.py`（pydantic-settings，前缀 `SIM_`，元组字段用 JSON 解析）：

```python
"""全局配置：spec §5.4/§10，所有阈值容量周期收敛于此，支持 SIM_ 前缀环境变量覆盖。"""
from functools import lru_cache
from typing import Tuple

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SIM_")

    db_path: str = "data/similarity.db"
    device: str = "cpu"

    queue_capacity: int = 1000
    queue_wait_seconds: float = 5.0
    top_k: int = 16
    adaptive_k_steps: Tuple[int, ...] = (64, 256)
    window_days: int = 30

    sweep_interval_seconds: int = 300
    pending_stale_seconds: int = 600
    slow_task_warn_seconds: float = 5.0

    download_retries: int = 3
    download_retry_backoff_seconds: Tuple[float, ...] = (1.0, 4.0, 16.0)
    download_timeout_seconds: float = 30.0
    image_max_bytes: int = 20 * 1024 * 1024
    text_max_chars: int = 5000
    graceful_shutdown_seconds: float = 30.0

    hard_downweight_threshold: float = 0.90
    soft_downweight_threshold: float = 0.75

    @field_validator("adaptive_k_steps", "download_retry_backoff_seconds", mode="before")
    @classmethod
    def _parse_tuple(cls, v):
        if isinstance(v, str):
            import json
            return tuple(json.loads(v))
        return tuple(v)

    @field_validator("soft_downweight_threshold")
    @classmethod
    def _check_thresholds(cls, v, info):
        hard = info.data.get("hard_downweight_threshold", 0.90)
        if v > hard:
            raise ValueError("soft_downweight_threshold 不得大于 hard_downweight_threshold")
        return v


@lru_cache
def load_config() -> Config:
    return Config()
```

`tests/conftest.py`:

```python
import pytest


@pytest.fixture
def tmp_db(tmp_path):
    """每个测试独立的 SQLite 路径。"""
    return str(tmp_path / "test.db")
```

- [ ] **Step 5: 运行确认通过并提交**

Run: `pytest tests/unit/test_config.py -v` → Expected: 3 passed

```bash
git add -A && git commit -m "feat: 项目脚手架与配置模块（SIM_ 环境变量覆盖）"
```

---

### Task 2: 领域模型与错误分类

**Files:**
- Create: `app/domain.py`
- Test: `tests/unit/test_domain.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `IndexStatus`（str Enum）: `PENDING="pending"`, `INDEXED="indexed"`, `FAILED="failed"`
  - `ErrorCode`（str Enum）: `IMAGE_DOWNLOAD_FAILED`, `IMAGE_DECODE_FAILED`, `ENCODE_FAILED`, `INTERNAL_ERROR`
  - `ProcessingError(Exception)` 带 `code: ErrorCode`；子类 `ImageDownloadError`/`ImageDecodeError`/`EncodeError`
  - `PostRecord` dataclass: `post_id, image_url, image_base64, text, created_at(datetime,UTC), enqueued_at(datetime,UTC), status, note`
  - `SimilarityResult` dataclass: `post_id, max_sim, sim_image, sim_text, matched_post_id(str|None), computed_at, note(str|None)`
  - `window_cutoff(now, window_days) -> datetime`

- [ ] **Step 1: 写失败测试**

`tests/unit/test_domain.py`:

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/unit/test_domain.py -v`
Expected: FAIL（`ModuleNotFoundError: app.domain`）

- [ ] **Step 3: 最小实现**

`app/domain.py`:

```python
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
    ENCODE_FAILED = "encode_failed"
    INTERNAL_ERROR = "internal_error"


class ProcessingError(Exception):
    code: ErrorCode = ErrorCode.INTERNAL_ERROR


class ImageDownloadError(ProcessingError):
    code = ErrorCode.IMAGE_DOWNLOAD_FAILED


class ImageDecodeError(ProcessingError):
    code = ErrorCode.IMAGE_DECODE_FAILED


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
```

- [ ] **Step 4: 运行确认通过并提交**

Run: `pytest tests/unit/test_domain.py -v` → Expected: 5 passed

```bash
git add app/domain.py tests/unit/test_domain.py && git commit -m "feat: 领域模型与错误分类"
```

---

### Task 3: SqliteStore（WAL + 线程局部连接）

**Files:**
- Create: `app/store/base.py`, `app/store/sqlite_store.py`
- Test: `tests/unit/test_sqlite_store.py`

**Interfaces:**
- Consumes: Task 2 `PostRecord/SimilarityResult/IndexStatus/ErrorCode/utcnow`
- Produces: `PostStore` Protocol 与 `SqliteStore` 实现，方法全部 async：
  - `upsert_post(rec: PostRecord) -> bool`（True=新帖已写入 pending；False=post_id 已存在，幂等不覆盖）
  - `get_post(post_id) -> PostRecord | None`
  - `persist_vectors(post_id, image_vec: bytes, text_vec: bytes)`
  - `mark_indexed(post_id, result: SimilarityResult)`（与结果同事务）
  - `mark_failed(post_id, code: ErrorCode)`
  - `get_result(post_id) -> SimilarityResult | None`
  - `list_stale_pending(older_than: datetime) -> list[str]`
  - `list_indexed_within(cutoff: datetime) -> list[tuple[str, bytes, bytes, datetime]]`（启动重建用：post_id/图向量/文向量/created_at）
  - `reset_failed_to_pending(post_id) -> bool`（/admin/replay 用）
  - `counts() -> dict`（index_count/failed_count，/health 用）
  - 向量以 float32 bytes 存取：`encode_vector(np.ndarray) -> bytes` / `decode_vector(bytes) -> np.ndarray` 模块函数

- [ ] **Step 1: 写失败测试**

`tests/unit/test_sqlite_store.py`:

```python
import numpy as np
import pytest
from datetime import datetime, timedelta, timezone

from app.domain import (ErrorCode, IndexStatus, PostRecord, SimilarityResult)
from app.store.sqlite_store import SqliteStore, decode_vector, encode_vector


@pytest.fixture
async def store(tmp_db):
    s = SqliteStore(tmp_db)
    await s.init()
    yield s
    await s.close()


def _rec(pid="p1", **kw):
    return PostRecord(post_id=pid, text="你好世界", **kw)


async def test_upsert_is_idempotent(store):
    assert await store.upsert_post(_rec()) is True
    assert await store.upsert_post(_rec(text="被覆盖?")) is False
    rec = await store.get_post("p1")
    assert rec.text == "你好世界"  # 重复提交不覆盖


async def test_vector_roundtrip_lossless(store):
    vec = np.random.rand(512).astype(np.float32)
    await store.upsert_post(_rec())
    await store.persist_vectors("p1", encode_vector(vec), encode_vector(vec[:384]))
    rows = await store.list_indexed_within(datetime(2000, 1, 1, tzinfo=timezone.utc))
    assert rows == []  # 尚未 indexed，重建不含它


async def test_mark_indexed_with_result_in_one_transaction(store):
    await store.upsert_post(_rec())
    vec = np.ones(512, dtype=np.float32)
    await store.persist_vectors("p1", encode_vector(vec), encode_vector(vec))
    res = SimilarityResult(post_id="p1", max_sim=0.0, sim_image=0.0, sim_text=0.0,
                           matched_post_id=None,
                           computed_at=datetime.now(timezone.utc), note=None)
    await store.mark_indexed("p1", res)
    rec = await store.get_post("p1")
    assert rec.status == IndexStatus.INDEXED
    got = await store.get_result("p1")
    assert got.max_sim == 0.0 and got.matched_post_id is None
    rows = await store.list_indexed_within(datetime(2000, 1, 1, tzinfo=timezone.utc))
    assert len(rows) == 1 and rows[0][0] == "p1"
    assert np.allclose(decode_vector(rows[0][1]), vec)


async def test_mark_failed_and_reset(store):
    await store.upsert_post(_rec())
    await store.mark_failed("p1", ErrorCode.IMAGE_DOWNLOAD_FAILED)
    rec = await store.get_post("p1")
    assert rec.status == IndexStatus.FAILED
    assert rec.note == "image_download_failed"
    assert await store.reset_failed_to_pending("p1") is True
    assert (await store.get_post("p1")).status == IndexStatus.PENDING


async def test_list_stale_pending_filters_by_time(store):
    await store.upsert_post(_rec("fresh"))
    await store.upsert_post(_rec("stale"))
    # 手工把 stale帖的 enqueued_at 拨早
    await store._execute(
        "UPDATE posts SET enqueued_at=? WHERE post_id=?",
        ("2026-01-01T00:00:00+00:00", "stale"))
    stale = await store.list_stale_pending(datetime(2026, 8, 1, tzinfo=timezone.utc))
    assert stale == ["stale"]


async def test_counts(store):
    await store.upsert_post(_rec("a"))
    await store.upsert_post(_rec("b"))
    await store.mark_failed("b", ErrorCode.IMAGE_DECODE_FAILED)
    c = await store.counts()
    assert c["failed_count"] == 1
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/unit/test_sqlite_store.py -v`
Expected: FAIL（`ModuleNotFoundError: app.store.sqlite_store`）

- [ ] **Step 3: 最小实现**

`app/store/base.py`:

```python
"""PostStore Protocol（spec §3.3）：Demo=SqliteStore，生产=MongoStore（motor），签名对齐。"""
from datetime import datetime
from typing import Protocol

import numpy as np

from app.domain import ErrorCode, PostRecord, SimilarityResult


class PostStore(Protocol):
    async def init(self) -> None: ...
    async def close(self) -> None: ...
    async def upsert_post(self, rec: PostRecord) -> bool: ...
    async def get_post(self, post_id: str) -> PostRecord | None: ...
    async def persist_vectors(self, post_id: str, image_vec: bytes, text_vec: bytes) -> None: ...
    async def mark_indexed(self, post_id: str, result: SimilarityResult) -> None: ...
    async def mark_failed(self, post_id: str, code: ErrorCode) -> None: ...
    async def get_result(self, post_id: str) -> SimilarityResult | None: ...
    async def list_stale_pending(self, older_than: datetime) -> list[str]: ...
    async def list_indexed_within(self, cutoff: datetime) -> list[tuple[str, bytes, bytes, datetime]]: ...
    async def reset_failed_to_pending(self, post_id: str) -> bool: ...
    async def counts(self) -> dict: ...
```

`app/store/sqlite_store.py`:

```python
"""SqliteStore：WAL + 线程局部连接（spec §8.1 grill-me 决议 8）。

sqlite3 为同步阻塞 API：所有方法经 asyncio.to_thread 桥接；
连接存于 threading.local，绝不跨线程复用。
"""
import asyncio
import sqlite3
import threading
from datetime import datetime, timezone

import numpy as np

from app.domain import ErrorCode, IndexStatus, PostRecord, SimilarityResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
    post_id     TEXT PRIMARY KEY,
    image_url   TEXT,
    image_base64 TEXT,
    text        TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    enqueued_at TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    note        TEXT,
    image_vec   BLOB,
    text_vec    BLOB
);
CREATE TABLE IF NOT EXISTS results (
    post_id         TEXT PRIMARY KEY REFERENCES posts(post_id),
    max_sim         REAL NOT NULL,
    sim_image       REAL NOT NULL,
    sim_text        REAL NOT NULL,
    matched_post_id TEXT,
    computed_at     TEXT NOT NULL,
    note            TEXT
);
CREATE INDEX IF NOT EXISTS idx_posts_status ON posts(status, enqueued_at);
"""


def encode_vector(v: np.ndarray) -> bytes:
    return np.ascontiguousarray(v, dtype=np.float32).tobytes()


def decode_vector(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32).copy()


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse(s: str) -> datetime:
    return datetime.fromisoformat(s)


class SqliteStore:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._local = threading.local()

    # ---- 线程局部连接 ----
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def _execute(self, sql: str, params=()):
        conn = self._conn()
        with conn:  # 事务边界
            cur = conn.execute(sql, params)
        return cur

    def _query(self, sql: str, params=()):
        return self._conn().execute(sql, params).fetchall()

    # ---- 公开 async API（全部 to_thread 桥接）----
    async def init(self):
        await asyncio.to_thread(self._conn().executescript, _SCHEMA)

    async def close(self):
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    async def upsert_post(self, rec: PostRecord) -> bool:
        def _do():
            cur = self._execute(
                """INSERT INTO posts(post_id,image_url,image_base64,text,created_at,
                                     enqueued_at,status,note)
                   VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(post_id) DO NOTHING""",
                (rec.post_id, rec.image_url, rec.image_base64, rec.text,
                 _iso(rec.created_at), _iso(rec.enqueued_at),
                 rec.status.value, rec.note))
            return cur.rowcount == 1
        return await asyncio.to_thread(_do)

    async def get_post(self, post_id: str) -> PostRecord | None:
        rows = await asyncio.to_thread(
            self._query,
            "SELECT post_id,image_url,image_base64,text,created_at,enqueued_at,status,note "
            "FROM posts WHERE post_id=?", (post_id,))
        if not rows:
            return None
        r = rows[0]
        return PostRecord(post_id=r[0], image_url=r[1], image_base64=r[2], text=r[3],
                          created_at=_parse(r[4]), enqueued_at=_parse(r[5]),
                          status=IndexStatus(r[6]), note=r[7])

    async def persist_vectors(self, post_id, image_vec: bytes, text_vec: bytes):
        await asyncio.to_thread(
            self._execute,
            "UPDATE posts SET image_vec=?, text_vec=? WHERE post_id=?",
            (image_vec, text_vec, post_id))

    async def mark_indexed(self, post_id, result: SimilarityResult):
        def _do():
            conn = self._conn()
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO results VALUES(?,?,?,?,?,?,?)",
                    (post_id, result.max_sim, result.sim_image, result.sim_text,
                     result.matched_post_id, _iso(result.computed_at), result.note))
                conn.execute("UPDATE posts SET status='indexed' WHERE post_id=?", (post_id,))
        await asyncio.to_thread(_do)

    async def mark_failed(self, post_id, code: ErrorCode):
        await asyncio.to_thread(
            self._execute,
            "UPDATE posts SET status='failed', note=? WHERE post_id=?",
            (code.value, post_id))

    async def get_result(self, post_id) -> SimilarityResult | None:
        rows = await asyncio.to_thread(
            self._query, "SELECT * FROM results WHERE post_id=?", (post_id,))
        if not rows:
            return None
        r = rows[0]
        return SimilarityResult(post_id=r[0], max_sim=r[1], sim_image=r[2], sim_text=r[3],
                                matched_post_id=r[4], computed_at=_parse(r[5]), note=r[6])

    async def list_stale_pending(self, older_than: datetime) -> list[str]:
        rows = await asyncio.to_thread(
            self._query,
            "SELECT post_id FROM posts WHERE status='pending' AND enqueued_at<?",
            (_iso(older_than),))
        return [r[0] for r in rows]

    async def list_indexed_within(self, cutoff: datetime):
        rows = await asyncio.to_thread(
            self._query,
            "SELECT post_id,image_vec,text_vec,created_at FROM posts "
            "WHERE status='indexed' AND image_vec IS NOT NULL AND created_at>=?",
            (_iso(cutoff),))
        return [(r[0], r[1], r[2], _parse(r[3])) for r in rows]

    async def reset_failed_to_pending(self, post_id) -> bool:
        cur = await asyncio.to_thread(
            self._execute,
            "UPDATE posts SET status='pending', note=NULL WHERE post_id=? AND status='failed'",
            (post_id,))
        return cur.rowcount == 1

    async def counts(self) -> dict:
        rows = await asyncio.to_thread(
            self._query, "SELECT status, COUNT(*) FROM posts GROUP BY status")
        d = {s: n for s, n in rows}
        return {"index_count": d.get("indexed", 0), "failed_count": d.get("failed", 0)}
```

- [ ] **Step 4: 运行确认通过并提交**

Run: `pytest tests/unit/test_sqlite_store.py -v` → Expected: 6 passed

```bash
git add app/store/ tests/unit/test_sqlite_store.py && git commit -m "feat: SqliteStore（WAL+线程局部连接+asyncio.to_thread 桥接）"
```


---

### Task 4: Embedder 协议与 FakeEmbedder

**Files:**
- Create: `app/embedder/base.py`, `app/embedder/fake.py`
- Test: `tests/unit/test_fake_embedder.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `Embedder` Protocol：`embed_image(image_bytes: bytes) -> np.ndarray`（shape (512,) float32，L2 范数=1）、`embed_text(text: str, *, prefix: str) -> np.ndarray`（shape (384,) float32，L2 范数=1）、`load() -> None`、`image_dim -> int`、`text_dim -> int`
  - `normalize(v: np.ndarray) -> np.ndarray` 工具函数
  - `FakeEmbedder`：确定性伪向量——图片按字节 sha256 播种生成（相同字节→相同向量；字节差异越大向量越不相似：对首 1KB 做字节直方图特征 + 固定 basis 混合），文字按字符 n-gram 哈希累加生成（语义近似的长文本向量接近，用于集成测试断言相对大小关系）；prefix 参数会被并入哈希（测试前缀传导）

- [ ] **Step 1: 写失败测试**

`tests/unit/test_fake_embedder.py`:

```python
import numpy as np
import pytest

from app.embedder.base import normalize
from app.embedder.fake import FakeEmbedder


@pytest.fixture
def emb():
    e = FakeEmbedder()
    e.load()
    return e


def test_normalize_unit_norm():
    v = normalize(np.array([3.0, 4.0]))
    assert abs(np.linalg.norm(v) - 1.0) < 1e-6


def test_image_embedding_shape_norm_determinism(emb):
    b = b"\x01" * 100
    v1, v2 = emb.embed_image(b), emb.embed_image(b)
    assert v1.shape == (512,) and v1.dtype == np.float32
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-6
    assert np.allclose(v1, v2)


def test_different_images_low_similarity(emb):
    a = emb.embed_image(b"AAAA" * 300)
    b = emb.embed_image(b"BBBB" * 300)
    assert float(a @ b) < 0.3


def test_similar_bytes_high_similarity(emb):
    """仅改 1 字节，相似度应显著高于完全不同的内容。"""
    base = bytearray(b"X" * 2000)
    near = bytes(base[:-1]) + b"Y"
    far = b"Z" * 2000
    v0, v1, v2 = emb.embed_image(bytes(base)), emb.embed_image(near), emb.embed_image(far)
    assert float(v0 @ v1) > float(v0 @ v2)


def test_text_embedding_shape_norm(emb):
    v = emb.embed_text("今天天气不错", prefix="passage: ")
    assert v.shape == (384,) and abs(np.linalg.norm(v) - 1.0) < 1e-6


def test_prefix_changes_vector(emb):
    """e5 前缀必须实际影响向量（防静默漏前缀）。"""
    p = emb.embed_text("同一段文字", prefix="passage: ")
    q = emb.embed_text("同一段文字", prefix="query: ")
    assert not np.allclose(p, q)


def test_similar_texts_more_similar(emb):
    a = emb.embed_text("今天天气很好我们去公园散步", prefix="passage: ")
    b = emb.embed_text("今天天气很好我们去公园散步呀", prefix="query: ")
    c = emb.embed_text("量子色动力学中的渐近自由现象", prefix="query: ")
    assert float(a @ b) > float(a @ c)
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/unit/test_fake_embedder.py -v`
Expected: FAIL（`ModuleNotFoundError: app.embedder`）

- [ ] **Step 3: 最小实现**

`app/embedder/base.py`:

```python
"""Embedder 协议（spec §8.1）：FakeEmbedder 进 CI，ClipE5Embedder 本地冒烟。"""
from typing import Protocol

import numpy as np

IMAGE_DIM = 512   # CLIP ViT-B/32
TEXT_DIM = 384    # multilingual-e5-small


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("零向量无法归一化")
    return v / n


class Embedder(Protocol):
    image_dim: int
    text_dim: int

    def load(self) -> None: ...
    def embed_image(self, image_bytes: bytes) -> np.ndarray: ...
    def embed_text(self, text: str, *, prefix: str) -> np.ndarray: ...
```

`app/embedder/fake.py`:

```python
"""FakeEmbedder：确定性伪向量，测试主力（spec §8.1）。

图片：sha256 播种主方向 + 首 1KB 字节直方图特征（固定 basis），
字节微小差异→向量接近，跨输入一致。
文字：字符 2-gram 哈希累加，共享 n-gram 越多越接近。
prefix 独立哈希为固定小扰动方向：同内容跨前缀仍高相似（仿 e5 语义契约），
且不同前缀向量确实不同（防静默漏前缀）。
"""
import hashlib

import numpy as np

from app.embedder.base import IMAGE_DIM, TEXT_DIM, normalize

# 固定 basis：使直方图特征→向量的映射跨输入一致（不能用内容播种，否则 1 字节变化即失效）
_BASIS = np.random.default_rng(0x5EED).standard_normal((256, IMAGE_DIM)).astype(np.float32)


class FakeEmbedder:
    image_dim = IMAGE_DIM
    text_dim = TEXT_DIM

    def load(self) -> None:
        pass

    def embed_image(self, image_bytes: bytes) -> np.ndarray:
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(image_bytes).digest()[:8], "big"))
        v = rng.standard_normal(IMAGE_DIM).astype(np.float32)
        head = image_bytes[:1024]
        if head:
            # 字节直方图特征注入低频分量：相近内容获得相近偏移
            feats = np.zeros(256, dtype=np.float32)
            for byte in head:
                feats[byte] += 1.0
            feats /= 1024.0
            v = v * 0.1 + feats @ _BASIS
        return normalize(v)

    def embed_text(self, text: str, *, prefix: str) -> np.ndarray:
        v = np.zeros(TEXT_DIM, dtype=np.float32)
        for i in range(len(text) - 1):
            gram = text[i:i + 2]
            h = int.from_bytes(hashlib.sha256(gram.encode()).digest()[:4], "big")
            v[h % TEXT_DIM] += 1.0
        if v.sum() == 0:  # 空文本兜底：固定方向
            v[0] = 1.0
        v = normalize(v)
        # prefix 独立哈希方向的小扰动：向量随前缀变化，但内容信号主导（同文跨前缀 >0.9）
        ph = int.from_bytes(hashlib.sha256(prefix.encode()).digest()[:4], "big")
        rng = np.random.default_rng(ph)
        pv = rng.standard_normal(TEXT_DIM).astype(np.float32)
        return normalize(v + 0.15 * normalize(pv))
```

- [ ] **Step 4: 运行确认通过并提交**

Run: `pytest tests/unit/test_fake_embedder.py -v` → Expected: 7 passed

```bash
git add app/embedder/ tests/unit/test_fake_embedder.py && git commit -m "feat: Embedder 协议与 FakeEmbedder（确定性伪向量）"
```

---

### Task 5: ClipE5Embedder（真模型）

**Files:**
- Create: `app/embedder/clip_e5.py`
- Test: `tests/slow/test_real_embedder.py`

**Interfaces:**
- Consumes: Task 4 `base.py`（IMAGE_DIM/TEXT_DIM/normalize）
- Produces: `ClipE5Embedder(device: str = "cpu")` 实现 `Embedder` 协议；`load()` 拉取 `ViT-B-32`（open_clip pretrained=laion2b_s34b_b79k）与 `intfloat/multilingual-e5-small`；首次运行自动下载模型到本地缓存（~600MB，联网）

- [ ] **Step 1: 写失败测试（slow 标记，默认 skip）**

`tests/slow/test_real_embedder.py`:

```python
import io

import numpy as np
import pytest
from PIL import Image

from app.embedder.clip_e5 import ClipE5Embedder

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def emb():
    e = ClipE5Embedder(device="cpu")
    e.load()
    return e


def _patterned_png(seed=0, size=(640, 480)) -> bytes:
    """带随机矩形的图案图（纯色图裁剪后无信息量，测不出裁剪鲁棒性）。"""
    import random
    from PIL import ImageDraw
    rng = random.Random(seed)
    img = Image.new("RGB", size, (200, 200, 200))
    d = ImageDraw.Draw(img)
    for _ in range(10):
        x, y = rng.randrange(size[0] - 100), rng.randrange(size[1] - 100)
        d.rectangle([x, y, x + 80, y + 60],
                    fill=(rng.randrange(256), rng.randrange(256), rng.randrange(256)))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_image_shape_and_norm(emb):
    v = emb.embed_image(_patterned_png(seed=1))
    assert v.shape == (512,) and abs(np.linalg.norm(v) - 1.0) < 1e-5


def test_crop_keeps_high_similarity(emb):
    """裁剪鲁棒性基线（spec §11 决议1）：中心裁剪 50% 后相似度应 >0.8。"""
    full = _patterned_png(seed=0)
    img = Image.open(io.BytesIO(full))
    w, h = img.size
    buf = io.BytesIO()
    img.crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4)).save(buf, format="PNG")
    sim = float(emb.embed_image(full) @ emb.embed_image(buf.getvalue()))
    assert sim > 0.8, f"裁剪相似度过低: {sim}"


def test_text_prefix_contract(emb):
    p = emb.embed_text("今天天气很好", prefix="passage: ")
    q = emb.embed_text("今天天气很好", prefix="query: ")
    assert p.shape == (384,) and abs(np.linalg.norm(q) - 1.0) < 1e-5
    assert not np.allclose(p, q)  # 前缀必须生效


def test_semantic_similarity_higher_than_random(emb):
    a = emb.embed_text("小猫在沙发上睡觉", prefix="passage: ")
    b = emb.embed_text("一只猫咪正在沙发上休息", prefix="query: ")
    c = emb.embed_text("央行宣布下调存款准备金率", prefix="query: ")
    assert float(a @ b) > float(a @ c)
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/slow/test_real_embedder.py -v -m slow`
Expected: FAIL（`ModuleNotFoundError: app.embedder.clip_e5`）

- [ ] **Step 3: 最小实现**

`app/embedder/clip_e5.py`:

```python
"""ClipE5Embedder：真模型实现（spec §5.1）。

关键点：
1. CLIP 预处理关闭 center crop——全图 resize 到 224×224（允许轻微形变），
   对近重复匹配有实打实提升（spec §11 决议1）。禁用 open_clip 默认 transform。
2. e5 必须带前缀：调用方传 "passage: " / "query: "，本类不做兜底拼接，
   前缀缺失属于调用方缺陷（spec §5.1）。
3. encode 为同步 CPU 密集操作，由调用方放入线程池执行。
"""
import io

import numpy as np
import torch
from PIL import Image

from app.embedder.base import IMAGE_DIM, TEXT_DIM, normalize


class ClipE5Embedder:
    image_dim = IMAGE_DIM
    text_dim = TEXT_DIM

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._clip = None
        self._clip_preprocess = None
        self._e5 = None

    def load(self) -> None:
        import open_clip
        from sentence_transformers import SentenceTransformer

        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k")
        self._clip = model.eval().to(self.device)

        from torchvision import transforms
        # 全图 resize，不 center crop（spec §11 决议1）
        self._clip_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711)),
        ])

        self._e5 = SentenceTransformer("intfloat/multilingual-e5-small",
                                       device=self.device)

    def embed_image(self, image_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self._clip_preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self._clip.encode_image(tensor)
        return normalize(feat[0].float().cpu().numpy())

    def embed_text(self, text: str, *, prefix: str) -> np.ndarray:
        vec = self._e5.encode([prefix + text], normalize_embeddings=False)[0]
        return normalize(np.asarray(vec, dtype=np.float32))
```

注意：`torchvision` 随 `open-clip-torch` 依赖自动安装；若环境报缺，执行 `uv pip install torchvision`。

- [ ] **Step 4: 运行确认通过并提交**

Run: `pytest tests/slow/test_real_embedder.py -v -m slow` → Expected: 4 passed（首次运行需联网下载模型，约 5-10 分钟）

```bash
git add app/embedder/clip_e5.py tests/slow/ && git commit -m "feat: ClipE5Embedder（全图 resize 预处理 + e5 前缀契约）"
```

---

### Task 6: StageTimer 分阶段耗时打点

**Files:**
- Create: `app/timer.py`
- Test: `tests/unit/test_timer.py`

**Interfaces:**
- Consumes: 无
- Produces: `StageTimer` 类——`with timer.stage("download"):` 上下文记录各阶段耗时；`timer.total_ms() -> float`；`timer.summary() -> str`（输出 `breakdown: download=812ms image_embed=385ms ...` 格式）；`timer.stages -> dict[str, float]`（毫秒）

- [ ] **Step 1: 写失败测试**

`tests/unit/test_timer.py`:

```python
import time

from app.timer import StageTimer


def test_stage_records_elapsed_ms():
    t = StageTimer()
    with t.stage("download"):
        time.sleep(0.02)
    assert t.stages["download"] >= 20.0
    assert t.total_ms() >= 20.0


def test_summary_format_contains_all_stages():
    t = StageTimer()
    with t.stage("download"):
        pass
    with t.stage("image_embed"):
        pass
    s = t.summary()
    assert "download=" in s and "image_embed=" in s and s.endswith("ms")


def test_failed_stage_still_recorded():
    """任一步骤失败时已完成阶段耗时仍可用（spec §7.4）。"""
    t = StageTimer()
    with t.stage("download"):
        time.sleep(0.01)
    try:
        with t.stage("image_embed"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert "download" in t.stages and "image_embed" in t.stages
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/unit/test_timer.py -v`
Expected: FAIL（`ModuleNotFoundError: app.timer`）

- [ ] **Step 3: 最小实现**

`app/timer.py`:

```python
"""StageTimer：单帖生命周期分阶段耗时打点（spec §7.4）。"""
import time
from contextlib import contextmanager


class StageTimer:
    def __init__(self):
        self.stages: dict[str, float] = {}
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        s = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = (time.perf_counter() - s) * 1000.0

    def total_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def summary(self) -> str:
        parts = " ".join(f"{k}={v:.0f}ms" for k, v in self.stages.items())
        return f"breakdown: {parts}"
```

- [ ] **Step 4: 运行确认通过并提交**

Run: `pytest tests/unit/test_timer.py -v` → Expected: 3 passed

```bash
git add app/timer.py tests/unit/test_timer.py && git commit -m "feat: StageTimer 分阶段耗时打点"
```

---

### Task 7: IndexService（Faiss 双索引 + 窗口过滤 + 自排除 + 自适应 K）

**Files:**
- Create: `app/index/service.py`
- Test: `tests/unit/test_index_service.py`

**Interfaces:**
- Consumes: Task 4 `IMAGE_DIM/TEXT_DIM`
- Produces: `IndexService(image_dim=512, text_dim=384)`：
  - `add(post_id: str, created_at: datetime, image_vec: np.ndarray, text_vec: np.ndarray)`
  - `search(image_vec, text_vec, *, exclude_id: str, cutoff: datetime, top_k: int, adaptive_k_steps: tuple[int,...]) -> SearchHit`，其中 `SearchHit` dataclass 含 `sim_image, sim_text, matched_post_id(str|None), adaptive_expanded: bool`；`sim = max(sim_image, sim_text)` 由调用方计算
  - `rebuild(rows: list[tuple[str, datetime, np.ndarray, np.ndarray]])`（启动重建/压缩）
  - `size -> int`、`compact(now, window_days) -> int`（剔除窗口外向量，返回剔除数）
  - 内部：Faiss `IndexIDMap2(IndexFlatIP)` 以自增 int64 id 映射，另维护 `dict[int, (post_id, created_at)]` 元数据表

- [ ] **Step 1: 写失败测试**

`tests/unit/test_index_service.py`:

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/unit/test_index_service.py -v`
Expected: FAIL（`ModuleNotFoundError: app.index.service`）

- [ ] **Step 3: 最小实现**

`app/index/service.py`:

```python
"""IndexService：Faiss 双索引检索（spec §3.4/§5.2）。

- IndexFlatIP + IndexIDMap2：归一化向量内积 = 余弦相似度
- 30 天窗口与自排除在检索后按元数据过滤（spec §3.4①）
- 自适应 K：Top-K 过滤后全空且确有命中时逐级扩 K 重查（spec §5.2 grill-me 决议3）
- 单线程使用（Worker 串行追加 + 检索在线程池），不加锁
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
        # matched_post_id 取 max 信号方；并列时优先图片信号
        if sim_image >= sim_text:
            matched = img_hit[1] if img_hit and sim_image > 0 else None
        else:
            matched = txt_hit[1] if txt_hit else None
        return SearchHit(sim_image=max(sim_image, 0.0), sim_text=max(sim_text, 0.0),
                         matched_post_id=matched, adaptive_expanded=expanded)
```

- [ ] **Step 4: 运行确认通过并提交**

Run: `pytest tests/unit/test_index_service.py -v` → Expected: 8 passed

```bash
git add app/index/ tests/unit/test_index_service.py && git commit -m "feat: IndexService（Faiss 双索引+窗口过滤+自排除+自适应K）"
```


---

### Task 8: 图片下载与解码（退避重试）

**Files:**
- Create: `app/pipeline/downloader.py`
- Test: `tests/unit/test_downloader.py`

**Interfaces:**
- Consumes: Task 1 `Config`、Task 2 `ImageDownloadError/ImageDecodeError`
- Produces:
  - `async fetch_image(rec: PostRecord, cfg: Config) -> bytes`：优先 `image_base64`（base64 解码），否则 httpx 下载 `image_url`；下载失败按 `download_retry_backoff_seconds` 退避重试共 `download_retries` 次，最终抛 `ImageDownloadError`
  - `decode_and_validate(image_bytes: bytes, cfg: Config) -> bytes`：PIL 打开验证格式（JPEG/PNG/WebP），校验 ≤ `image_max_bytes`，失败抛 `ImageDecodeError`；返回原始字节
  - 测试用 sleep 注入：`fetch_image(..., sleep=asyncio.sleep)` 参数可替换，测试传空操作加速

- [ ] **Step 1: 写失败测试**

`tests/unit/test_downloader.py`:

```python
import base64
import io

import httpx
import pytest
from PIL import Image

from app.config import Config
from app.domain import ImageDecodeError, ImageDownloadError, PostRecord
from app.pipeline.downloader import decode_and_validate, fetch_image


def _cfg(**kw):
    return Config(db_path="/tmp/x.db", **kw)


def _png_bytes(color=(1, 2, 3), fmt="PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), color).save(buf, format=fmt)
    return buf.getvalue()


async def _no_sleep(_):
    pass


async def test_base64_channel_decodes():
    raw = _png_bytes()
    rec = PostRecord(post_id="p", text="t",
                     image_base64=base64.b64encode(raw).decode())
    got = await fetch_image(rec, _cfg(), sleep=_no_sleep)
    assert got == raw


async def test_url_channel_success(monkeypatch):
    raw = _png_bytes()

    async def fake_get(self, url, timeout):
        return httpx.Response(200, content=raw, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    rec = PostRecord(post_id="p", text="t", image_url="http://x/a.png")
    assert await fetch_image(rec, _cfg(), sleep=_no_sleep) == raw


async def test_url_channel_retries_then_fails(monkeypatch):
    calls = {"n": 0}

    async def fake_get(self, url, timeout):
        calls["n"] += 1
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    rec = PostRecord(post_id="p", text="t", image_url="http://x/a.png")
    with pytest.raises(ImageDownloadError):
        await fetch_image(rec, _cfg(download_retries=3), sleep=_no_sleep)
    assert calls["n"] == 3  # 共尝试 download_retries 次


async def test_http_404_is_download_error(monkeypatch):
    async def fake_get(self, url, timeout):
        return httpx.Response(404, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    rec = PostRecord(post_id="p", text="t", image_url="http://x/a.png")
    with pytest.raises(ImageDownloadError):
        await fetch_image(rec, _cfg(download_retries=1), sleep=_no_sleep)


async def test_neither_source_raises():
    rec = PostRecord(post_id="p", text="t")
    with pytest.raises(ImageDownloadError):
        await fetch_image(rec, _cfg(), sleep=_no_sleep)


def test_decode_validates_format_and_size():
    ok = _png_bytes(fmt="JPEG")
    assert decode_and_validate(ok, _cfg()) == ok
    with pytest.raises(ImageDecodeError):
        decode_and_validate(b"not an image", _cfg())
    with pytest.raises(ImageDecodeError):  # 超限
        decode_and_validate(_png_bytes(), _cfg(image_max_bytes=10))
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/unit/test_downloader.py -v`
Expected: FAIL（`ModuleNotFoundError: app.pipeline.downloader`）

- [ ] **Step 3: 最小实现**

`app/pipeline/downloader.py`:

```python
"""图片获取：base64 解码 / httpx 下载（退避重试）+ 格式大小校验（spec §4.3/§6.1）。"""
import asyncio
import base64
import io

import httpx
from PIL import Image, UnidentifiedImageError

from app.config import Config
from app.domain import ImageDecodeError, ImageDownloadError, PostRecord

_ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


async def fetch_image(rec: PostRecord, cfg: Config, sleep=asyncio.sleep) -> bytes:
    if rec.image_base64:
        try:
            return base64.b64decode(rec.image_base64, validate=True)
        except Exception as e:
            raise ImageDownloadError(f"base64 解码失败: {e}") from e
    if not rec.image_url:
        raise ImageDownloadError("image_url 与 image_base64 均缺失")

    last_err: Exception | None = None
    for attempt in range(cfg.download_retries):
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                resp = await client.get(rec.image_url, timeout=cfg.download_timeout_seconds)
            if resp.status_code != 200:
                raise ImageDownloadError(f"HTTP {resp.status_code}")
            return resp.content
        except ImageDownloadError as e:
            last_err = e
            if resp.status_code in (404, 403):  # 确定性失败不重试
                break
        except Exception as e:
            last_err = e
        if attempt < cfg.download_retries - 1:
            await sleep(cfg.download_retry_backoff_seconds[attempt])
    raise ImageDownloadError(f"下载失败（{cfg.download_retries} 次尝试）: {last_err}")


def decode_and_validate(image_bytes: bytes, cfg: Config) -> bytes:
    if len(image_bytes) > cfg.image_max_bytes:
        raise ImageDecodeError(f"图片超限: {len(image_bytes)} > {cfg.image_max_bytes}")
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.verify()
            fmt = img.format
    except (UnidentifiedImageError, OSError) as e:
        raise ImageDecodeError(f"图片解码失败: {e}") from e
    if fmt not in _ALLOWED_FORMATS:
        raise ImageDecodeError(f"不支持的图片格式: {fmt}")
    return image_bytes
```

- [ ] **Step 4: 运行确认通过并提交**

Run: `pytest tests/unit/test_downloader.py -v` → Expected: 6 passed

```bash
git add app/pipeline/downloader.py tests/unit/test_downloader.py && git commit -m "feat: 图片下载与解码（退避重试+格式校验）"
```

---

### Task 9: 单帖处理编排（Pipeline Processor）

**Files:**
- Create: `app/pipeline/processor.py`
- Test: `tests/unit/test_processor.py`

**Interfaces:**
- Consumes: Task 1 `Config`、Task 2 全部领域类型、Task 3 `SqliteStore`、Task 4 `Embedder` 协议、Task 6 `StageTimer`、Task 7 `IndexService`、Task 8 `fetch_image/decode_and_validate`
- Produces:
  - `class Processor(store, index, embedder, cfg)`
  - `async process(post_id: str) -> None`：完整单帖流程（顺序铁律：下载→解码→编码(线程池)→persist_vectors→search→mark_indexed→index.add；失败→mark_failed 后 re-raise `ProcessingError`）；超窗帖（created_at < cutoff）短路：`mark_indexed` 记零分结果 note="created_at_outside_window"，不提特征不注册（spec §3.4⑧）
  - `async to_thread` 包装 embedder 同步调用
  - 日志：处理完成输出 `post_id=... phase=complete total=...ms breakdown: ...`；total > `slow_task_warn_seconds*1000` 输出 WARNING；失败输出 `phase=failed reason=<code> <已完成阶段耗时>`
  - `encode_vector` 来自 Task 3（向量落库序列化）

- [ ] **Step 1: 写失败测试**

`tests/unit/test_processor.py`:

```python
import base64
import io
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from PIL import Image

from app.config import Config
from app.domain import IndexStatus, PostRecord
from app.embedder.fake import FakeEmbedder
from app.index.service import IndexService
from app.pipeline.processor import Processor
from app.store.sqlite_store import SqliteStore

NOW = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)


def _png(color=(9, 9, 9)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
async def env(tmp_db):
    cfg = Config(db_path=tmp_db)
    store = SqliteStore(tmp_db)
    await store.init()
    index = IndexService()
    proc = Processor(store=store, index=index, embedder=FakeEmbedder(), cfg=cfg)
    yield cfg, store, index, proc
    await store.close()


async def _submit(store, pid, png, text, created_at):
    rec = PostRecord(post_id=pid, image_base64=base64.b64encode(png).decode(),
                     text=text, created_at=created_at)
    assert await store.upsert_post(rec)


async def test_first_post_scores_zero(env):
    _, store, index, proc = env
    await _submit(store, "p1", _png(), "第一帖", NOW)
    await proc.process("p1")
    res = await store.get_result("p1")
    assert res.max_sim == 0.0 and res.matched_post_id is None
    assert (await store.get_post("p1")).status == IndexStatus.INDEXED
    assert index.size == 1  # 已注册


async def test_duplicate_image_high_similarity(env):
    _, store, index, proc = env
    png = _png()
    await _submit(store, "p1", png, "第一帖", NOW - timedelta(hours=1))
    await proc.process("p1")
    await _submit(store, "p2", png, "换个文案重发", NOW)
    await proc.process("p2")
    res = await store.get_result("p2")
    assert res.sim_image > 0.99
    assert res.max_sim == res.sim_image
    assert res.matched_post_id == "p1"


async def test_outside_window_short_circuits(env):
    """spec §3.4⑧：created_at 超窗 → 零分短路，不提特征不注册。"""
    cfg, store, index, proc = env
    await _submit(store, "old", _png(), "回填老帖", NOW - timedelta(days=40))
    await proc.process("old")
    res = await store.get_result("old")
    assert res.max_sim == 0.0 and res.note == "created_at_outside_window"
    assert index.size == 0  # 未注册索引


async def test_failure_marks_failed_and_reraises(env):
    _, store, index, proc = env
    rec = PostRecord(post_id="bad", image_base64=base64.b64encode(b"broken").decode(),
                     text="坏图", created_at=NOW)
    await store.upsert_post(rec)
    from app.domain import ProcessingError
    with pytest.raises(ProcessingError):
        await proc.process("bad")
    post = await store.get_post("bad")
    assert post.status == IndexStatus.FAILED
    assert post.note == "image_decode_failed"
    assert index.size == 0


async def test_order_vector_persisted_before_index_add(env):
    """顺序铁律：mark_indexed 后向量已在库（重启可重建）。"""
    _, store, index, proc = env
    await _submit(store, "p1", _png(), "t", NOW)
    await proc.process("p1")
    rows = await store.list_indexed_within(NOW - timedelta(days=30))
    assert len(rows) == 1 and rows[0][1] is not None and rows[0][2] is not None
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/unit/test_processor.py -v`
Expected: FAIL（`ModuleNotFoundError: app.pipeline.processor`）

- [ ] **Step 3: 最小实现**

`app/pipeline/processor.py`:

```python
"""单帖处理编排（spec §4.1 顺序铁律 + §7.4 耗时打点 + §3.4⑧ 超窗短路）。"""
import asyncio
import logging

import numpy as np

from app.config import Config
from app.domain import ErrorCode, SimilarityResult, utcnow, window_cutoff
from app.embedder.base import Embedder
from app.index.service import IndexService
from app.pipeline.downloader import decode_and_validate, fetch_image
from app.store.base import PostStore
from app.store.sqlite_store import encode_vector
from app.timer import StageTimer

logger = logging.getLogger("pipeline")


class Processor:
    def __init__(self, store: PostStore, index: IndexService,
                 embedder: Embedder, cfg: Config):
        self.store = store
        self.index = index
        self.embedder = embedder
        self.cfg = cfg

    async def process(self, post_id: str) -> None:
        timer = StageTimer()
        rec = await self.store.get_post(post_id)
        if rec is None:
            logger.warning("post_id=%s 不存在，跳过", post_id)
            return
        try:
            # ---- 超窗短路（spec §3.4⑧）----
            cutoff = window_cutoff(utcnow(), self.cfg.window_days)
            if rec.created_at < cutoff:
                await self.store.mark_indexed(post_id, SimilarityResult(
                    post_id=post_id, max_sim=0.0, sim_image=0.0, sim_text=0.0,
                    matched_post_id=None, computed_at=utcnow(),
                    note="created_at_outside_window"))
                logger.info("post_id=%s phase=skip reason=created_at_outside_window", post_id)
                return

            # ---- 无图帖：前置字段检查跳过图片通道，零向量落库（sim 恒 0 不误匹配）----
            has_image = bool(rec.image_url or rec.image_base64)
            if has_image:
                with timer.stage("download"):
                    raw = await fetch_image(rec, self.cfg)
                    image_bytes = decode_and_validate(raw, self.cfg)
                with timer.stage("image_embed"):
                    image_vec = await asyncio.to_thread(self.embedder.embed_image, image_bytes)
            else:
                image_vec = np.zeros(self.embedder.image_dim, dtype=np.float32)
            with timer.stage("text_embed"):
                # 检索用 query 前缀；落库/注册用 passage 前缀（库存侧契约，spec §5.1）
                text_vec = await asyncio.to_thread(
                    lambda: self.embedder.embed_text(rec.text, prefix="query: "))
                passage_vec = await asyncio.to_thread(
                    lambda: self.embedder.embed_text(rec.text, prefix="passage: "))

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
            await self.store.mark_indexed(post_id, result)

            with timer.stage("faiss_add"):
                # 注册用 passage 前缀向量（库存侧契约，spec §5.1）
                await asyncio.to_thread(
                    self.index.add, post_id, rec.created_at, image_vec, passage_vec)

            self._log_complete(post_id, timer)
        except Exception as e:
            code = getattr(e, "code", None)
            if code is None:  # 意外异常兜底：记 INTERNAL_ERROR，避免 pending 被无限重试
                code = ErrorCode.INTERNAL_ERROR
            await self.store.mark_failed(post_id, code)
            logger.error("post_id=%s phase=failed reason=%s %s",
                         post_id, code.value, timer.summary())
            raise

    def _log_complete(self, post_id: str, timer: StageTimer):
        total = timer.total_ms()
        msg = f"post_id={post_id} phase=complete total={total:.0f}ms {timer.summary()}"
        if total > self.cfg.slow_task_warn_seconds * 1000:
            logger.warning("SLOW %s", msg)
        else:
            logger.info(msg)
```

注意文字向量的前缀契约：**检索时新帖用 `query:` 前缀**，但**注册进索引供后续帖子匹配的必须是 `passage:` 前缀向量**——因此 passage 向量在 search 前多编码一次并用于 `index.add`。FakeEmbedder 对前缀敏感，测试中两帖文字相同时 passage 向量一致，不影响断言。

- [ ] **Step 4: 运行确认通过并提交**

Run: `pytest tests/unit/test_processor.py -v` → Expected: 5 passed（全量回归：`pytest -q`）

```bash
git add app/pipeline/processor.py tests/unit/test_processor.py && git commit -m "feat: 单帖处理编排（顺序铁律+超窗短路+耗时打点）"
```

---

### Task 10: API 层（schemas + routes）

**Files:**
- Create: `app/api/schemas.py`, `app/api/routes.py`
- Test: `tests/api/test_routes.py`

**Interfaces:**
- Consumes: Task 1 `Config`、Task 2 `IndexStatus`、Task 3 `SqliteStore`、Task 4 `FakeEmbedder`、Task 7 `IndexService`、Task 9 `Processor`；以及一个由调用方注入的 `enqueue(post_id) -> bool` 回调（Task 11 提供真队列，测试用桩）
- Produces:
  - `SubmitPostRequest`（post_id 1-128 字符；text ≤5000 必填；image_url/image_base64 至少一个）、`SubmitPostResponse`、`SimilarityResponse`、`ErrorResponse`
  - `create_router(store, processor, cfg, enqueue, health_state) -> APIRouter`，health_state 为可变 dict（`{"ready": bool}`，Task 11 生命周期管理）
  - 端点行为（spec §6）：
    - `POST /posts`：新帖→写库(pending)→enqueue→202；重复 post_id→200+当前状态；校验失败 422（统一错误格式）
    - `GET /posts/{post_id}/similarity`：pending→202；indexed→200 结果；failed→200+reason；未知→404
    - `GET /health`：ready=false 时 503
    - `POST /admin/replay`：body 可选 `{"post_id": "..."}`（重置单帖 failed→pending 并入队）或 `{"mode": "compact"}`（压缩索引）

- [ ] **Step 1: 写失败测试**

`tests/api/test_routes.py`:

```python
import base64
import io

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from PIL import Image

from app.api.routes import create_router, register_error_handlers
from app.config import Config
from app.embedder.fake import FakeEmbedder
from app.index.service import IndexService
from app.pipeline.processor import Processor
from app.store.sqlite_store import SqliteStore


def _png() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (5, 5, 5)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
async def env(tmp_db):
    """纯 ASGI 测试：httpx.AsyncClient + ASGITransport，与 pytest-asyncio 同循环。"""
    cfg = Config(db_path=tmp_db)
    store = SqliteStore(tmp_db)
    await store.init()
    index = IndexService()
    proc = Processor(store=store, index=index, embedder=FakeEmbedder(), cfg=cfg)
    enqueued: list[str] = []
    app = FastAPI()
    app.include_router(create_router(store, proc, cfg, enqueued.append, {"ready": True}))
    register_error_handlers(app)
    async with AsyncClient(transport=ASGITransport(app=app),
                           base_url="http://test") as client:
        yield client, store, proc, enqueued
    await store.close()


async def test_submit_returns_202_and_enqueues(env):
    client, _, _, enqueued = env
    r = await client.post("/posts", json={"post_id": "p1", "image_base64": _png(), "text": "你好"})
    assert r.status_code == 202
    assert r.json() == {"post_id": "p1", "status": "pending"}
    assert enqueued == ["p1"]


async def test_duplicate_post_id_is_idempotent(env):
    client, _, _, enqueued = env
    body = {"post_id": "p1", "image_base64": _png(), "text": "你好"}
    assert (await client.post("/posts", json=body)).status_code == 202
    r = await client.post("/posts", json=body)
    assert r.status_code == 200
    assert r.json()["status"] == "pending"
    assert enqueued == ["p1"]  # 未二次入队


async def test_validation_errors(env):
    client, _, _, _ = env
    r = await client.post("/posts", json={"post_id": "p", "text": "无图"})
    assert r.status_code == 422 and "error" in r.json()
    r = await client.post("/posts", json={"image_base64": _png(), "text": "无id"})
    assert r.status_code == 422
    r = await client.post("/posts", json={"post_id": "x" * 129, "image_base64": _png(), "text": "t"})
    assert r.status_code == 422
    r = await client.post("/posts", json={"post_id": "p", "image_base64": _png(), "text": "t" * 5001})
    assert r.status_code == 422


async def test_similarity_three_states(env):
    client, _, proc, _ = env
    assert (await client.get("/posts/nope/similarity")).status_code == 404
    await client.post("/posts", json={"post_id": "p1", "image_base64": _png(), "text": "你好"})
    r = await client.get("/posts/p1/similarity")
    assert r.status_code == 202 and r.json()["status"] == "pending"

    await proc.process("p1")  # 同循环直接驱动处理器
    r = await client.get("/posts/p1/similarity")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "indexed" and body["max_sim"] == 0.0
    assert set(body) >= {"max_sim", "sim_image", "sim_text", "matched_post_id", "computed_at"}


async def test_failed_state_returns_reason(env):
    client, _, proc, _ = env
    await client.post("/posts", json={"post_id": "bad",
                                      "image_base64": base64.b64encode(b"x").decode(),
                                      "text": "t"})
    from app.domain import ProcessingError
    with pytest.raises(ProcessingError):
        await proc.process("bad")
    r = await client.get("/posts/bad/similarity")
    assert r.status_code == 200
    assert r.json() == {"status": "failed", "reason": "image_decode_failed"}


async def test_health_not_ready_503(tmp_db):
    cfg = Config(db_path=tmp_db)
    store = SqliteStore(tmp_db)
    await store.init()
    proc = Processor(store=store, index=IndexService(),
                     embedder=FakeEmbedder(), cfg=cfg)
    app = FastAPI()
    app.include_router(create_router(store, proc, cfg, lambda p: None, {"ready": False}))
    register_error_handlers(app)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        assert (await c.get("/health")).status_code == 503
    await store.close()
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/api/test_routes.py -v`
Expected: FAIL（`ModuleNotFoundError: app.api.routes`）

- [ ] **Step 3: 最小实现**

`app/api/schemas.py`:

```python
"""API 数据契约（spec §6）。"""
from pydantic import BaseModel, Field, model_validator


class SubmitPostRequest(BaseModel):
    post_id: str = Field(min_length=1, max_length=128)
    image_url: str | None = None
    image_base64: str | None = None
    text: str = Field(min_length=1, max_length=5000)
    created_at: str | None = None  # ISO8601 UTC，可选（spec §3.4⑧）

    @model_validator(mode="after")
    def _require_image(self):
        if not self.image_url and not self.image_base64:
            raise ValueError("image_url 与 image_base64 至少提供一个")
        return self


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
```

`app/api/routes.py`:

```python
"""API 端点（spec §6）。enqueue 与 health_state 由 service 层注入。"""
from datetime import datetime, timezone
from typing import Callable

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.api.schemas import SubmitPostRequest, SubmitPostResponse
from app.config import Config
from app.domain import IndexStatus, PostRecord, utcnow
from app.pipeline.processor import Processor
from app.store.base import PostStore


def create_router(store: PostStore, processor: Processor, cfg: Config,
                  enqueue: Callable[[str], bool], health_state: dict) -> APIRouter:
    router = APIRouter()

    @router.post("/posts", status_code=202, response_model=SubmitPostResponse)
    async def submit_post(req: SubmitPostRequest):
        existing = await store.get_post(req.post_id)
        if existing is not None:  # 幂等：不重新计算
            return JSONResponse(status_code=200, content={
                "post_id": req.post_id, "status": existing.status.value})
        created_at = utcnow()
        if req.created_at:
            created_at = datetime.fromisoformat(req.created_at)
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
        rec = PostRecord(post_id=req.post_id, text=req.text,
                         image_url=req.image_url, image_base64=req.image_base64,
                         created_at=created_at)
        await store.upsert_post(rec)
        enqueue(req.post_id)
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
            removed = processor.index.compact(utcnow(), cfg.window_days)
            return {"mode": "compact", "removed": removed}
        pid = payload.get("post_id")
        if pid:
            ok = await store.reset_failed_to_pending(pid)
            if ok:
                enqueue(pid)
            return {"post_id": pid, "reset": ok}
        # 无参数：全量回放由 service 层触发（返回提示）
        return {"hint": "提供 post_id 或 mode=compact"}

    return router
```

`app/api/routes.py` 另外导出 `register_error_handlers(app)`：

```python
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def register_error_handlers(app: FastAPI) -> None:
    """统一错误格式 {"error": {code, message}}（spec §6）。必须在 app 层注册。"""

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={
            "error": {"code": "validation_error", "message": str(exc.errors()[:3])}})
```

`create_router` 内不含 exception_handler（router 级不生效），由 `build_app`（Task 11）与测试 fixture 统一调用 `register_error_handlers`。

- [ ] **Step 4: 运行确认通过并提交**

Run: `pytest tests/api/ -v` → Expected: 6 passed

```bash
git add app/api/ tests/api/ && git commit -m "feat: API 层（幂等提交+三态查询+health+admin replay）"
```


---

### Task 11: Service 装配（生命周期 / Worker / 队列 / 兜底扫描 / 优雅停机）

**Files:**
- Create: `app/service.py`, `app/main.py`
- Modify: `app/api/routes.py`（enqueue 调用点兼容异步，见 Step 3 说明）
- Test: `tests/integration/test_service.py`

**Interfaces:**
- Consumes: Task 1-10 全部模块
- Produces:
  - `build_app(cfg: Config, embedder: Embedder | None = None) -> FastAPI`：唯一装配入口。embedder 缺省 `ClipE5Embedder(cfg.device)`（生产），测试注入 FakeEmbedder
  - lifespan 启动序列（spec §3.4②）：`embedder.load()`（线程池）→ `store.init()` → 从库重建索引（`list_indexed_within(cutoff)` + `decode_vector`）→ 回放全部 pending → `health_state["ready"]=True` → 启动 worker 与 sweep 协程
  - `enqueue(post_id)`：async；`put_nowait` 满则等待 ≤`queue_wait_seconds`，超时仅告警（帖子已落库 pending，sweep 兜底）
  - worker：串行消费 + `in_flight` 集合（spec §7.3）；`ProcessingError` 静默（已 mark_failed），其他异常 mark_failed(INTERNAL_ERROR)
  - 优雅停机（spec §7.5）：lifespan 退出时投递哨兵、等待 worker ≤`graceful_shutdown_seconds`、取消 sweep、`store.close()`
  - `app/main.py`：`python -m app.main` 以 uvicorn 启动（host/port 可 env 覆盖）

- [ ] **Step 1: 写失败测试**

`tests/integration/test_service.py`:

```python
import asyncio
import base64
import io

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from app.config import Config
from app.domain import SimilarityResult, PostRecord, utcnow, IndexStatus
from app.embedder.fake import FakeEmbedder
from app.service import build_app
from app.store.sqlite_store import SqliteStore, encode_vector


def _png(color=(7, 7, 7)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (12, 12), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


async def _wait_until(pred, timeout=5.0, interval=0.05):
    async def wrap():
        while True:
            if await pred():
                return
            await asyncio.sleep(interval)
    await asyncio.wait_for(wrap(), timeout)


@pytest.fixture
async def app_env(tmp_db):
    cfg = Config(db_path=tmp_db, sweep_interval_seconds=600)
    app = build_app(cfg, embedder=FakeEmbedder())
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app),
                               base_url="http://test") as client:
            yield client, cfg
    # lifespan 退出即完成优雅停机


async def test_end_to_end_submit_and_result(app_env):
    client, _ = app_env
    r = await client.post("/posts", json={"post_id": "p1", "image_base64": _png(), "text": "第一帖"})
    assert r.status_code == 202

    async def done():
        return (await client.get("/posts/p1/similarity")).json().get("status") == "indexed"
    await _wait_until(done)
    body = (await client.get("/posts/p1/similarity")).json()
    assert body["max_sim"] == 0.0  # 首帖


async def test_duplicate_detected_by_worker(app_env):
    client, _ = app_env
    png = _png()
    await client.post("/posts", json={"post_id": "p1", "image_base64": png, "text": "原帖"})

    async def p1_done():
        return (await client.get("/posts/p1/similarity")).json().get("status") == "indexed"
    await _wait_until(p1_done)

    await client.post("/posts", json={"post_id": "p2", "image_base64": png, "text": "换文案重发"})

    async def p2_done():
        return (await client.get("/posts/p2/similarity")).json().get("status") == "indexed"
    await _wait_until(p2_done)
    body = (await client.get("/posts/p2/similarity")).json()
    assert body["matched_post_id"] == "p1" and body["sim_image"] > 0.99


async def test_startup_rebuild_and_replay(tmp_db):
    """spec §3.4②：重启后索引从库重建 + pending 回放，零数据丢失。"""
    cfg = Config(db_path=tmp_db)
    png = _png()

    # ---- 第一生命周期：完整索引 p1 ----
    app = build_app(cfg, embedder=FakeEmbedder())
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            await c.post("/posts", json={"post_id": "p1", "image_base64": png, "text": "原帖"})
            async def done():
                return (await c.get("/posts/p1/similarity")).json().get("status") == "indexed"
            await _wait_until(done)

    # ---- 模拟崩溃残留：直接向库插入 pending 的 p2（队列任务已丢失）----
    store = SqliteStore(tmp_db)
    await store.init()
    await store.upsert_post(PostRecord(post_id="p2", text="重发帖", image_base64=png))
    await store.close()

    # ---- 第二生命周期：启动重建索引 + 回放 p2 ----
    app2 = build_app(cfg, embedder=FakeEmbedder())
    async with app2.router.lifespan_context(app2):
        async with AsyncClient(transport=ASGITransport(app=app2), base_url="http://t") as c:
            async def p2_done():
                return (await c.get("/posts/p2/similarity")).json().get("status") == "indexed"
            await _wait_until(p2_done)
            body = (await c.get("/posts/p2/similarity")).json()
            assert body["matched_post_id"] == "p1"  # p1 由启动重建恢复进索引
            health = (await c.get("/health")).json()
            assert health["status"] == "ok" and health["index_count"] == 2


async def test_sweep_reenqueues_stale_pending(tmp_db):
    cfg = Config(db_path=tmp_db, sweep_interval_seconds=1, pending_stale_seconds=0)
    store = SqliteStore(tmp_db)
    await store.init()
    await store.upsert_post(PostRecord(post_id="stale", text="滞留帖", image_base64=_png()))
    await store.close()

    app = build_app(cfg, embedder=FakeEmbedder())
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            async def done():
                return (await c.get("/posts/stale/similarity")).json().get("status") == "indexed"
            await _wait_until(done, timeout=10.0)  # sweep 兜底处理
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/integration/test_service.py -v`
Expected: FAIL（`ModuleNotFoundError: app.service`）

- [ ] **Step 3: 最小实现**

先修改 `app/api/routes.py` 两处 `enqueue(...)` 调用为同步/异步兼容（测试桩是同步函数，生产是协程）：

```python
import inspect
...
result = enqueue(req.post_id)          # submit_post 内
if inspect.isawaitable(result):
    await result
...
result = enqueue(pid)                  # admin_replay 内
if inspect.isawaitable(result):
    await result
```

`app/service.py`:

```python
"""应用装配与生命周期（spec §3.4②启动序列 / §7.3 in-flight / §7.5 优雅停机）。"""
import asyncio
import contextlib
import logging
from datetime import timedelta

from fastapi import FastAPI

from app.api.routes import create_router, register_error_handlers
from app.config import Config
from app.domain import ErrorCode, ProcessingError, utcnow, window_cutoff
from app.embedder.base import Embedder
from app.index.service import IndexService
from app.pipeline.processor import Processor
from app.store.sqlite_store import SqliteStore, decode_vector

logger = logging.getLogger("service")


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
            if post_id == "":  # 停机哨兵
                queue.task_done()
                break
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
            health_state["live"] = {"queue_size": queue.qsize(),
                                    "in_flight": len(in_flight)}

    app = FastAPI(title="post-similarity")
    register_error_handlers(app)
    app.include_router(create_router(store, processor, cfg, enqueue, health_state))

    @app.on_event("startup")
    async def startup():
        import os
        os.makedirs(os.path.dirname(cfg.db_path) or ".", exist_ok=True)
        await asyncio.to_thread(embedder.load)
        await store.init()
        # ---- 启动序列（spec §3.4②）：重建索引 → 回放 pending ----
        cutoff = window_cutoff(utcnow(), cfg.window_days)
        rows = await store.list_indexed_within(cutoff)
        index.rebuild([(pid, ts, decode_vector(iv), decode_vector(tv))
                       for pid, iv, tv, ts in rows])
        logger.info("启动重建索引完成：%d 条", index.size)
        far_future = utcnow() + timedelta(days=1)
        for pid in await store.list_stale_pending(far_future):  # 全部 pending 回放
            await queue.put(pid)
        tasks["worker"] = asyncio.create_task(worker())
        tasks["sweep"] = asyncio.create_task(sweep())
        tasks["stats"] = asyncio.create_task(stats())
        health_state["ready"] = True

    @app.on_event("shutdown")
    async def shutdown():
        health_state["ready"] = False
        with contextlib.suppress(asyncio.QueueFull):
            queue.put_nowait("")  # 哨兵唤醒 worker
        try:
            await asyncio.wait_for(tasks["worker"], timeout=cfg.graceful_shutdown_seconds)
        except asyncio.TimeoutError:
            tasks["worker"].cancel()
        for name in ("sweep", "stats"):
            tasks[name].cancel()
        await store.close()
        logger.info("优雅停机完成")

    return app
```

`app/main.py`:

```python
"""入口：python -m app.main"""
import logging
import os

import uvicorn

from app.config import load_config
from app.service import build_app

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")

cfg = load_config()
app = build_app(cfg)

if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("SIM_HOST", "127.0.0.1"),
                port=int(os.environ.get("SIM_PORT", "8000")))
```

- [ ] **Step 4: 运行确认通过并提交**

Run: `pytest tests/integration/test_service.py -v` → Expected: 4 passed；全量回归 `pytest -q` 全绿

```bash
git add app/service.py app/main.py app/api/routes.py tests/integration/ && git commit -m "feat: Service 装配（启动重建+回放+worker+sweep+优雅停机）"
```

---

### Task 12: 崩溃恢复与不变式验收（跨进程重启）

**Files:**
- Test: `tests/integration/test_crash_recovery.py`

**Interfaces:**
- Consumes: Task 11 `build_app`、Task 3 `SqliteStore`
- Produces: spec §7.1 三条不变式的验收测试（跨两个独立生命周期实例，模拟崩溃后重启）

- [ ] **Step 1: 写失败测试**

`tests/integration/test_crash_recovery.py`:

```python
import asyncio
import base64
import io
import sqlite3

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from app.config import Config
from app.service import build_app
from app.embedder.fake import FakeEmbedder


def _png(color=(3, 4, 5)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (12, 12), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


async def _wait_until(pred, timeout=5.0):
    async def wrap():
        while True:
            if await pred():
                return
            await asyncio.sleep(0.05)
    await asyncio.wait_for(wrap(), timeout)


async def test_invariants_after_simulated_crash(tmp_db):
    """spec §7.1 验收：
    不变式1 —— indexed 帖库中必有向量与结果；
    不变式2 —— 重启后索引 == 库中 indexed 集合（重建是纯函数）；
    不变式3 —— 崩溃丢失的在途任务（pending）被回放补齐，零数据丢失。
    """
    cfg = Config(db_path=tmp_db)
    png = _png()

    # 实例一：索引 p1，随后"崩溃"（lifespan 直接退出，不走优雅停机流程验证重启自愈）
    app = build_app(cfg, embedder=FakeEmbedder())
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            await c.post("/posts", json={"post_id": "p1", "image_base64": png, "text": "原帖"})
            async def p1():
                return (await c.get("/posts/p1/similarity")).json().get("status") == "indexed"
            await _wait_until(p1)

    # 崩溃现场取证：p1 的向量与结果确实在库（不变式1）
    conn = sqlite3.connect(tmp_db)
    row = conn.execute(
        "SELECT image_vec IS NOT NULL, text_vec IS NOT NULL, status FROM posts "
        "WHERE post_id='p1'").fetchone()
    res = conn.execute("SELECT COUNT(*) FROM results WHERE post_id='p1'").fetchone()
    conn.close()
    assert row == (1, 1, "indexed") and res[0] == 1

    # 模拟崩溃丢失在途任务：p2 直接以 pending 落库（未出队即进程死亡）
    conn = sqlite3.connect(tmp_db)
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO posts(post_id,text,image_base64,created_at,enqueued_at,status) "
        "VALUES('p2','重发帖',?,?,?, 'pending')", (png, now, now))
    conn.commit()
    conn.close()

    # 实例二：全新进程（空索引）启动
    app2 = build_app(cfg, embedder=FakeEmbedder())
    async with app2.router.lifespan_context(app2):
        async with AsyncClient(transport=ASGITransport(app=app2), base_url="http://t") as c:
            # 不变式2：p1 经启动重建回到索引 —— 通过 p2 能匹配到 p1 验证
            async def p2():
                return (await c.get("/posts/p2/similarity")).json().get("status") == "indexed"
            await _wait_until(p2)
            body = (await c.get("/posts/p2/similarity")).json()
            assert body["matched_post_id"] == "p1"
            assert body["sim_image"] > 0.99  # 不变式3：pending 回放补齐，零丢失

            health = (await c.get("/health")).json()
            assert health["index_count"] == 2
```

- [ ] **Step 2: 运行确认失败→通过**

Run: `pytest tests/integration/test_crash_recovery.py -v`
Expected: 首跑应 FAIL（若 Task 11 实现有缺口）；修正后 PASS。若已 PASS 说明 Task 11 已覆盖不变式，本测试作为回归守卫保留。

- [ ] **Step 3: 全量回归并提交**

Run: `pytest -q` → Expected: 全部通过（slow 默认 skip）

```bash
git add tests/integration/test_crash_recovery.py && git commit -m "test: 崩溃恢复与三条一致性不变式验收"
```

---

### Task 13: 效果评估脚本 + README + 最终回归

**Files:**
- Create: `eval/synthetic_calibration.py`, `README.md`

**Interfaces:**
- Consumes: Task 5 `ClipE5Embedder`（需先跑过 `pytest -m slow` 完成模型下载）
- Produces: 合成校准流水线（spec §5.3/§8.4）——生成正负例对 → 计算分数分布 → 输出统计 JSON 与阈值建议

- [ ] **Step 1: 实现合成校准脚本**

`eval/synthetic_calibration.py`:

```python
"""合成校准集（spec §5.3 两阶段校准第一阶段 + §8.4）。

Demo 无生产数据，校准集由合成变体构成：
- 图片正例：同一合成图的裁剪(50%/25-75%)/缩放(0.5x,1.5x)/轻度调色变体
- 图片负例：不同随机合成图
- 文字正例：句尾加语气词/换标点/同近义改写模板
- 文字负例：无关随机句
输出：eval/calibration_report.json + 终端分布摘要与阈值建议。
阈值仅为合成意义下的起点；上生产后必须用真实流量重校（spec §5.3）。
"""
import io
import itertools
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

from app.embedder.clip_e5 import ClipE5Embedder

OUT = Path(__file__).parent / "calibration_report.json"


def make_image(seed: int) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("RGB", (640, 480), (rng.randrange(256),) * 3)
    d = ImageDraw.Draw(img)
    for _ in range(rng.randrange(5, 15)):
        x0, y0 = rng.randrange(600), rng.randrange(440)
        d.ellipse([x0, y0, x0 + rng.randrange(20, 120), y0 + rng.randrange(20, 120)],
                  fill=(rng.randrange(256), rng.randrange(256), rng.randrange(256)))
    return img


def to_bytes(img: Image.Image, fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def image_variants(img: Image.Image) -> dict[str, bytes]:
    w, h = img.size
    return {
        "crop_center_50": to_bytes(img.crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4))),
        "crop_offset_25_75": to_bytes(img.crop((0, 0, int(w * .75), int(h * .75)))),
        "resize_0.5x": to_bytes(img.resize((w // 2, h // 2))),
        "resize_1.5x": to_bytes(img.resize((int(w * 1.5), int(h * 1.5)))),
        "color_jitter": to_bytes(ImageEnhance.Color(img).enhance(1.3)),
    }


TEXT_BASES = [
    "今天去公园散步天气特别好",
    "这家店的咖啡味道真的很不错",
    "新买的耳机音质超出预期",
    "周末和朋友爬山看到了日落",
    "家里的小猫又把杯子打翻了",
]
TEXT_POS = {t: [t + "！", t.replace("。", "") + "，推荐", t[:4] + "，" + t[4:]] for t in TEXT_BASES}
TEXT_NEG = [
    "央行今日开展逆回购操作",
    "量子计算的最新进展引发讨论",
    "本市地铁新线路下月开通",
    "股市今日收盘小幅上涨",
    "秋季招聘会将在校体育馆举行",
]


def main():
    emb = ClipE5Embedder(device="cpu")
    emb.load()

    img_scores: dict[str, list[float]] = {}
    for seed in range(10):  # 10 个基础图
        base_img = make_image(seed)
        v_base = emb.embed_image(to_bytes(base_img))
        for name, variant in image_variants(base_img).items():
            img_scores.setdefault(name, []).append(float(v_base @ emb.embed_image(variant)))
        for other in range(seed + 1, seed + 3):  # 负例：相邻 seed 的图
            img_scores.setdefault("negative", []).append(
                float(v_base @ emb.embed_image(to_bytes(make_image(other)))))

    txt_pos, txt_neg = [], []
    for base in TEXT_BASES:
        vb = emb.embed_text(base, prefix="passage: ")
        for pos in TEXT_POS[base]:
            txt_pos.append(float(vb @ emb.embed_text(pos, prefix="query: ")))
        for neg in TEXT_NEG:
            txt_neg.append(float(vb @ emb.embed_text(neg, prefix="query: ")))

    def stat(xs):
        xs = sorted(xs)
        return {"n": len(xs), "min": round(xs[0], 4), "median": round(xs[len(xs) // 2], 4),
                "max": round(xs[-1], 4)}

    report = {
        "image": {k: stat(v) for k, v in img_scores.items()},
        "text_positive": stat(txt_pos),
        "text_negative": stat(txt_neg),
        "note": "合成校准仅为起点；上生产后必须真实流量重校（spec §5.3）",
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # 阈值建议：正例最小值与负例最大值的中点（各信号独立）
    pos_min = min(min(v) for k, v in img_scores.items() if k != "negative")
    neg_max = max(img_scores["negative"])
    print(f"\n[图片] 正例min={pos_min:.3f} 负例max={neg_max:.3f} "
          f"建议分割点={(pos_min + neg_max) / 2:.3f}")
    print(f"[文字] 正例min={min(txt_pos):.3f} 负例max={max(txt_neg):.3f} "
          f"建议分割点={(min(txt_pos) + max(txt_neg)) / 2:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行校准脚本（需真模型）**

Run: `python eval/synthetic_calibration.py`
Expected: 输出分布 JSON；检查 `crop_center_50` 的 min 是否 ≥0.8（spec §11 决议1 的裁剪鲁棒性基线）。若紧裁剪掉分明显，记录到报告 note 并按 spec 处置（阈值校准问题，暂不换模型）。

- [ ] **Step 3: README**

`README.md`（内容要点，逐项写全）：
- 项目一句话介绍 + 指向 `docs/superpowers/specs/2026-08-23-post-similarity-design.md`
- 环境准备：`uv venv --python 3.11 && uv pip install -e ".[dev]"`；首次运行 `pytest -m slow` 下载模型
- 启动：`python -m app.main`（env：SIM_DB_PATH/SIM_PORT/SIM_DEVICE 等，列表指向 `app/config.py`）
- 联调 curl 示例：提交帖（base64 与 url 两种）、轮询结果、health、admin replay
- 测试：`pytest -q`（单元/API/集成）、`pytest -m slow`（真模型冒烟）、`python eval/synthetic_calibration.py`（校准）
- 已知边界（摘自 spec §9）：Demo 单图、failed 盲区接受、短文本虚高风险、阈值需生产重校

- [ ] **Step 4: 最终全量回归并提交**

Run: `pytest -q` → Expected: 全部通过

```bash
git add eval/ README.md && git commit -m "feat: 合成校准评估脚本与 README"
```

---

## 计划自查结论

**1. Spec 覆盖核对**（spec 章节 → 任务）：
- §3.2/3.3 组件与模块 → Task 4/7/3/9/10/11；§3.4① 窗口过滤+压缩 → Task 7（filter/compact）+ Task 10（admin compact）
- §3.4② 启动重建+回放 → Task 11；§3.4⑦ 单 Worker → Task 11（单 worker 协程）；§3.4⑧ 超窗短路/created_at → Task 9/10
- §4 数据流全链路 → Task 9/11；§4.2 幂等 → Task 3/10；§4.3 失败策略 → Task 8/9/11；兜底扫描 → Task 11；§4.4 三态 → Task 10
- §5.1 模型细节（全图 resize/e5 前缀）→ Task 5/9；§5.2 自适应 K → Task 7；§5.3 阈值外置 → Task 1（config）+ Task 13（校准）
- §6 API 全部 → Task 10/11；§6.4 请求体限制属反代层部署事项 → README 记录
- §7.1 不变式 → Task 12 验收；§7.2 故障全景 → Task 8/9/11；§7.3 in-flight → Task 11；§7.4 耗时打点 → Task 6/9；§7.5 优雅停机 → Task 11
- §8.1 FakeEmbedder/sqlite 线程约束 → Task 4/3；§8.2-8.3 测试层次与场景 → Task 9/10/11/12；§8.4 效果评估 → Task 13
- §9 假设边界 → README；§10 配置项 → Task 1

**2. 类型一致性**：`SearchHit`（Task 7 定义，Task 9 消费 `hit.sim_image/sim_text/matched_post_id/adaptive_expanded`）；`encode_vector/decode_vector`（Task 3 定义，Task 9/11 消费）；`register_error_handlers`（Task 10 定义，Task 11 消费）；`ProcessingError.code`（Task 2 定义，Task 9/11 消费）——签名一致。

**3. 占位符扫描**：无 TBD/TODO；所有代码步骤含完整代码块。

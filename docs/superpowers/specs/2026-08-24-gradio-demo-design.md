# Gradio 交互演示（Demo）设计文档

- 日期：2026-08-24
- 状态：待评审
- 关联：基于《帖子相似度检测系统设计文档》（2026-08-23）已交付的核心链路

---

## 1. 背景与目标

核心相似度检测链路（下载 → 向量化 → 检索 → 落库 → 注册）已完成并可通过 HTTP API 使用。为了让使用者能直观体验检测效果、同时观测各环节性能，新增一个 Gradio 网页 Demo：

- 用户手动上传图片、输入文字，作为新帖子提交；
- 页面展示该帖子的相似度结果：综合分数（`max_sim`）以及图片、文字两个模态的分数（`sim_image` / `sim_text`）；
- 页面展示处理流水线各阶段耗时：图片下载、图片向量化、文字向量化、Faiss 检索、结果落库、向量注册；
- 页面展示与新帖最相似的 Top-3 帖子的完整内容（图片 + 文字）及各自分数。

## 2. 已确认的关键决策

| 决策点 | 结论 |
|---|---|
| 接入方式 | **进程内直接复用组件**：Gradio 应用自行装配 `SqliteStore` / `IndexService` / `ClipE5Embedder` / `Processor`，同步执行流水线并直接获取耗时与结果；不走 HTTP 调用现有 FastAPI 服务，也不与服务共进程挂载 |
| 数据来源 | **直接使用生产库** `data/similarity.db`；Demo 提交的帖子正常落库并注册进 Faiss 索引（与生产链路行为一致） |
| Embedder | 使用真实模型（CLIP ViT-B/32 + multilingual-e5-small），与生产一致 |
| 依赖归属 | `gradio` 加入 Poetry **dev 依赖组**，不进生产依赖 |

### 运行约束（重要）

Demo 与 FastAPI 生产服务**不得同时运行**：

1. 两进程各自维护一份内存 Faiss 索引，互相看不到对方的写入，检索结果会发散；
2. 两进程同时写 SQLite 存在写锁竞争。

文档与启动日志中明确提示：运行 Demo 前先停止 FastAPI 服务。Demo 启动时与生产一致地从库重建索引，保证索引与 SQLite（唯一事实源）对齐。

## 3. 对现有代码的改动（最小侵入，共两处）

### 3.1 `Processor.process` 增加返回值

现状：`process()` 返回 `None`，分阶段耗时由 `StageTimer` 记录后仅写日志。

改动：`process()` 返回 `ProcessOutcome` 数据类，并新增一处计时点：

```python
@dataclass
class ProcessOutcome:
    result: SimilarityResult | None   # 失败时为 None
    error_code: ErrorCode | None      # 失败原因，成功时为 None
    timings_ms: dict[str, float]      # StageTimer.stages 快照
    total_ms: float
```

- 新增计时点：`mark_indexed`（结果写库）目前未被 `StageTimer` 覆盖，用 `timer.stage("result_persist")` 包裹，使"结果落库"耗时可展示；
- 失败路径不再向 worker 抛异常的行为保持不变，但 `process()` 失败时同样返回 outcome（`result=None`、`error_code` 有值、附已完成阶段的耗时）。生产 Worker 继续依赖异常判定 mark_failed 已完成的语义——实现上 `process()` 在返回 outcome 前仍按原逻辑 raise，由 demo 调用层捕获并转为 outcome，或调整为返回 outcome 同时保留 worker 侧异常契约，二选一以既有测试全绿为准；
- 生产 Worker（`service.py` 中的 `worker()`）行为不变。

### 3.2 `IndexService` 新增 `top_hits()` 方法

现状：`search()` 只返回每个模态的**单条最佳命中**（`SearchHit`），无法满足 Top-3 展示。

改动：新增方法，`search()` 保持不动：

```python
def top_hits(self, image_vec, text_vec, *, exclude_id, cutoff,
             k: int = 3, candidate_k: int | None = None) -> list[TopHit]:
    """返回按 max(sim_image, sim_text) 降序的 Top-K 候选。"""
```

实现要点：

- 从图片、文字两个索引各取 Top-`candidate_k`（默认 `k * 2`）候选，复用现有窗口过滤与自排除逻辑；
- 按 `post_id` 合并：同一候选同时携带两个模态的分数，缺失模态记 0；
- 按 `max(sim_image, sim_text)` 降序排序，取前 `k` 条；
- 分数处理与 `search()` 一致（负分归零）。

`TopHit` 数据类字段：`post_id`、`sim_image`、`sim_text`。

## 4. Demo 应用结构

新增 `demo/app.py`（独立入口，不进 `app` 包），内部按职责拆分：

| 单元 | 职责 |
|---|---|
| 装配（启动时一次） | 加载 `Config` → 加载 Embedder → 打开 `SqliteStore(cfg.db_path)` → 从库重建 `IndexService`（复用 `list_indexed_within` + `decode_vector`，与生产 lifespan 相同逻辑） |
| 处理函数（纯逻辑，可测） | 接收图片字节与文字 → 生成 `post_id`（`demo-{uuid4.hex[:12]}`）→ `upsert_post` → 同步执行 `Processor.process`（`asyncio.run` 桥接）→ 调用 `top_hits()` → 查询 Top-3 帖子内容 → 返回结构化结果 |
| UI（Gradio Blocks） | 只负责渲染与事件绑定，不含业务逻辑 |

### 4.1 页面布局

- **输入区**：图片上传组件 + 文字输入框 + 提交按钮。
- **结果区**：
  1. 分数卡片：`max_sim` / `sim_image` / `sim_text`；
  2. 耗时表格，按**真实执行顺序**展示 7 行（与需求措辞的映射见表后说明）：

     | 页面标签 | 对应 timer 阶段 | 说明 |
     |---|---|---|
     | 图片下载（本地读取+校验） | `download` | 本地上传无网络下载，此阶段为文件读取 + 解码校验 |
     | 图片向量化 | `image_embed` | 无图帖跳过，显示"未执行" |
     | 文字向量化 | `text_embed` | 含 query/passage 两次编码 |
     | 向量落库 | `persist` | 图/文向量持久化到 posts 表 |
     | Faiss 检索 | `search` | 双索引检索 + 窗口过滤 + 自适应 K |
     | 结果写库 | `result_persist` | mark_indexed（新增计时点，见 §3.1） |
     | 向量注册 | `faiss_add` | Faiss 双索引 add |

     需求中"结果落库"对应"向量落库 + 结果写库"两行；总耗时（`total_ms`）展示在表格下方。
  3. Top-3 相似帖子卡片：每张卡片含图片、文字、`sim_image` / `sim_text` 分数。

### 4.2 Top-3 帖子内容获取与图片渲染

- 通过 `store.get_post(post_id)` 取回帖子记录；
- 图片渲染规则：
  - 库存有 `image_base64`：解码后显示；
  - 仅有 `image_url`：尝试下载加载，失败则显示占位符（"图片不可用"）；
  - 无图帖：显示占位符。
- 若 Top-3 不足 3 条（如空索引、全部被窗口过滤），按实际条数展示并提示。

### 4.3 错误处理

- 处理失败（`ProcessingError` 各错误码，如 `image_decode_failed` / `image_too_large` / `encode_failed`）：页面展示错误码与已完成阶段的耗时，不抛异常堆栈；
- 未上传图片且文字为空：前端校验拦截；
- 文字超长（`cfg.text_max_chars`）：与 API 一致的校验提示。

## 5. 依赖与运行

- `poetry add --group dev gradio`；
- 启动：`poetry run python demo/app.py`，监听本机 `127.0.0.1:7860`；
- 启动日志打印：索引重建条数、库路径、以及"请确保 FastAPI 服务未同时运行"的提示。

## 6. 测试计划

| 层 | 内容 |
|---|---|
| 单元 | `IndexService.top_hits()`：合并去重、按 max 排序、自排除、窗口过滤、负分归零、空索引 |
| 单元 | `Processor.process()` 成功/失败路径均返回 `ProcessOutcome`，耗时字典含各阶段键 |
| 冒烟 | Demo 处理函数（与 UI 解耦）用 `FakeEmbedder` + 临时库跑通端到端：提交 → 返回分数/耗时/Top-3，不启动 Gradio 服务 |
| 回归 | 现有测试套件全量通过（`process()` 返回值变更不得破坏 worker 与既有断言） |

## 7. 不做的事（YAGNI）

- 不做帖子删除/管理界面；
- 不做批量提交、历史记录浏览页；
- 不做认证/多用户隔离（本机演示用途）；
- 不改动生产 HTTP API 与异步链路行为。

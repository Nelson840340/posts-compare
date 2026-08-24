# Gradio 交互演示（Demo）设计文档

- 日期：2026-08-24
- 状态：待评审（grill-me 拷问修订：8 项决议，见附录 A 及各节标注）
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
| 输入形态 | 图片上传与图片 URL **二选一**（都填报错）；走 URL 时复用生产 `fetch_image` 完整下载链路（附录 A-5） |
| 并发模型 | 处理函数层全局 `threading.Lock`，提交串行执行（附录 A-1） |
| 启动姿态 | 立即开门：页面先起，模型加载与索引重建后台进行；就绪前提交返回"服务准备中"（附录 A-8） |

### 运行约束（重要）

Demo 与 FastAPI 生产服务**不得同时运行**：

1. 两进程各自维护一份内存 Faiss 索引，互相看不到对方的写入，检索结果会发散；
2. 两进程同时写 SQLite 存在写锁竞争。

文档与启动日志中明确提示：运行 Demo 前先停止 FastAPI 服务。Demo 启动时与生产一致地从库重建索引，保证索引与 SQLite（唯一事实源）对齐。

### 已接受风险

- **demo 数据污染（附录 A-7，明确接受）**：demo 帖留在生产库，不提供清理脚本，靠 30 天窗口自然过期。已知后果：窗口期内切回生产服务时，真实帖可能与 demo 帖碰撞而被判重降权。

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
- **失败路径契约（附录 A-2）**：`process()` 增加显式开关 `raise_on_error: bool = True`。生产 Worker 走默认值，行为逐字节不变（`mark_failed` 后仍 raise）；demo 传 `False`，失败时 `mark_failed` 后不 raise，直接返回 outcome（`result=None`、`error_code` 有值、附已完成阶段耗时）。不采用"demo 层捕获异常取耗时"——`timer` 是 `process()` 局部变量，异常抛出后耗时数据即丢失；
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
- 分数处理与 `search()` 一致（负分归零）；
- **不做自适应 K 扩 K（附录 A-3）**：双模态各 6 条候选过滤后不足 3 条时按实际条数展示。明确接受：极端挤占场景（大量过期帖挤占候选位）下 demo 展示可能与生产 `search()` 自适应后的结论不一致——`top_hits()` 仅服务展示，判重检测仍走 `search()`，两条路径语义独立。

`TopHit` 数据类字段：`post_id`、`sim_image`、`sim_text`。

## 4. Demo 应用结构

新增 `demo/app.py`（独立入口，不进 `app` 包），内部按职责拆分：

| 单元 | 职责 |
|---|---|
| 装配与就绪（附录 A-8） | Gradio 页面先启动；后台任务加载 `Config` → Embedder → 打开 `SqliteStore(cfg.db_path)` → 从库重建 `IndexService`（复用 `list_indexed_within` + `decode_vector`，与生产 lifespan 相同逻辑）→ 置位 ready 标志 |
| 处理函数（纯逻辑，可测） | **全局 `threading.Lock` 串行化全程（附录 A-1）**：就绪检查 → 接收图片字节/URL 与文字 → 生成 `post_id`（`demo-{uuid4.hex[:12]}`）→ `upsert_post` → 同步执行 `Processor.process(raise_on_error=False)`（`asyncio.run` 桥接）→ 调用 `top_hits()` → 查询 Top-3 帖子内容 → 返回结构化结果 |
| UI（Gradio Blocks） | 只负责渲染与事件绑定，不含业务逻辑 |

### 4.1 页面布局

- **输入区**：图片上传组件 + **可选图片 URL 输入框（与上传二选一，都填报错提示，附录 A-5）** + 文字输入框 + 提交按钮。
- **结果区**：
  1. 分数卡片：`max_sim` / `sim_image` / `sim_text`，**附阈值标档辅助标注：从 `cfg` 动态读取 `hard_downweight_threshold` / `soft_downweight_threshold` 划档（≥hard 强降权区间 / 两阈值之间软降权区间 / <soft 正常），并注明"阈值仅供参考，降权判定由下游推荐系统执行"（附录 A-6）**；
  2. 耗时表格，按**真实执行顺序**展示 7 行（与需求措辞的映射见表后说明）：

     | 页面标签 | 对应 timer 阶段 | 说明 |
     |---|---|---|
     | 图片下载 | `download` | 上传形态：base64 解码 + 图片校验；URL 形态：生产 `fetch_image` 网络下载（含重试退避，附录 A-5）；页面标签随输入形态动态切换 |
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
  - 仅有 `image_url`：**独立于生产链路的轻量下载：单次请求、3 秒超时、不重试**，失败显示占位符（"图片不可用"）（附录 A-4；不复用 `fetch_image` 的重试退避，避免页面渲染被阻塞到分钟级）；
  - 无图帖：显示占位符。
- 若 Top-3 不足 3 条（如空索引、全部被窗口过滤），按实际条数展示并提示。

### 4.3 错误处理

- 处理失败（`ProcessingError` 各错误码，如 `image_decode_failed` / `image_too_large` / `encode_failed`）：页面展示错误码与已完成阶段的耗时，不抛异常堆栈；
- **上传与 URL 同时提供：前端校验拦截，提示二选一（附录 A-5）**；
- **未就绪（模型加载/索引重建未完成）：提交返回"服务准备中，请稍后"（附录 A-8）**；
- 未上传图片且文字为空：前端校验拦截；
- 文字超长（`cfg.text_max_chars`）：与 API 一致的校验提示。

## 5. 依赖与运行

- `poetry add --group dev gradio`；
- 启动：`poetry run python demo/app.py`，监听本机 `127.0.0.1:7860`；
- **启动行为：页面立即开门，模型加载与索引重建后台进行（附录 A-8）**；启动日志打印：库路径、就绪完成后的索引重建条数、以及"请确保 FastAPI 服务未同时运行"的提示。

## 6. 测试计划

| 层 | 内容 |
|---|---|
| 单元 | `IndexService.top_hits()`：合并去重、按 max 排序、自排除、窗口过滤、负分归零、空索引 |
| 单元 | `Processor.process()` 成功/失败路径均返回 `ProcessOutcome`，耗时字典含各阶段键 |
| 冒烟 | Demo 处理函数（与 UI 解耦）用 `FakeEmbedder` + 临时库跑通端到端：提交 → 返回分数/耗时/Top-3，不启动 Gradio 服务 |
| 回归 | 现有测试套件全量通过（`process()` 返回值变更不得破坏 worker 与既有断言） |

## 7. 不做的事（YAGNI）

- 不做帖子删除/管理界面；
- **不做 demo 数据清理脚本/工具**（数据污染风险已明确接受，见 §2"已接受风险"与附录 A-7）；
- 不做批量提交、历史记录浏览页；
- 不做认证/多用户隔离（本机演示用途）；
- 不改动生产 HTTP API 与异步链路行为。

## 附录 A：grill-me 拷问决议清单（2026-08-24）

| # | 问题 | 决议 |
|---|---|---|
| A-1 | Gradio 并发请求 vs IndexService 非线程安全 | 处理函数层全局 `threading.Lock`，提交串行执行，与生产"单 Worker 串行"不变式对齐 |
| A-2 | `process()` 失败路径的 outcome 契约 | 增加 `raise_on_error` 开关：生产默认 raise 行为不变；demo 传 `False` 拿到带耗时与 `error_code` 的 outcome |
| A-3 | `top_hits()` 候选池被窗口过滤/自排除挤占 | 不做自适应扩 K：双模态各 6 条候选，不足按实际条数展示；接受极端场景与 `search()` 结论可能不一致 |
| A-4 | Top-3 中 URL 帖图片的渲染下载 | 轻量下载：单次请求、3 秒超时、不重试，失败占位符；不复用生产重试退避 |
| A-5 | "图片下载"耗时名不副实 | 输入支持可选 URL（与上传二选一，都填报错），URL 走生产 `fetch_image` 完整下载链路 |
| A-6 | 裸分数对观众无语义 | 分数旁附阈值标档辅助标注，阈值从 `cfg` 动态读取，注明判定权在下游 |
| A-7 | demo 帖污染生产库 | 接受风险：不做清理，30 天窗口自然过期；知悉切回生产可能与 demo 帖碰撞降权 |
| A-8 | 启动时机 | 立即开门：后台加载，未就绪提交返回"服务准备中" |

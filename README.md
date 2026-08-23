# 帖子相似度检测系统（Demo）

新帖入库后异步检测重复发帖（同图多发、裁剪重发、文字相似），返回其在最近 30 天窗口内所有帖子中的最大相似度。相似度 = max(图片相似度, 文字相似度)。

设计与决策记录见 [spec](docs/superpowers/specs/2026-08-23-post-similarity-design.md)。

**架构要点**：FastAPI 单进程 + asyncio 有界队列 + 单 Worker 串行 + Faiss 双索引（CLIP 图 512d / multilingual-e5-small 文 384d）+ SQLite(WAL) 唯一事实源；崩溃重启后索引从库重建、pending 任务自动回放，零数据丢失。

## 环境准备

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
```

模型首次下载需可达 Hugging Face（本环境使用镜像）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
pytest -m slow   # 真模型冒烟测试，首次运行自动下载 CLIP ViT-B/32 与 multilingual-e5-small
```

模型下载后缓存在 `~/.cache/huggingface/hub`，后续运行无需联网。

## 启动

```bash
python -m app.main        # 默认 127.0.0.1:8000
```

所有配置项及默认值见 [app/config.py](app/config.py)，支持 `SIM_` 前缀环境变量覆盖，常用：

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `SIM_DB_PATH` | `data/similarity.db` | SQLite 库路径 |
| `SIM_PORT` / `SIM_HOST` | `8000` / `127.0.0.1` | uvicorn 监听（main.py 读取） |
| `SIM_DEVICE` | `cpu` | 推理设备（ROCm 可填对应 device，可选开关） |
| `SIM_WINDOW_DAYS` | `30` | 相似度检索窗口 |
| `SIM_TOP_K` | `16` | 每索引检索候选数（自适应扩 K 至 64/256） |
| `SIM_HARD_DOWNGRADE_THRESHOLD` / `SIM_SOFT_DOWNGRADE_THRESHOLD` | `0.90` / `0.75` | 分数带阈值，供下游参考 |

## 联调示例

```bash
# 提交帖（base64 图片；text 与 image 至少提供正文，无图帖合法）
curl -X POST http://127.0.0.1:8000/posts -H 'Content-Type: application/json' -d '{
  "post_id": "p-001",
  "text": "今天去公园散步天气特别好",
  "image_base64": "<png/jpeg base64，≤20MB>"
}'
# → 202 {"post_id":"p-001","status":"pending"}；重复提交同 post_id → 200（幂等）

# 提交帖（url 图片，由服务端下载）
curl -X POST http://127.0.0.1:8000/posts -H 'Content-Type: application/json' -d '{
  "post_id": "p-002",
  "text": "这家店的咖啡味道真的很不错",
  "image_url": "https://example.com/photo.jpg"
}'

# 轮询结果（pending→202；indexed→分数；failed→原因）
curl http://127.0.0.1:8000/posts/p-001/similarity
# → {"status":"indexed","max_sim":0.42,"sim_image":0.42,"sim_text":0.31,
#    "matched_post_id":"p-000","computed_at":"...","note":null}

# 健康检查（模型/索引计数/队列深度；未就绪时 503）
curl http://127.0.0.1:8000/health

# 重放失败帖（failed → pending 重新入队）
curl -X POST http://127.0.0.1:8000/admin/replay -H 'Content-Type: application/json' \
  -d '{"post_id": "p-001"}'

# 压缩索引（清除窗口外条目，经 Worker 串行执行）
curl -X POST http://127.0.0.1:8000/admin/replay -H 'Content-Type: application/json' \
  -d '{"mode": "compact"}'
# → {"mode":"compact","status":"scheduled"}（实际清除数见 Worker 日志）
```

## 测试与校准

```bash
pytest -q                          # 单元/API/集成（默认排除 slow）
pytest -m slow                     # 真模型冒烟（需 HF_ENDPOINT，首次下载模型）
HF_ENDPOINT=https://hf-mirror.com python eval/synthetic_calibration.py   # 合成校准
```

校准脚本生成正负例对（裁剪/缩放/调色变体 vs 不同图；改写句 vs 无关句），输出 `eval/calibration_report.json` 与阈值建议。合成校准仅为起点，上生产后必须用真实流量重校（spec §5.3）。

当前校准结论（2026-08-23 实测）：文字信号分离度良好（正例 min 0.937 vs 负例 max 0.851，建议分割点 0.894）；图片紧裁剪 `crop_center_50` min 0.648 低于 0.8 基线、合成负例偏高，按 spec §11 决议1 处置为阈值校准问题，暂不换模型。

## 已知边界（摘自 spec §9 与实现修订）

- Demo 每帖单图；**无图帖合法**（图片通道记零分，仅比文字），不做多图比较。
- failed 帖不进索引造成检测盲区（已确认接受）："原帖 failed + 仅一次重发"完全逃逸，依赖人工 `/admin/replay`。
- 短文本余弦虚高可能造成文字信号误伤（已确认接受），不做长度门槛，靠阈值带隔离。
- 相似度仅覆盖已入库且在 30 天窗口内的帖子；系统启动前/窗口外的帖子不参与比较。
- 本系统只输出分数，"是否降权、降多少"由下游推荐系统决策。
- 生产部署必须反代层限制请求体 ≤30MB（image_base64 通道，spec §6.4/§11 决议9）。

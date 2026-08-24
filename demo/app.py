"""Gradio demo 入口（spec 2026-08-24 §4/§5）。

启动：poetry run python demo/app.py（127.0.0.1:7860）
约束：不得与 FastAPI 生产服务同时运行（双份内存索引互不可见、SQLite 写锁竞争）。
页面立即开门，模型加载与索引重建后台进行（A-8）；未就绪提交返回"服务准备中"。
"""
import logging
import os
import threading

# macOS 上 faiss 与 torch 各自携带 libomp，双份加载会在首次检索时 SIGABRT；
# 必须在任何扩展库加载前设置（Linux 无此冲突，setdefault 不覆盖用户显式配置）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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

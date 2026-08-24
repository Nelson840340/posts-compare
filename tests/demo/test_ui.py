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

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

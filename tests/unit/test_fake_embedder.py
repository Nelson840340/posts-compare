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

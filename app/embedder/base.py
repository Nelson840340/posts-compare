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

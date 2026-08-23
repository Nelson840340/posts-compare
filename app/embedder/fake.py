"""FakeEmbedder：确定性伪向量，测试主力（spec §8.1）。

图片：sha256 播种主方向 + 首 1KB 字节直方图特征（固定 basis），
字节微小差异→向量接近，跨输入一致。
文字：字符 2-gram 哈希累加，共享 n-gram 越多越接近。
prefix 并入哈希，用于验证前缀传导。
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
        salted = prefix + text
        for i in range(len(salted) - 1):
            gram = salted[i:i + 2]
            h = int.from_bytes(hashlib.sha256(gram.encode()).digest()[:4], "big")
            v[h % TEXT_DIM] += 1.0
        if v.sum() == 0:  # 空文本兜底：固定方向
            v[0] = 1.0
        return normalize(v)

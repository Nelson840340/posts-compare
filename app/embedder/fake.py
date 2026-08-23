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

"""ClipE5Embedder：真模型实现（spec §5.1）。

关键点：
1. CLIP 预处理关闭 center crop——全图 resize 到 224×224（允许轻微形变），
   对近重复匹配有实打实提升（spec §11 决议1）。禁用 open_clip 默认 transform。
2. e5 必须带前缀：调用方传 "passage: " / "query: "，本类不做兜底拼接，
   前缀缺失属于调用方缺陷（spec §5.1）。
3. encode 为同步 CPU 密集操作，由调用方放入线程池执行。
"""
import io

import numpy as np
import torch
from PIL import Image

from app.embedder.base import IMAGE_DIM, TEXT_DIM, normalize


class ClipE5Embedder:
    image_dim = IMAGE_DIM
    text_dim = TEXT_DIM

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._clip = None
        self._clip_preprocess = None
        self._e5 = None

    def load(self) -> None:
        import open_clip
        from sentence_transformers import SentenceTransformer

        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k")
        self._clip = model.eval().to(self.device)

        from torchvision import transforms
        # 全图 resize，不 center crop（spec §11 决议1）
        self._clip_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711)),
        ])

        self._e5 = SentenceTransformer("intfloat/multilingual-e5-small",
                                       device=self.device)

    def embed_image(self, image_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self._clip_preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self._clip.encode_image(tensor)
        return normalize(feat[0].float().cpu().numpy())

    def embed_text(self, text: str, *, prefix: str) -> np.ndarray:
        vec = self._e5.encode([prefix + text], normalize_embeddings=False)[0]
        return normalize(np.asarray(vec, dtype=np.float32))

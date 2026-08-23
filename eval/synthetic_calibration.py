"""合成校准集（spec §5.3 两阶段校准第一阶段 + §8.4）。

Demo 无生产数据，校准集由合成变体构成：
- 图片正例：同一合成图的裁剪(50%/25-75%)/缩放(0.5x,1.5x)/轻度调色变体
- 图片负例：不同随机合成图
- 文字正例：句尾加语气词/换标点/同近义改写模板
- 文字负例：无关随机句
输出：eval/calibration_report.json + 终端分布摘要与阈值建议。
阈值仅为合成意义下的起点；上生产后必须用真实流量重校（spec §5.3）。
"""
import io
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance

from app.embedder.clip_e5 import ClipE5Embedder

OUT = Path(__file__).parent / "calibration_report.json"


def make_image(seed: int) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("RGB", (640, 480), (rng.randrange(256),) * 3)
    d = ImageDraw.Draw(img)
    for _ in range(rng.randrange(5, 15)):
        x0, y0 = rng.randrange(600), rng.randrange(440)
        d.ellipse([x0, y0, x0 + rng.randrange(20, 120), y0 + rng.randrange(20, 120)],
                  fill=(rng.randrange(256), rng.randrange(256), rng.randrange(256)))
    return img


def to_bytes(img: Image.Image, fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def image_variants(img: Image.Image) -> dict[str, bytes]:
    w, h = img.size
    return {
        "crop_center_50": to_bytes(img.crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4))),
        "crop_offset_25_75": to_bytes(img.crop((0, 0, int(w * .75), int(h * .75)))),
        "resize_0.5x": to_bytes(img.resize((w // 2, h // 2))),
        "resize_1.5x": to_bytes(img.resize((int(w * 1.5), int(h * 1.5)))),
        "color_jitter": to_bytes(ImageEnhance.Color(img).enhance(1.3)),
    }


TEXT_BASES = [
    "今天去公园散步天气特别好",
    "这家店的咖啡味道真的很不错",
    "新买的耳机音质超出预期",
    "周末和朋友爬山看到了日落",
    "家里的小猫又把杯子打翻了",
]
TEXT_POS = {t: [t + "！", t.replace("。", "") + "，推荐", t[:4] + "，" + t[4:]] for t in TEXT_BASES}
TEXT_NEG = [
    "央行今日开展逆回购操作",
    "量子计算的最新进展引发讨论",
    "本市地铁新线路下月开通",
    "股市今日收盘小幅上涨",
    "秋季招聘会将在校体育馆举行",
]


def main():
    emb = ClipE5Embedder(device="cpu")
    emb.load()

    img_scores: dict[str, list[float]] = {}
    for seed in range(10):  # 10 个基础图
        base_img = make_image(seed)
        v_base = emb.embed_image(to_bytes(base_img))
        for name, variant in image_variants(base_img).items():
            img_scores.setdefault(name, []).append(float(v_base @ emb.embed_image(variant)))
        for other in range(seed + 1, seed + 3):  # 负例：相邻 seed 的图
            img_scores.setdefault("negative", []).append(
                float(v_base @ emb.embed_image(to_bytes(make_image(other)))))

    txt_pos, txt_neg = [], []
    for base in TEXT_BASES:
        vb = emb.embed_text(base, prefix="passage: ")
        for pos in TEXT_POS[base]:
            txt_pos.append(float(vb @ emb.embed_text(pos, prefix="query: ")))
        for neg in TEXT_NEG:
            txt_neg.append(float(vb @ emb.embed_text(neg, prefix="query: ")))

    def stat(xs):
        xs = sorted(xs)
        return {"n": len(xs), "min": round(xs[0], 4), "median": round(xs[len(xs) // 2], 4),
                "max": round(xs[-1], 4)}

    report = {
        "image": {k: stat(v) for k, v in img_scores.items()},
        "text_positive": stat(txt_pos),
        "text_negative": stat(txt_neg),
        "note": ("合成校准仅为起点；上生产后必须真实流量重校（spec §5.3）。"
                 "2026-08-23 实测：crop_center_50 min=0.6479 < 0.8 基线（spec §11 决议1），"
                 "且合成负例偏高（max=0.9565，随机椭圆图结构相似致 CLIP 分数重叠）；"
                 "按 spec 处置为阈值校准问题，暂不换模型"),
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # 阈值建议：正例最小值与负例最大值的中点（各信号独立），持久化进 report
    pos_min = min(min(v) for k, v in img_scores.items() if k != "negative")
    neg_max = max(img_scores["negative"])
    report["threshold_suggestion"] = {
        "image_split": round((pos_min + neg_max) / 2, 4),
        "text_split": round((min(txt_pos) + max(txt_neg)) / 2, 4),
        "caveat": "图片正负分布重叠，分割点仅供参考",
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[图片] 正例min={pos_min:.3f} 负例max={neg_max:.3f} "
          f"建议分割点={(pos_min + neg_max) / 2:.3f}")
    print(f"[文字] 正例min={min(txt_pos):.3f} 负例max={max(txt_neg):.3f} "
          f"建议分割点={(min(txt_pos) + max(txt_neg)) / 2:.3f}")


if __name__ == "__main__":
    main()

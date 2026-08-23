"""图片获取：base64 解码 / httpx 下载（退避重试）+ 格式大小校验（spec §4.3/§6.1）。"""
import asyncio
import base64
import io

import httpx
from PIL import Image, UnidentifiedImageError

from app.config import Config
from app.domain import ImageDecodeError, ImageDownloadError, PostRecord

_ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


async def fetch_image(rec: PostRecord, cfg: Config, sleep=asyncio.sleep) -> bytes:
    if rec.image_base64:
        try:
            return base64.b64decode(rec.image_base64, validate=True)
        except Exception as e:
            raise ImageDownloadError(f"base64 解码失败: {e}") from e
    if not rec.image_url:
        raise ImageDownloadError("image_url 与 image_base64 均缺失")

    last_err: Exception | None = None
    for attempt in range(cfg.download_retries):
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                resp = await client.get(rec.image_url, timeout=cfg.download_timeout_seconds)
            if resp.status_code != 200:
                raise ImageDownloadError(f"HTTP {resp.status_code}")
            return resp.content
        except ImageDownloadError as e:
            last_err = e
            if resp.status_code in (404, 403):  # 确定性失败不重试
                break
        except Exception as e:
            last_err = e
        if attempt < cfg.download_retries - 1:
            await sleep(cfg.download_retry_backoff_seconds[attempt])
    raise ImageDownloadError(f"下载失败（{cfg.download_retries} 次尝试）: {last_err}")


def decode_and_validate(image_bytes: bytes, cfg: Config) -> bytes:
    if len(image_bytes) > cfg.image_max_bytes:
        raise ImageDecodeError(f"图片超限: {len(image_bytes)} > {cfg.image_max_bytes}")
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.verify()
            fmt = img.format
    except (UnidentifiedImageError, OSError) as e:
        raise ImageDecodeError(f"图片解码失败: {e}") from e
    if fmt not in _ALLOWED_FORMATS:
        raise ImageDecodeError(f"不支持的图片格式: {fmt}")
    return image_bytes

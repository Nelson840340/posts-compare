import base64
import io

import httpx
import pytest
from PIL import Image

from app.config import Config
from app.domain import ImageDecodeError, ImageDownloadError, PostRecord
from app.pipeline.downloader import decode_and_validate, fetch_image


def _cfg(**kw):
    return Config(db_path="/tmp/x.db", **kw)


def _png_bytes(color=(1, 2, 3), fmt="PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), color).save(buf, format=fmt)
    return buf.getvalue()


async def _no_sleep(_):
    pass


async def test_base64_channel_decodes():
    raw = _png_bytes()
    rec = PostRecord(post_id="p", text="t",
                     image_base64=base64.b64encode(raw).decode())
    got = await fetch_image(rec, _cfg(), sleep=_no_sleep)
    assert got == raw


async def test_url_channel_success(monkeypatch):
    raw = _png_bytes()

    async def fake_get(self, url, timeout):
        return httpx.Response(200, content=raw, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    rec = PostRecord(post_id="p", text="t", image_url="http://x/a.png")
    assert await fetch_image(rec, _cfg(), sleep=_no_sleep) == raw


async def test_url_channel_retries_then_fails(monkeypatch):
    calls = {"n": 0}

    async def fake_get(self, url, timeout):
        calls["n"] += 1
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    rec = PostRecord(post_id="p", text="t", image_url="http://x/a.png")
    with pytest.raises(ImageDownloadError):
        await fetch_image(rec, _cfg(download_retries=3), sleep=_no_sleep)
    assert calls["n"] == 3  # 共尝试 download_retries 次


async def test_http_404_is_download_error(monkeypatch):
    async def fake_get(self, url, timeout):
        return httpx.Response(404, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    rec = PostRecord(post_id="p", text="t", image_url="http://x/a.png")
    with pytest.raises(ImageDownloadError):
        await fetch_image(rec, _cfg(download_retries=1), sleep=_no_sleep)


async def test_neither_source_raises():
    rec = PostRecord(post_id="p", text="t")
    with pytest.raises(ImageDownloadError):
        await fetch_image(rec, _cfg(), sleep=_no_sleep)


def test_decode_validates_format_and_size():
    ok = _png_bytes(fmt="JPEG")
    assert decode_and_validate(ok, _cfg()) == ok
    with pytest.raises(ImageDecodeError):
        decode_and_validate(b"not an image", _cfg())
    with pytest.raises(ImageDecodeError):  # 超限
        decode_and_validate(_png_bytes(), _cfg(image_max_bytes=10))

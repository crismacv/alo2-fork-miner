"""Qwen-Image-Edit-2511 (Apache 2.0, open-weight) via chutes.ai.

Cleans the reference image before feeding it to the coder VLM:
  - background → neutral 50% gray
  - subject preserved (no color/shape changes)

The submission .js never sees these images — preprocess is invisible to validators.
On any failure (no API key, HTTP error, malformed response), returns the
original image unchanged so the pipeline always makes progress.
"""
from __future__ import annotations
import base64
import io
import logging
import os

import httpx
from PIL import Image

log = logging.getLogger("qwen.edit")

CHUTES_KEY = os.environ.get("CHUTES_API_KEY", "")
ENDPOINT = os.environ.get(
    "QWEN_EDIT_URL",
    "https://chutes-qwen-image-edit-2511.chutes.ai/generate",
)
ENABLED = os.environ.get("QWEN_EDIT_ENABLED", "1") not in ("0", "false", "")

DEFAULT_PROMPT = (
    "Remove the background and replace it with a plain neutral 50% gray (#808080). "
    "Keep the main object centered, fully visible, and crisp. "
    "Do not alter the object's colors, materials, or shape."
)


def _to_jpeg_b64(image_bytes: bytes) -> str:
    """Re-encode to JPEG (smaller payload) and base64."""
    im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if max(im.size) > 1024:
        im.thumbnail((1024, 1024), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def _to_png(img_bytes: bytes) -> bytes:
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    out = io.BytesIO()
    im.save(out, format="PNG", optimize=True)
    return out.getvalue()


async def edit(image_bytes: bytes, prompt: str = DEFAULT_PROMPT, *,
               seed: int = 42, steps: int = 28, cfg: float = 4.0,
               timeout: float = 90.0) -> bytes:
    """Run Qwen-Image-Edit; return cleaned PNG. Returns original on any failure."""
    if not ENABLED or not CHUTES_KEY:
        return image_bytes
    try:
        b64 = _to_jpeg_b64(image_bytes)
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                ENDPOINT,
                headers={"Authorization": f"Bearer {CHUTES_KEY}",
                         "Content-Type": "application/json"},
                json={
                    "seed": seed, "width": 1024, "height": 1024,
                    "prompt": prompt, "image_b64s": [b64],
                    "true_cfg_scale": cfg, "negative_prompt": "",
                    "num_inference_steps": steps,
                },
            )
            if r.status_code != 200:
                log.warning(f"chutes HTTP {r.status_code}: {r.text[:160]}")
                return image_bytes
            ct = r.headers.get("content-type", "")
            if not ct.startswith("image"):
                log.warning(f"chutes non-image response: {ct}")
                return image_bytes
            return _to_png(r.content)
    except Exception as e:
        log.warning(f"qwen_edit: {type(e).__name__}: {e}")
        return image_bytes

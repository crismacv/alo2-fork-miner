"""Multi-view reference augmentation via Chutes Qwen-Image-Edit-2511.

USED ONLY BY THE DEBUG/COMPARE PIPELINE (debug_one.py). NOT imported by
the production scene_coder agent — the deployed Docker image must not
talk to Chutes.

For a single reference image we ask Qwen-Image-Edit to produce three
rotated views (45°, 90°, 180°), then we lay out original + rotations in
a 2x2 grid PNG. That grid is passed to the coder so it can see the
object from multiple angles and pick up details that are invisible in
the single front view.
"""
from __future__ import annotations
import asyncio
import base64
import io
import logging
import os
from pathlib import Path

import httpx
from PIL import Image

log = logging.getLogger("multi_view")

# .env had Korean characters accidentally appended to the API key. Strip
# anything non-ASCII / not-allowed-in-token so the Authorization header
# is encodable. Token format is `cpk_<hex32>.<hex32>.<base62 32>`.
import re as _re
_raw_key = os.environ.get("CHUTES_API_KEY", "")
_m = _re.match(r"cpk_[A-Za-z0-9]+\.[A-Za-z0-9]+\.[A-Za-z0-9]+", _raw_key)
CHUTES_KEY = _m.group(0) if _m else "".join(c for c in _raw_key if ord(c) < 128).strip()
ENDPOINT = os.environ.get(
    "QWEN_EDIT_URL",
    "https://chutes-qwen-image-edit-2511.chutes.ai/generate",
)

ROTATION_PROMPTS = [
    "Rotate the object 45 degrees to the right around its vertical axis. "
    "Show its three-quarter angle. Keep the same object, same colors, same "
    "scale, plain neutral light gray background.",
    "Rotate the object 90 degrees to show its side profile. Same object, "
    "same materials, plain neutral light gray background.",
    "Rotate the object 180 degrees to show its back. Same object, same "
    "scale and colors, plain neutral light gray background.",
]


def _to_jpeg_b64(image: Image.Image, max_side: int = 768) -> str:
    im = image.convert("RGB").copy()
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


async def _edit(client: httpx.AsyncClient, b64: str, prompt: str, seed: int,
                steps: int = 28, cfg: float = 4.0) -> bytes | None:
    if not CHUTES_KEY:
        return None
    try:
        r = await client.post(
            ENDPOINT,
            headers={"Authorization": f"Bearer {CHUTES_KEY}",
                     "Content-Type": "application/json"},
            json={
                "seed": seed, "width": 768, "height": 768,
                "prompt": prompt, "image_b64s": [b64],
                "true_cfg_scale": cfg, "negative_prompt": "",
                "num_inference_steps": steps,
            },
        )
        if r.status_code != 200:
            log.warning(f"rotation HTTP {r.status_code}: {r.text[:200]}")
            return None
        ct = r.headers.get("content-type", "")
        if not ct.startswith("image"):
            log.warning(f"rotation non-image content-type {ct}, body[:120]: {r.text[:120]}")
            return None
        return r.content
    except Exception as e:
        log.warning(f"rotation exception {type(e).__name__}: {e}")
        return None


async def multi_view_grid(image_bytes: bytes, *, seed: int = 42,
                           timeout: float = 180.0) -> bytes:
    """Return a 2x2 PNG of [original, 45°, 90°, 180°]. On any failure
    falls back to the original image (untouched). Suitable as the
    reference image passed to the coder."""
    if not CHUTES_KEY:
        log.info("CHUTES_API_KEY not set — multi_view skipped")
        return image_bytes
    try:
        front = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        b64 = _to_jpeg_b64(front)
        async with httpx.AsyncClient(timeout=timeout) as client:
            rotations = await asyncio.gather(*[
                _edit(client, b64, p, seed=seed + i * 7)
                for i, p in enumerate(ROTATION_PROMPTS)
            ])
        tiles: list[Image.Image] = [front]
        ok_count = 0
        for body in rotations:
            if body is None:
                tiles.append(front)
            else:
                try:
                    tiles.append(Image.open(io.BytesIO(body)).convert("RGB"))
                    ok_count += 1
                except Exception:
                    tiles.append(front)
        side = 512
        grid = Image.new("RGB", (side * 2, side * 2), (128, 128, 128))
        for i, t in enumerate(tiles[:4]):
            t2 = t.copy()
            t2.thumbnail((side, side), Image.LANCZOS)
            ox = (i % 2) * side + (side - t2.size[0]) // 2
            oy = (i // 2) * side + (side - t2.size[1]) // 2
            grid.paste(t2, (ox, oy))
        out = io.BytesIO()
        grid.save(out, format="PNG", optimize=True)
        log.info(f"multi_view: {ok_count}/3 rotations OK")
        return out.getvalue()
    except Exception as e:
        log.warning(f"multi_view aborted: {type(e).__name__}: {e}")
        return image_bytes

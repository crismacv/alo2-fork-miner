"""Multi-view augmentation via Qwen-Image-Edit-2511.

For each reference, generate 1-3 plausible rotations and stitch them with the
original into a single 2x2 grid image. The coder VLM gets richer 3D context
than a single front view, while the submission .js remains plain procedural
Three.js (no image leakage).

Best effort: if any rotation fails the slot falls back to the original. If the
whole stage fails we just return the (already preprocessed) input.
"""
from __future__ import annotations
import asyncio
import base64
import io
import logging
import os

import httpx
from PIL import Image

log = logging.getLogger("multi_view")

CHUTES_KEY = os.environ.get("CHUTES_API_KEY", "")
ENDPOINT = os.environ.get(
    "QWEN_EDIT_URL",
    "https://chutes-qwen-image-edit-2511.chutes.ai/generate",
)
ENABLED = os.environ.get("MULTI_VIEW_ENABLED", "0") not in ("0", "false", "")

ROTATION_PROMPTS = [
    "Rotate the object 45 degrees to the right around its vertical axis. "
    "Show its right side at three-quarter angle. Keep a plain neutral gray background.",
    "Rotate the object 90 degrees, showing only its side profile. "
    "Keep a plain neutral gray background.",
    "Rotate the object 180 degrees. Show the back of the object. "
    "Keep a plain neutral gray background.",
]


def _to_jpeg_b64(image: Image.Image, max_side: int = 768) -> str:
    im = image.convert("RGB").copy()
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


async def _edit(client: httpx.AsyncClient, b64: str, prompt: str, seed: int,
                steps: int = 22, cfg: float = 4.0) -> bytes | None:
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
            log.warning(f"multi_view rotation HTTP {r.status_code}")
            return None
        ct = r.headers.get("content-type", "")
        if not ct.startswith("image"):
            return None
        return r.content
    except Exception as e:
        log.warning(f"multi_view rotation: {type(e).__name__}: {e}")
        return None


async def augment(image_bytes: bytes, *, seed: int = 42, timeout: float = 120.0) -> bytes:
    """Return a 2x2 grid PNG of [front, 45°, 90°, back]. Falls back to input on failure."""
    if not ENABLED or not CHUTES_KEY:
        return image_bytes
    try:
        front = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        b64 = _to_jpeg_b64(front)
        async with httpx.AsyncClient(timeout=timeout) as client:
            rotations = await asyncio.gather(*[
                _edit(client, b64, p, seed=seed + i * 7)
                for i, p in enumerate(ROTATION_PROMPTS)
            ])
        # Open each successful rotation; fall back to front when missing
        tiles: list[Image.Image] = [front]
        for body in rotations:
            if body is None:
                tiles.append(front)
            else:
                try:
                    tiles.append(Image.open(io.BytesIO(body)).convert("RGB"))
                except Exception:
                    tiles.append(front)
        # Resize each tile to a uniform square and place in a 2x2 grid.
        side = 512
        grid = Image.new("RGB", (side * 2, side * 2), (128, 128, 128))
        for i, t in enumerate(tiles[:4]):
            t2 = t.copy()
            t2.thumbnail((side, side), Image.LANCZOS)
            # paste centered into the slot
            ox = (i % 2) * side + (side - t2.size[0]) // 2
            oy = (i // 2) * side + (side - t2.size[1]) // 2
            grid.paste(t2, (ox, oy))
        out = io.BytesIO()
        grid.save(out, format="PNG", optimize=True)
        return out.getvalue()
    except Exception as e:
        log.warning(f"multi_view augment: {type(e).__name__}: {e}")
        return image_bytes

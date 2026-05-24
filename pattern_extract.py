"""Pattern / surface-decoration extraction for the debug pipeline.

USED ONLY BY debug_one.py — production scene_coder is untouched (no Chutes,
no Pillow extras beyond what's already imported).

Why: many lost stems have a body shape we model OK but lose the surface
PATTERN (paisley sofa, iridescent balloon, Coca-Cola script, polka dots).
The coder has trouble inventing those patterns from prose. Giving it a
second image — a zoomed crop of the surface — plus a short text descriptor
of the dominant colors and arrangement, lets it reproduce the pattern as
colored sub-meshes far more reliably.

Flow per stem:
  1. has_complex_pattern() asks the GLM judge: does this object have a
     surface pattern that would be lost without explicit handling?
     (Single image, ~5s, ~$0.01 per call.)
  2. If YES:
       extract_pattern_image() asks Chutes Qwen-Image-Edit to isolate
       the pattern as a flat tileable square. PIL center-crop fallback
       if Chutes fails or returns junk.
       analyze_colors() runs PIL k-means on the resulting image and
       returns up to 5 dominant colors and a guessed arrangement
       ("stripes_horizontal" / "polka_dots" / "swirl_paisley" /
       "gradient" / "solid" / "irregular").
       compose_grid() builds a 2x1 PNG (top half = original ref,
       bottom half = extracted pattern at high zoom) plus a small caption
       strip naming the colors.
  3. If NO: returns None and the caller uses the original ref.
"""
from __future__ import annotations
import asyncio
import base64
import io
import json
import logging
import os
import re
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("pattern_extract")

# --- Chutes Qwen-Image-Edit token (same fix-up as multi_view.py) ---
_raw_key = os.environ.get("CHUTES_API_KEY", "")
_m = re.match(r"cpk_[A-Za-z0-9]+\.[A-Za-z0-9]+\.[A-Za-z0-9]+", _raw_key)
CHUTES_KEY = _m.group(0) if _m else "".join(c for c in _raw_key if ord(c) < 128).strip()
CHUTES_ENDPOINT = os.environ.get(
    "QWEN_EDIT_URL",
    "https://chutes-qwen-image-edit-2511.chutes.ai/generate",
)


# =====================================================================
# 1. Gate: does the reference have a surface pattern worth extracting?
# =====================================================================

_GATE_PROMPT = """Look at the reference object. Does its SURFACE carry a
visible printed/painted/woven/dyed PATTERN or DECORATION (beyond uniform
color)? Examples of YES: paisley cushions, polka-dot fabric, plaid, rainbow
stripes, holographic iridescent, logo/label, floral painting on ceramic,
camo, gradient, tiled texture. Examples of NO: plain solid color, a small
material-finish detail like brushed metal, a clean polished surface.

Output JSON only: {"has_pattern": true|false, "pattern_kind":
"<stripes_horizontal | stripes_vertical | polka_dots | plaid | paisley |
floral | logo_text | iridescent | gradient | wood_grain | tiled |
camouflage | mixed | none>", "reason": "<1 short sentence>"}"""


def _b64url(b: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(b).decode()}"


async def has_complex_pattern(client, model: str, ref_bytes: bytes) -> dict:
    """Returns {has_pattern, pattern_kind, reason}. Defaults to no on parse
    failure (safer to skip than to apply junk pattern info)."""
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "You output structured JSON only — no prose, no markdown."},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": _b64url(ref_bytes)}},
                    {"type": "text", "text": _GATE_PROMPT},
                ]},
            ],
            max_tokens=512, temperature=0.0, seed=42,
        )
        text = resp.choices[0].message.content or ""
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
        m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if not m:
            return {"has_pattern": False, "pattern_kind": "none", "reason": "parse_fail"}
        data = json.loads(m.group(0))
        data.setdefault("has_pattern", False)
        data.setdefault("pattern_kind", "none")
        data.setdefault("reason", "")
        return data
    except Exception as e:
        log.warning(f"gate exception {type(e).__name__}: {e}")
        return {"has_pattern": False, "pattern_kind": "none", "reason": f"err:{e}"}


# =====================================================================
# 2a. Chutes Qwen-Image-Edit: isolate the pattern as a flat tileable
#     square. We ask once; on failure we fall back to PIL center crop.
# =====================================================================

_PATTERN_PROMPT = (
    "Extract just the surface PATTERN from this object. Show ONLY the "
    "pattern as a flat tileable square — no 3D form, no shadows, no object "
    "outline, no background, no perspective. Keep the exact colors and motifs."
)


def _to_jpeg_b64(image: Image.Image, max_side: int = 768) -> str:
    im = image.convert("RGB").copy()
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


async def _chutes_extract(ref_bytes: bytes, *, seed: int = 42,
                          timeout: float = 180.0) -> bytes | None:
    if not CHUTES_KEY:
        return None
    try:
        im = Image.open(io.BytesIO(ref_bytes)).convert("RGB")
        b64 = _to_jpeg_b64(im)
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                CHUTES_ENDPOINT,
                headers={"Authorization": f"Bearer {CHUTES_KEY}",
                         "Content-Type": "application/json"},
                json={
                    "seed": seed, "width": 768, "height": 768,
                    "prompt": _PATTERN_PROMPT, "image_b64s": [b64],
                    "true_cfg_scale": 4.0, "negative_prompt": "",
                    "num_inference_steps": 28,
                },
            )
        if r.status_code != 200:
            log.warning(f"chutes pattern extract HTTP {r.status_code}: {r.text[:160]}")
            return None
        if not r.headers.get("content-type", "").startswith("image"):
            return None
        return r.content
    except Exception as e:
        log.warning(f"chutes pattern extract exception {type(e).__name__}: {e}")
        return None


def _pil_center_crop(ref_bytes: bytes, side: int = 512) -> bytes:
    """Fallback: center-crop a square of the reference. Includes some
    object curvature / lighting but better than nothing."""
    im = Image.open(io.BytesIO(ref_bytes)).convert("RGB")
    w, h = im.size
    cx, cy = w // 2, h // 2
    half = min(w, h) // 3  # take inner third — usually all pattern, no edge
    box = (cx - half, cy - half, cx + half, cy + half)
    crop = im.crop(box).resize((side, side), Image.LANCZOS)
    out = io.BytesIO()
    crop.save(out, format="PNG", optimize=True)
    return out.getvalue()


# =====================================================================
# 2b. Color analysis on the extracted pattern image.
# =====================================================================

def analyze_colors(pattern_bytes: bytes, *, k: int = 5) -> list[dict]:
    """Returns [{hex, fraction}] sorted by fraction descending."""
    im = Image.open(io.BytesIO(pattern_bytes)).convert("RGB")
    im.thumbnail((128, 128), Image.LANCZOS)
    # PIL quantize → palette mode, then read palette + counts.
    pal_im = im.quantize(colors=k, method=Image.Quantize.MEDIANCUT, kmeans=1)
    pal = pal_im.getpalette()[:k * 3]
    counts = pal_im.getcolors(maxcolors=k * 4) or []
    total = sum(c for c, _ in counts) or 1
    out = []
    for count, idx in sorted(counts, reverse=True):
        if idx * 3 + 2 >= len(pal):
            continue
        r, g, b = pal[idx * 3], pal[idx * 3 + 1], pal[idx * 3 + 2]
        out.append({"hex": f"#{r:02x}{g:02x}{b:02x}", "fraction": count / total})
    return out


# =====================================================================
# 3. Compose the 2-panel image we feed to the coder.
# =====================================================================

def compose_grid(ref_bytes: bytes, pattern_bytes: bytes, *,
                  colors: list[dict] | None = None,
                  pattern_kind: str = "") -> bytes:
    """2x1 layout (top: ref, bottom: pattern crop) + caption strip with
    the dominant-color hex codes. Output is a single PNG."""
    ref = Image.open(io.BytesIO(ref_bytes)).convert("RGB")
    pat = Image.open(io.BytesIO(pattern_bytes)).convert("RGB")
    W = 800
    ref_h = 600
    pat_h = 600
    cap_h = 80
    out = Image.new("RGB", (W, ref_h + pat_h + cap_h), (200, 200, 200))
    # Top: ref
    ref2 = ref.copy(); ref2.thumbnail((W, ref_h), Image.LANCZOS)
    out.paste(ref2, ((W - ref2.size[0]) // 2, (ref_h - ref2.size[1]) // 2))
    # Bottom: pattern
    pat2 = pat.copy(); pat2.thumbnail((W, pat_h), Image.LANCZOS)
    out.paste(pat2, ((W - pat2.size[0]) // 2, ref_h + (pat_h - pat2.size[1]) // 2))
    # Caption strip with color swatches
    drw = ImageDraw.Draw(out)
    drw.rectangle([0, ref_h + pat_h, W, ref_h + pat_h + cap_h], fill=(35, 35, 40))
    label = f"top=shape  bottom=pattern  kind={pattern_kind}"
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    drw.text((10, ref_h + pat_h + 8), label, fill=(220, 220, 220), font=font)
    # Swatches
    if colors:
        sw_x = 10
        for c in colors[:5]:
            hex_rgb = c["hex"].lstrip("#")
            rgb = (int(hex_rgb[0:2], 16), int(hex_rgb[2:4], 16), int(hex_rgb[4:6], 16))
            drw.rectangle([sw_x, ref_h + pat_h + 36, sw_x + 80, ref_h + pat_h + 72], fill=rgb)
            drw.text((sw_x + 4, ref_h + pat_h + 38),
                     f"{c['hex']} {c['fraction']:.0%}", fill=(240, 240, 240),
                     font=font)
            sw_x += 90
    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# =====================================================================
# 4. End-to-end helper.
# =====================================================================

async def extract_if_patterned(client, model: str, ref_bytes: bytes,
                                 *, seed: int = 42) -> dict | None:
    """Returns None when the gate says NO. Otherwise:
       {grid_bytes, pattern_bytes, pattern_kind, colors, source}
       where source = 'chutes' or 'pil_fallback'."""
    gate = await has_complex_pattern(client, model, ref_bytes)
    log.info(f"pattern gate: has={gate.get('has_pattern')} kind={gate.get('pattern_kind')}")
    if not gate.get("has_pattern"):
        return None
    pat_bytes = await _chutes_extract(ref_bytes, seed=seed)
    source = "chutes"
    if pat_bytes is None or len(pat_bytes) < 1000:
        log.info("pattern: falling back to PIL center crop")
        pat_bytes = _pil_center_crop(ref_bytes)
        source = "pil_fallback"
    try:
        colors = analyze_colors(pat_bytes, k=5)
    except Exception as e:
        log.warning(f"color analysis fail: {e}")
        colors = []
    kind = gate.get("pattern_kind", "")
    grid = compose_grid(ref_bytes, pat_bytes, colors=colors, pattern_kind=kind)
    return {
        "grid_bytes": grid,
        "pattern_bytes": pat_bytes,
        "pattern_kind": kind,
        "colors": colors,
        "source": source,
        "gate_reason": gate.get("reason", ""),
    }

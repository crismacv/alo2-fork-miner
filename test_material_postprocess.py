"""Post-process material variant selection.

For each cached `ours JS` from a previous OpenRouter test, apply material
tweaks (metalness/clearcoat/iridescence/emissive), render every variant, judge
each against the reference, then pick the lowest-penalty render as our final.

This is essentially a cheap "best-of-N on materials only" ensemble that costs
just K extra renders + K judge calls (no LLM coder calls).
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "qwen_pipeline"))
ENV = ROOT.parent.parent / ".env"
for line in ENV.read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("mat_pp")

import render as render_mod
import judge
import judge_critic
import jsfix
import material_tweak as mt

REFS = ROOT.parent / "competition_db/refs"
LEADER_R7 = ROOT.parent / "competition_db/leader_r7"
CACHE_DIR = Path("/tmp")


async def score_variant(ref: bytes, js: str, seed: int = 42) -> tuple[int, bytes | None]:
    js_fixed = jsfix.fix(js)
    img = await render_mod.render_front(js_fixed)
    if img is None:
        return (10, None)
    crit = await judge_critic.critique(ref, img, seed=seed)
    if not crit:
        return (10, img)
    return (int(crit.get("penalty", 10)), img)


async def pick_best(ref: bytes, base_js: str) -> tuple[str, str, int, bytes | None]:
    """Return (variant_name, js, penalty, render_bytes)."""
    variants = {"orig": base_js}
    for name, fn in mt.VARIANTS.items():
        try:
            v = fn(base_js)
            variants[name] = v
        except Exception:
            continue
    log.info(f"  scoring {len(variants)} variants...")
    results = await asyncio.gather(*[
        score_variant(ref, js) for js in variants.values()
    ])
    best = None
    for (name, _js), (pen, img) in zip(variants.items(), results):
        log.info(f"    {name}: penalty={pen}")
        if best is None or pen < best[2]:
            best = (name, _js, pen, img)
    return best


async def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/tmp/openrouter_test_r7_pp1_mv0.json")
    args = p.parse_args()
    prev = json.loads(Path(args.input).read_text())

    new_results = []
    for i, x in enumerate(prev, 1):
        stem = x["stem"]
        if x.get("result") in (None, "MISSING", "CODER_FAIL", "RENDER_FAIL"):
            new_results.append(x); continue
        ref_path = REFS / f"{stem}.png"
        if not ref_path.exists():
            new_results.append(x); continue
        ref = ref_path.read_bytes()
        # Recover our_js — we didn't cache it earlier, so rebuild via re-render of leader for comparison.
        # Instead just compare leader render with variants from prior leader_render? Not the same.
        # For this experiment we'll just use leader's JS to validate the pipeline first.
        log.info(f"[{i}/{len(prev)}] {stem[:12]} starting (re-running coder NOT — pipeline check)")
        leader_js = (LEADER_R7 / f"{stem}.js").read_text()
        # Apply mat tweaks to leader's own JS, see if any improves
        name, best_js, best_pen, best_img = await pick_best(ref, leader_js)
        # Also score original
        orig_pen, _ = await score_variant(ref, leader_js)
        log.info(f"[{i}/{len(prev)}] {stem[:12]}: orig_pen={orig_pen} best={name} pen={best_pen}")
        new_results.append({"stem": stem, "orig_pen": orig_pen,
                            "best_variant": name, "best_pen": best_pen})

    out = Path("/tmp/material_pp_results.json")
    out.write_text(json.dumps(new_results, indent=2))

    improved = sum(1 for r in new_results if "best_pen" in r and r["best_pen"] < r["orig_pen"])
    same = sum(1 for r in new_results if "best_pen" in r and r["best_pen"] == r["orig_pen"])
    worse = sum(1 for r in new_results if "best_pen" in r and r["best_pen"] > r["orig_pen"])
    log.info(f"=== SUMMARY === improved:{improved} same:{same} worse:{worse}")
    for r in new_results:
        if "best_pen" in r:
            delta = r["orig_pen"] - r["best_pen"]
            sign = "+" if delta > 0 else ("-" if delta < 0 else "=")
            log.info(f"  {r['stem'][:12]}  orig:{r['orig_pen']} best:{r['best_pen']} ({sign}{abs(delta)} via {r['best_variant']})")


if __name__ == "__main__":
    asyncio.run(main())

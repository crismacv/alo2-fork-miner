"""Test the alo2 fork end-to-end using OpenRouter (qwen3.5-397b-a17b).

Bypasses the FastAPI service: directly instantiates the SceneCoderAgent and
exercises the same code path the pipeline would.

Compares against the R7 leader (alo2 itself) on a small batch of R7 prompts to
see if our preprocess + multi-view + top-3 idioms make a real difference.
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
sys.path.insert(0, str(ROOT / "pipeline_service"))
sys.path.insert(0, str(ROOT.parent / "qwen_pipeline"))

# Load .env
ENV = ROOT.parent.parent / ".env"
for line in ENV.read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("openrouter_test")

# alo2 imports
from llm.session_store import SessionStore
from modules.scene_coder.agent import SceneCoderAgent
from modules.qwen_edit import edit as qwen_edit
from modules.multi_view import augment as multi_view
from openai import AsyncOpenAI

# Local render/judge (reuse our qwen_pipeline)
import render as render_mod
import judge
import judge_critic
import jsfix
import material_tweak as mt

OR_BASE = "https://openrouter.ai/api/v1"
OR_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "qwen/qwen3.5-397b-a17b"

REFS = ROOT.parent / "competition_db/refs"
LEADER_R7 = ROOT.parent / "competition_db/leader_r7"
LEADER_R5 = ROOT.parent / "competition_db/leader_r5"


async def run_one(stem: str, *, use_preprocess: bool, use_multiview: bool) -> dict:
    ref_path = REFS / f"{stem}.png"
    leader_path = LEADER_R7 / f"{stem}.js"
    if not leader_path.exists():
        leader_path = LEADER_R5 / f"{stem}.js"
    if not ref_path.exists() or not leader_path.exists():
        return {"stem": stem, "result": "MISSING"}
    ref = ref_path.read_bytes()
    leader_js = leader_path.read_text()

    t0 = time.time()
    img = ref
    if use_preprocess:
        os.environ["QWEN_EDIT_ENABLED"] = "1"
        img = await qwen_edit(img)
    if use_multiview:
        os.environ["MULTI_VIEW_ENABLED"] = "1"
        img = await multi_view(img, seed=42)
    log.info(f"[{stem[:12]}] preprocess+mv done in {time.time()-t0:.1f}s, img bytes={len(img)}")

    # Build coder agent (single image, no OSD since planner disabled)
    client = AsyncOpenAI(base_url=OR_BASE, api_key=OR_KEY, timeout=240)
    session_store = SessionStore()
    coder = SceneCoderAgent(
        client=client, model=MODEL, session_store=session_store,
        temperature=0.0, seed=42, max_tokens=8192, backend="openrouter",
        total_stages=6,
    )

    t1 = time.time()
    try:
        js_code = await coder.code(task_id=stem, image_bytes=img, image_mime="image/png")
    except Exception as e:
        log.exception(f"coder failed: {e}")
        return {"stem": stem, "result": "CODER_FAIL", "err": str(e)}
    log.info(f"[{stem[:12]}] coder done in {time.time()-t1:.1f}s, bytes={len(js_code)}")

    # Material-tweak post-processing: render+judge orig and 4 variants; pick best.
    variants = {"orig": jsfix.fix(js_code)}
    for name, fn in mt.VARIANTS.items():
        try:
            variants[name] = jsfix.fix(fn(js_code))
        except Exception:
            continue
    renders = await asyncio.gather(*[render_mod.render_front(v) for v in variants.values()])
    crits = await asyncio.gather(*[
        judge_critic.critique(ref, r, seed=42) if r is not None else asyncio.sleep(0, result=None)
        for r in renders
    ])
    best_name = "orig"; best_pen = 10; best_js = variants["orig"]; best_render = renders[0]
    for (name, js), r, c in zip(variants.items(), renders, crits):
        if r is None:
            continue
        pen = int(c.get("penalty", 10)) if c else 10
        log.info(f"[{stem[:12]}]   {name}: penalty={pen}")
        if pen < best_pen:
            best_pen = pen; best_name = name; best_js = js; best_render = r
    log.info(f"[{stem[:12]}] picked {best_name} (penalty={best_pen})")

    leader_render = await render_mod.render_front(leader_js)
    if best_render is None or leader_render is None:
        return {"stem": stem, "result": "RENDER_FAIL",
                "ours_ok": best_render is not None,
                "leader_ok": leader_render is not None}
    pa, pb = await judge.duel(ref, best_render, leader_render, seed=42)
    verdict = "WIN" if pa < pb else ("LOSS" if pa > pb else "DRAW")
    log.info(f"[{stem[:12]}] ours={pa} leader={pb} → {verdict} (total {time.time()-t0:.1f}s, variant={best_name})")
    return {"stem": stem, "result": verdict, "ours": pa, "leader": pb,
            "best_variant": best_name, "dt": time.time() - t0}


async def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--no-preprocess", action="store_true")
    p.add_argument("--no-multiview", action="store_true")
    p.add_argument("--source", choices=["r7", "r5"], default="r7")
    args = p.parse_args()

    pool = LEADER_R7 if args.source == "r7" else LEADER_R5
    stems = sorted(p.stem for p in pool.glob("*.js"))[:args.n]
    log.info(f"running {len(stems)} prompts from {args.source}, "
             f"preprocess={not args.no_preprocess}, multiview={not args.no_multiview}")

    out_path = Path(f"/tmp/openrouter_test_{args.source}_pp{int(not args.no_preprocess)}_mv{int(not args.no_multiview)}.json")
    if out_path.exists():
        cached = {r["stem"]: r for r in json.loads(out_path.read_text())}
    else:
        cached = {}
    results = []
    for i, stem in enumerate(stems, 1):
        if stem in cached and cached[stem].get("result") not in (None, "CODER_FAIL"):
            log.info(f"[{i}/{len(stems)}] {stem[:12]} cached → {cached[stem]['result']}")
            results.append(cached[stem]); continue
        log.info(f"[{i}/{len(stems)}] {stem[:12]} starting")
        r = await run_one(stem, use_preprocess=not args.no_preprocess,
                          use_multiview=not args.no_multiview)
        results.append(r)
        out_path.write_text(json.dumps(results, indent=2))

    w = sum(1 for r in results if r.get("result") == "WIN")
    d = sum(1 for r in results if r.get("result") == "DRAW")
    l_ = sum(1 for r in results if r.get("result") == "LOSS")
    f_ = sum(1 for r in results if r.get("result") in ("MISSING", "CODER_FAIL", "RENDER_FAIL"))
    log.info(f"=== SUMMARY === W:{w} D:{d} L:{l_} F:{f_} (total {len(results)})")


if __name__ == "__main__":
    asyncio.run(main())
